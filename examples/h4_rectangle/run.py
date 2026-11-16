"""Run the six-geometry H4 scratch experiment on one accelerator."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import runpy
import shlex
import sys
import time
import traceback

REPO = Path(__file__).resolve().parents[2]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def parse_upstream(command):
    spec = importlib.util.spec_from_file_location('h4_upstream_args', REPO / 'scripts/args.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.add_transferable_args(parser)
    return parser.parse_args(command[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['train', 'smoke'], default='train')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mol-batch-size', type=int, choices=[1, 2, 3, 6], default=6)
    parser.add_argument('--steps', type=int)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.mode == 'smoke' and args.resume:
        parser.error('Each smoke check must use a new output directory.')
    steps = args.steps if args.steps is not None else (2 if args.mode == 'smoke' else 16000)
    if steps < 2:
        parser.error('--steps must be at least 2 for the upstream learning-rate schedule.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('Use one Python process on one node.')
    output = args.output.expanduser().resolve()
    data = sorted((REPO / 'data/h4_globe6').glob('*.yaml'))
    if [p.stem for p in data] != [f'{i:02d}' for i in range(6)]:
        parser.error('Generate data/h4_globe6/00.yaml through 05.yaml first.')
    settings = json.loads((Path(__file__).parent / 'setting.json').read_text())
    command = [str(REPO / 'scripts/transferable.py'), *settings['common_args'],
               '--mol-batch-size', str(args.mol_batch_size), '--train-steps', str(steps),
               '--max-eq-steps', '1' if args.mode == 'smoke' else '500', '-w', str(output)]
    upstream = parse_upstream(command)
    assert upstream.chkpt is None and upstream.pretrain_steps == 0
    metadata = dict(mode=args.mode, initialization='own_checkpoint_resume' if args.resume else 'random',
                    pretrained_checkpoint=None, hf_pretrain_steps=0, gpu_count=1,
                    geometry_count=6, geometry_batch=args.mol_batch_size,
                    electron_configurations_per_geometry=upstream.electron_batch_size,
                    electronic_batch_global=args.mol_batch_size * upstream.electron_batch_size,
                    target_updates=steps, seed=upstream.seed, command=[sys.executable, *command],
                    upstream_args=vars(upstream),
                    geometry_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data})
    print(shlex.join(metadata['command']), flush=True)
    if args.dry_run:
        print(json.dumps(metadata, indent=2))
        return
    if args.resume:
        if not list((output / 'training').glob('chkpt-*.pt')):
            parser.error('No checkpoint found in this output directory.')
    elif output.exists() and any(output.iterdir()):
        parser.error('Output directory is not empty; use a new directory or --resume.')
    output.mkdir(parents=True, exist_ok=True)
    status = dict(stage='INITIALIZING', mode=args.mode, initialization=metadata['initialization'],
                  started_at=datetime.now(timezone.utc).isoformat(), geometry_batch=args.mol_batch_size,
                  electronic_batch_global=metadata['electronic_batch_global'], target_updates=steps)
    started = time.monotonic()
    save(output / 'runtime_status.json', status)
    save(output / ('launch_resume.json' if args.resume else 'launch.json'), metadata)
    try:
        sys.path.insert(0, str(REPO / 'src'))
        import jax
        import numpy as np
        import yaml
        from oneqmc.molecule import Molecule

        assert jax.process_count() == 1 and jax.local_device_count() == 1, jax.devices()
        assert jax.default_backend() != 'cpu', 'No accelerator is visible.'
        for path in data:
            molecule = Molecule.from_dict(yaml.safe_load(path.read_text()))
            assert molecule.n_nuc == 4 and molecule.n_up == molecule.n_down == 2
            np.testing.assert_allclose(np.linalg.norm(molecule.coords, axis=1), 3.2843, atol=1e-12)
        if args.resume:
            previous = yaml.safe_load((output / 'training/config.yaml').read_text())['args']
            differences = {key: (previous.get(key), value) for key, value in vars(upstream).items()
                           if key != 'workdir' and previous.get(key) != value}
            if differences:
                raise ValueError(f'Resume configuration differs from saved run: {differences}')
        status.update(stage='TRAINING', devices=[d.device_kind for d in jax.devices()],
                      jax_version=jax.__version__, backend=jax.default_backend())
        save(output / 'runtime_status.json', status)
        os.environ['AMLT_DATA_DIR'] = str(REPO / 'data')
        os.environ['ORBFORMER_PICKLE_LOADING'] = '1'
        os.chdir(REPO)
        sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
        from oneqmc.entrypoint import load_chkpt_file
        final_path = output / 'training' / f'chkpt-{steps}.pt'
        if not final_path.is_file():
            final_path = output / 'training/chkpt--1.pt'
        state, completed = load_chkpt_file(str(final_path), False)
        assert completed == steps and state.opt is not None
        leaves = jax.tree_util.tree_leaves(state.params)
        assert leaves and all(np.isfinite(np.asarray(x)).all() for x in leaves)
        changed = None
        if args.mode == 'smoke':
            initial, initial_step = load_chkpt_file(str(output / 'training/chkpt-0.pt'), False)
            assert initial_step == 0
            changed = sum(not np.array_equal(np.asarray(a), np.asarray(b))
                          for a, b in zip(jax.tree_util.tree_leaves(initial.params), leaves, strict=True))
            assert changed > 0
        import h5py
        with h5py.File(output / 'training/result.h5', 'r') as stream:
            indices = stream['metrics/mol_idx'][:].astype(int)
            energies = stream['metrics/E_loc/mean_elec'][:]
            assert indices.shape == energies.shape and np.isfinite(energies).all()
            if args.mol_batch_size == 6:
                assert all(sorted(row.tolist()) == list(range(6)) for row in indices)
        status.update(stage='COMPLETE', completed_updates=completed, final_checkpoint=str(final_path),
                      checkpoint_reload='PASSED', changed_parameter_tensors=changed,
                      parameter_count=sum(np.asarray(x).size for x in leaves),
                      finite_logged_energies=True, elapsed_seconds=time.monotonic() - started,
                      device_memory_stats=[d.memory_stats() or {} for d in jax.devices()])
        save(output / 'runtime_status.json', status)
        print(json.dumps(status, indent=2), flush=True)
    except BaseException as error:
        status.update(stage='FAILED', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        save(output / 'runtime_status.json', status)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
