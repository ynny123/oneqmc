"""Portable smoke/train/evaluate launcher for the documented Formamide setting."""
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
    spec = importlib.util.spec_from_file_location('formamide_upstream_args', REPO / 'scripts/args.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = argparse.ArgumentParser()
    module.add_transferable_args(parser)
    return parser.parse_args(command[1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'train', 'evaluate'], default='train')
    parser.add_argument('--output', type=Path, required=True,
                        help='Upstream -w directory; fresh runs require a new/empty directory.')
    parser.add_argument('--devices', type=int, help='Visible local device count (default: smoke 1, otherwise 4).')
    parser.add_argument('--steps', type=int, help='Override default 2/32000/30000 global iterations.')
    parser.add_argument('--checkpoint', type=Path, help='Required only for evaluate mode.')
    parser.add_argument('--resume', action='store_true', help='Resume this output directory with the same training target.')
    parser.add_argument('--dry-run', action='store_true', help='Parse and print the command without JAX initialization or file writes.')
    args = parser.parse_args()
    devices = args.devices if args.devices is not None else (1 if args.mode == 'smoke' else 4)
    steps = args.steps if args.steps is not None else {'smoke': 2, 'train': 32000, 'evaluate': 30000}[args.mode]
    if devices < 1 or steps < 1:
        parser.error('--devices and --steps must be positive')
    if args.mode == 'smoke' and devices != 1:
        parser.error('smoke uses one geometry on one device')
    if args.mode != 'smoke' and 20 % devices:
        parser.error('--devices must divide the 20-geometry dataset (1, 2, 4, 5, 10 or 20)')
    if args.mode == 'evaluate' and (args.checkpoint is None or args.resume):
        parser.error('evaluate requires --checkpoint and a fresh output directory')
    if args.mode != 'evaluate' and args.checkpoint is not None:
        parser.error('Scratch training does not accept an external checkpoint; use --resume for this run.')
    if args.mode == 'smoke' and args.resume:
        parser.error('Use a new output directory for each smoke run.')
    if int(os.environ.get('WORLD_SIZE', '1')) != 1:
        parser.error('This example uses one process on one node; do not launch it with torchrun/mpirun.')
    output = args.output.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint is not None else None
    data = sorted((REPO / 'data/bbmep/Formamide').glob('*.yaml'))
    assert [p.stem for p in data] == [f'{i:02d}' for i in range(20)]
    with (Path(__file__).parent / 'setting.json').open() as stream:
        setting = json.load(stream)
    command = [str(REPO / 'scripts/transferable.py'), *setting['common_args'],
               '--mol-batch-size', str(devices), '--train-steps', str(steps),
               '--max-eq-steps', '1' if args.mode == 'smoke' else '500', '-w', str(output)]
    if args.mode == 'smoke':
        command += ['--data-file-whitelist', '.*00[.]yaml$']
    if args.mode == 'evaluate':
        assert checkpoint.is_file(), checkpoint
        command += ['--test', '-c', str(checkpoint), '--discard-sampler-state', '--no-autoresume']
    upstream = parse_upstream(command)
    metadata = dict(mode=args.mode, output=str(output), devices=devices, geometry_count=20, electrons_per_configuration=24,
                    global_geometry_batch=devices, electron_configurations_per_geometry=256,
                    global_electronic_batch=devices * 256, target_global_steps=steps, seed=219,
                    initialization='checkpoint_evaluation' if checkpoint else ('own_checkpoint_resume' if args.resume else 'random'),
                    checkpoint=str(checkpoint) if checkpoint else None,
                    hf_pretrain_steps=upstream.pretrain_steps,
                    command=[sys.executable, *command], upstream_args=vars(upstream),
                    geometry_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in data})
    print(shlex.join([sys.executable, *command]), flush=True)
    if args.dry_run:
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
        return
    if args.resume:
        assert list((output / 'training').glob('chkpt-*.pt')), 'No checkpoint found in this run.'
    elif output.exists() and any(output.iterdir()):
        raise RuntimeError('Output directory is not empty. Choose a new directory, or --resume for an existing training run.')
    output.mkdir(parents=True, exist_ok=True)
    launch_name = ('launch_resume_' + datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '.json'
                   if args.resume else 'launch.json')
    save(output / launch_name, metadata)
    status = dict(stage='INITIALIZING', mode=args.mode, started_at=datetime.now(timezone.utc).isoformat())
    started = time.monotonic()
    try:
        import jax
        import numpy as np
        import yaml

        for path in data:
            molecule = yaml.safe_load(path.read_text())
            assert sum(molecule['charges']) - molecule['charge'] == 24
            assert molecule['spin'] == 0 and np.asarray(molecule['coords']).shape == (6, 3)
        assert jax.process_count() == 1
        if jax.default_backend() == 'cpu':
            raise RuntimeError('Only CPU devices are visible. Check the accelerator allocation and JAX installation.')
        assert jax.local_device_count() == devices, jax.devices()
        if args.resume:
            config = yaml.safe_load((output / 'training/config.yaml').read_text())
            previous = config['args']
            differences = {key: (previous.get(key), value) for key, value in vars(upstream).items()
                           if key != 'workdir' and previous.get(key) != value}
            if differences:
                raise RuntimeError(f'Resume configuration differs from the saved run: {differences}')
        from functools import partial
        @partial(jax.pmap, axis_name='formamide_devices')
        def collective(x):
            return jax.lax.psum(x, 'formamide_devices')
        check = collective(jax.numpy.arange(devices, dtype=jax.numpy.float32))
        np.testing.assert_allclose(check, devices * (devices - 1) / 2)
        status.update(stage='RUNNING', devices=[d.device_kind for d in jax.devices()],
                      jax_version=jax.__version__, backend=jax.default_backend(), collective_check='PASSED')
        save(output / 'runtime_status.json', status)
        os.chdir(REPO)
        # Keep the upstream AML environment hooks from redirecting this example's bundled data.
        os.environ['AMLT_DATA_DIR'] = str(REPO / 'data')
        sys.path[:0] = [str(REPO / 'scripts'), str(REPO / 'src')]
        sys.argv = command
        runpy.run_path(command[0], run_name='__main__')
        from oneqmc.entrypoint import load_chkpt_file
        final_path = output / 'training' / f'chkpt-{steps}.pt'
        if not final_path.is_file():
            final_path = output / 'training/chkpt--1.pt'
        state, completed = load_chkpt_file(str(final_path), False)
        assert completed == steps, completed
        assert args.mode == 'evaluate' or state.opt is not None
        leaves = jax.tree_util.tree_leaves(state.params)
        assert all(np.isfinite(np.asarray(x)).all() for x in leaves)
        changed = None
        if args.mode == 'smoke':
            initial, _ = load_chkpt_file(str(output / 'training/chkpt-0.pt'), False)
            changed = sum(not np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                          zip(jax.tree_util.tree_leaves(initial.params), leaves, strict=True))
            assert changed > 0 and state.opt is not None
        if args.mode == 'evaluate':
            initial, _ = load_chkpt_file(str(checkpoint), False)
            for a, b in zip(jax.tree_util.tree_leaves(initial.params), leaves, strict=True):
                np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        import h5py
        with h5py.File(output / 'training/result.h5', 'r') as stream:
            indices = stream['metrics/mol_idx'][:].astype(int)
            energies = stream['metrics/E_loc/mean_elec'][:]
            assert indices.shape == energies.shape == (steps, devices)
            assert np.isfinite(energies).all()
            if args.mode == 'smoke':
                assert np.all(indices == 0)
            else:
                assert sorted(np.unique(indices).tolist()) == list(range(20))
        status.update(stage='COMPLETE', completed_global_steps=completed,
                      geometry_batch=devices, electronic_batch_global=devices * 256,
                      finite_logged_energies=True, parameter_count=sum(np.asarray(x).size for x in leaves),
                      device_memory_stats=[d.memory_stats() or {} for d in jax.devices()],
                      final_checkpoint=str(final_path), checkpoint_reload='PASSED',
                      changed_parameter_tensors=changed, elapsed_seconds=time.monotonic() - started)
        save(output / 'runtime_status.json', status)
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    except BaseException as error:
        status.update(stage='FAILED', error_type=type(error).__name__, error=str(error),
                      elapsed_seconds=time.monotonic() - started)
        save(output / 'runtime_status.json', status)
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
