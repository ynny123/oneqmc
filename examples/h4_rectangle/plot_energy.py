"""Export the H4 absolute-energy curve independently of other experiments."""
import argparse
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('training_curves', REPO / 'examples/plot_training_curves.py')
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=REPO / 'runs/h4_globe6_scratch_1gpu_16k_s42/results')
    parser.add_argument('--window', type=int, default=200)
    args = parser.parse_args()
    run = args.run_dir.expanduser().resolve()
    if json.loads((run / 'runtime_status.json').read_text())['stage'] != 'COMPLETE':
        raise ValueError('Wait for this H4 training run to complete before exporting the final curve.')
    config = yaml.safe_load((run / 'training/config.yaml').read_text())['args']
    assert not config['test']
    with h5py.File(run / 'training/result.h5', 'r') as stream:
        raw = curves.sorted_geometry_energies(stream['metrics/mol_idx'][:], stream['metrics/E_loc/mean_elec'][:], 6)
    steps = np.arange(1, len(raw) + 1)
    mean = raw.mean(axis=1)
    smoothed = curves.trailing_mean(mean, args.window)
    output = run / 'analysis'
    output.mkdir(exist_ok=True)
    curves.write_csv(output / 'h4_convergence.csv',
                     ['optimizer_updates', 'mean_training_energy_Ha', f'mean_energy_trailing_{args.window}_Ha'],
                     [(int(step), float(energy), float(avg) if np.isfinite(avg) else '')
                      for step, energy, avg in zip(steps, mean, smoothed, strict=True)])
    curves.write_csv(output / 'energy_by_geometry.csv',
                     ['optimizer_updates', *[f'E_geometry_{i:02d}_Ha' for i in range(6)]],
                     [(int(step), *map(float, energy)) for step, energy in zip(steps, raw, strict=True)])
    for full_range in (False, True):
        fig, ax = curves.plt.subplots(figsize=(5, 4))
        curves.h4_panel(ax, steps, smoothed, mean if full_range else None, config['seed'])
        fig.text(.5, .015, f'{args.window}-step mean; absolute energy in Hartree.', ha='center', fontsize=8)
        fig.tight_layout(rect=[0, .05, 1, 1])
        curves.save_figure(fig, output / ('h4_convergence_fullrange' if full_range else 'h4_convergence'))
        curves.plt.close(fig)
    metadata = dict(seed=config['seed'], steps=len(raw), geometry_count=6, energy_unit='Hartree',
                    energy_shift_Ha=0.0, smoothing=f'trailing {args.window} updates, full windows only',
                    final_smoothed_energy_Ha=float(smoothed[-1]),
                    energy_source='metrics/E_loc/mean_elec', source=str(run / 'training/result.h5'))
    (output / 'h4_convergence_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
