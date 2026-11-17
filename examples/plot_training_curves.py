"""Plot completed Ethane online MARE and H4 training-energy convergence."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
BLUE = '#245777'


def trailing_mean(values, window):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not 1 <= window <= len(values) or not np.isfinite(values).all():
        raise ValueError('Expected a finite vector and a valid rolling window.')
    sums = np.r_[0.0, np.cumsum(values)]
    result = np.full(values.shape, np.nan)
    result[window - 1:] = (sums[window:] - sums[:-window]) / window
    return result


def sorted_geometry_energies(indices, energies, count):
    indices = np.asarray(indices)
    energies = np.asarray(energies, dtype=np.float64)
    if indices.shape != energies.shape or indices.ndim != 2:
        raise ValueError('Geometry indices and energies must have matching two-dimensional shapes.')
    if not np.equal(indices, indices.astype(int)).all():
        raise ValueError('Geometry indices are not integers.')
    order = np.argsort(indices, axis=1)
    if not np.equal(np.take_along_axis(indices, order, axis=1), np.arange(count)).all():
        raise ValueError('Each logged update must contain every geometry exactly once.')
    if not np.isfinite(energies).all():
        raise ValueError('Non-finite training energies encountered.')
    return np.take_along_axis(energies, order, axis=1)


def write_csv(path, columns, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


def save_figure(fig, stem):
    for suffix in ('.png', '.pdf'):
        fig.savefig(stem.with_suffix(suffix), dpi=240, bbox_inches='tight', facecolor='white')


def style_axis(ax, last_step):
    ax.set_xlim(0, last_step)
    ax.set_xlabel('Training steps')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: '0' if x == 0 else f'{x / 1000:g}k'))
    ax.tick_params(direction='out', length=4, width=1)
    ax.grid(axis='y', color='#d0d0d0', linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)


def ethane_panel(ax, steps, mare, seed=None):
    label = 'Orbformer (scratch)' if seed is None else f'Orbformer (scratch, seed {seed})'
    ax.plot(steps, mare, color=BLUE, linewidth=1.3, label=label)
    ax.axhline(1.0, color='#757575', linestyle='--', linewidth=1,
               label=r'1 kcal mol$^{-1}$')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    ax.set_ylabel(r'MARE (kcal mol$^{-1}$)')
    ax.set_title('Ethane (20 geometries)')
    ax.set_xticks([0, 8000, 16000, 24000, 32000])
    style_axis(ax, int(steps[-1]))
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.plot(steps[-1], mare[-1], 'o', color=BLUE, markersize=3)
    ax.annotate(f'{mare[-1]:.3f}', (steps[-1], mare[-1]), xytext=(-8, 12),
                textcoords='offset points', ha='right', fontsize=10, color=BLUE)


def h4_panel(ax, steps, energy, raw_energy=None, seed=None):
    if raw_energy is not None:
        ax.plot(steps, raw_energy, color='#999999', linewidth=0.6, alpha=0.45,
                label='Unsmoothed six-geometry mean')
    label = 'Orbformer (scratch)' if seed is None else f'Orbformer (scratch, seed {seed})'
    ax.plot(steps, energy, color=BLUE, linewidth=1.5, label=label)
    ax.set_title(r'$\mathrm{H}_4(6)$')
    ax.set_ylabel(r'Mean energy ($E_h$)')
    ax.set_xticks([0, 8000, 16000])
    style_axis(ax, int(steps[-1]))
    ax.ticklabel_format(axis='y', style='plain', useOffset=False)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ethane-run', type=Path,
                        default=REPO / 'runs/ethane_scratch_4gpu_32k_s219/results')
    parser.add_argument('--h4-run', type=Path,
                        default=REPO / 'runs/h4_globe6_scratch_1gpu_16k_s42/results')
    parser.add_argument('--output-dir', type=Path, default=REPO / 'runs/analysis_ethane_s219_h4_s42')
    parser.add_argument('--window', type=int, default=200)
    args = parser.parse_args()
    ethane, h4 = args.ethane_run.resolve(), args.h4_run.resolve()
    combined = args.output_dir.resolve()
    combined.mkdir(parents=True, exist_ok=True)
    ethane_output, h4_output = ethane / 'analysis', h4 / 'analysis'
    ethane_output.mkdir(exist_ok=True)
    h4_output.mkdir(exist_ok=True)
    for run in (ethane, h4):
        if json.loads((run / 'runtime_status.json').read_text())['stage'] != 'COMPLETE':
            raise ValueError(f'This figure exporter expects a completed training run: {run}')

    mare_path = ethane_output / 'mare_online.csv'
    with mare_path.open() as stream:
        rows = list(csv.DictReader(stream))
    ethane_steps = np.array([int(r['optimizer_updates']) for r in rows])
    mare = np.array([float(r['MARE_online_EWM_kcal_per_mol']) for r in rows])
    with h5py.File(ethane / 'training/result.h5', 'r') as stream:
        ethane_count = len(stream['metrics/mol_idx'])
    if ethane_steps[-1] != ethane_count or not np.isfinite(mare).all() or np.any(mare <= 0):
        raise ValueError('Refresh the complete online MARE CSV before plotting.')

    with h5py.File(h4 / 'training/result.h5', 'r') as stream:
        indices = stream['metrics/mol_idx'][:]
        raw = sorted_geometry_energies(indices, stream['metrics/E_loc/mean_elec'][:], 6)
    h4_steps = np.arange(1, len(raw) + 1)
    mean_energy = raw.mean(axis=1)
    smoothed = trailing_mean(mean_energy, args.window)
    h4_config = yaml.safe_load((h4 / 'training/config.yaml').read_text())
    assert not h4_config['args']['test']
    ethane_config = yaml.safe_load((ethane / 'training/config.yaml').read_text())
    h4_seed, ethane_seed = h4_config['args']['seed'], ethane_config['args']['seed']

    write_csv(h4_output / 'h4_convergence.csv',
              ['optimizer_updates', 'mean_training_energy_Ha',
               f'mean_energy_trailing_{args.window}_Ha'],
              [(int(step), float(energy), float(avg) if np.isfinite(avg) else '')
               for step, energy, avg in zip(h4_steps, mean_energy, smoothed, strict=True)])
    write_csv(h4_output / 'energy_by_geometry.csv', ['optimizer_updates', *[f'E_geometry_{i:02d}_Ha' for i in range(6)]],
              [(int(step), *map(float, energy)) for step, energy in zip(h4_steps, raw, strict=True)])

    plt.rcParams.update({'font.family': 'DejaVu Serif', 'mathtext.fontset': 'stix',
                         'font.size': 11, 'axes.linewidth': 1.05,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, ax = plt.subplots(figsize=(5.0, 3.8))
    ethane_panel(ax, ethane_steps, mare, ethane_seed)
    fig.text(0.5, 0.01, 'Training EWM estimate; not independent checkpoint evaluation.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save_figure(fig, ethane_output / 'mare_online')
    plt.close(fig)

    for full_range in (False, True):
        fig, ax = plt.subplots(figsize=(4.2, 3.8))
        h4_panel(ax, h4_steps, smoothed, mean_energy if full_range else None, h4_seed)
        note = f'{args.window}-step trailing mean; absolute energy in Hartree.'
        fig.text(0.5, 0.03, note, ha='center', fontsize=7.7)
        fig.text(0.5, 0.0, 'All six training geometries; no external reference energy.', ha='center', fontsize=7.7)
        fig.tight_layout(rect=[0, 0.08, 1, 1])
        save_figure(fig, h4_output / ('h4_convergence_fullrange' if full_range else 'h4_convergence'))
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8))
    ethane_panel(axes[0], ethane_steps, mare, ethane_seed)
    h4_panel(axes[1], h4_steps, smoothed, seed=h4_seed)
    fig.text(0.27, 0.025, 'Online EWM MARE; not independent evaluation.', ha='center', fontsize=8)
    fig.text(0.77, 0.025, f'{args.window}-step mean; absolute energy in Hartree.', ha='center', fontsize=8)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    save_figure(fig, combined / 'ethane_h4_training_curves')
    plt.close(fig)

    metadata = dict(exported_at_beijing=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        ethane=dict(seed=ethane_seed, steps=int(ethane_steps[-1]), final_online_MARE_kcal_per_mol=float(mare[-1]),
                    source=str(ethane / 'training/result.h5'), data_csv=str(mare_path),
                    data_sha256=hashlib.sha256(mare_path.read_bytes()).hexdigest(),
                    interpretation='Training EWM, reference-centered MARE; not independent evaluation.'),
        h4=dict(seed=h4_seed, steps=int(h4_steps[-1]), geometry_count=6, energy_source='metrics/E_loc/mean_elec',
                source=str(h4 / 'training/result.h5'), smoothing=f'trailing {args.window} optimizer updates, full windows only',
                averaging='Equal-weight mean of the six per-geometry electron-batch means at each update.',
                energy_unit='Hartree', energy_shift_Ha=0.0,
                final_smoothed_energy_Ha=float(smoothed[-1]),
                final_unsmoothed_mean_energy_Ha=float(mean_energy[-1]),
                interpretation='Absolute mean training energy over six geometries; not independent checkpoint evaluation.',
                primary_plot_scale='linear, automatic limits, no axis offset',
                fullrange_plot='Absolute unsmoothed mean plus trailing mean; linear, automatic limits'),
        reproducible_command='python examples/plot_training_curves.py --window ' + str(args.window))
    (combined / 'training_curves_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (h4_output / 'h4_convergence_metadata.json').write_text(json.dumps(metadata['h4'], indent=2) + '\n')
    (combined / 'README.txt').write_text(
        'Ethane: full 32000-step online EWM MARE against the official 20-geometry reference curve.\n'
        f'H4: absolute equal-weight mean energy over the six training geometries, trailing {args.window}-step average, in Hartree.\n'
        'The H4 energy is displayed without subtracting a baseline or applying an axis offset.\n'
        f'The first {args.window - 1} H4 updates have no full smoothing window and are omitted from the smoothed line; raw data are retained.\n'
        'H4 fullrange view includes the unsmoothed mean from the first update as well as the smoothed curve.\n'
        'No Moon/FermiNet comparison curves or independent checkpoint evaluations are included.\n')
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
