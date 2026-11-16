"""Plot a snapshot of Formamide online MARE without importing JAX."""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.run_dir.expanduser().resolve() / 'analysis'
    with (output / 'mare_online.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    steps = np.array([int(r['optimizer_updates']) for r in rows])
    mare = np.array([float(r['MARE_online_EWM_kcal_per_mol']) for r in rows])
    if not len(rows) or not np.isfinite(mare).all() or np.any(mare < 0):
        raise ValueError('Export a finite online MARE curve covering all 20 geometries first.')
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(steps, mare, label='Orbformer-SE, scratch, seed 219')
    ax.axhline(1, color='gray', linestyle='--', label='1 kcal/mol')
    ax.set(xlabel='Optimizer updates', ylabel='Online EWM MARE (kcal/mol)',
           title=f'Formamide: 20 geometries, snapshot through {steps[-1]} updates')
    if np.all(mare > 0):
        ax.set_yscale('log')
    ax.grid(alpha=0.25)
    ax.legend()
    fig.text(0.5, 0.01, 'Training diagnostic; not independent checkpoint evaluation.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    for suffix in ('png', 'pdf'):
        fig.savefig(output / f'mare_online.{suffix}', dpi=200)
    plt.close(fig)
    print(f'Exported MARE figures through {steps[-1]} updates to {output}')


if __name__ == '__main__':
    main()
