"""Export asynchronous per-geometry EWM relative errors; no GPU imports."""
import argparse
import csv
from datetime import datetime, timezone
import io
import hashlib
import json
from pathlib import Path
import time

import h5py
import numpy as np


def atomic(path, content):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(content)
    temporary.replace(path)


REPO = Path(__file__).resolve().parents[2]
REFERENCE = REPO / 'data/references/licn_reference_energy.csv'


def export(run_dir):
    source = run_dir / 'training/result.h5'
    if not source.exists():
        return
    with REFERENCE.open() as stream:
        records = list(csv.DictReader(stream))
    assert [int(r['geometry']) for r in records] == list(range(5))
    reference = np.array([float(r['reference_energy_hartree']) for r in records])
    try:
        with h5py.File(source, 'r', swmr=True) as stream:
            indices = stream['metrics/mol_idx']
            energies = stream['metrics/E_loc/mean/ewm']
            indices.refresh()
            energies.refresh()
            count = min(len(indices), len(energies))
            mol_idx = indices[:count].astype(int)
            energy = energies[:count]
    except (OSError, KeyError):
        return
    latest = np.full(5, np.nan)
    seen_at = np.full(5, -1, dtype=int)
    rows = []
    for step, (ids, values) in enumerate(zip(mol_idx, energy)):
        for idx, value in zip(np.ravel(ids), np.ravel(values)):
            if not 0 <= idx < 5 or not np.isfinite(value):
                continue
            latest[idx] = value
            seen_at[idx] = step
        if np.isfinite(latest).all():
            residual = (latest - latest.mean()) - (reference - reference.mean())
            rows.append((step, step + 1, 627.5 * np.abs(residual).mean(), 5))
    if not rows:
        return
    output = run_dir / 'analysis'
    output.mkdir(exist_ok=True)
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['step', 'completed_updates', 'mare_kcal_mol', 'geometries_seen'])
    writer.writerows(rows)
    atomic(output / 'mare_online.csv', table.getvalue())
    table = io.StringIO()
    writer = csv.writer(table)
    writer.writerow(['geometry', 'last_step', 'ewm_energy_hartree', 'reference_energy_hartree'])
    writer.writerows(zip(range(5), seen_at.tolist(), latest.tolist(), reference.tolist()))
    atomic(output / 'energy_online_latest.csv', table.getvalue())
    atomic(output / 'mare_online_latest.json', json.dumps(dict(
        updated_at=datetime.now(timezone.utc).isoformat(), last_step=rows[-1][0],
        completed_updates=rows[-1][1], mare_kcal_mol=rows[-1][2],
        kind='online EWM diagnostic; latest observation per geometry; not independent evaluation',
        reference_source=str(REFERENCE), reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest(),
        source_h5=str(source)), indent=2) + '\n')
    print(f'Exported through {count} updates; MARE={rows[-1][2]:.6f} kcal/mol', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True,
                        help='Training output directory passed to upstream -w.')
    parser.add_argument('--watch', action='store_true')
    args = parser.parse_args()
    while True:
        try:
            export(args.run_dir.expanduser().resolve())
        except Exception as error:
            if not args.watch:
                raise
            print(f'Export retry: {type(error).__name__}: {error}', flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
