"""Export online EWM or independent-evaluation MARE for all 20 Formamide geometries."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]


def write_csv(path, header, rows):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)
    temporary.replace(path)


def export(run_dir, kind='online', output=None):
    run_dir = run_dir.expanduser().resolve()
    output = output.expanduser().resolve() if output else run_dir / 'analysis'
    reference_path = REPO / 'data/references/formamide_bbmep_reference_energy.csv'
    with reference_path.open() as stream:
        rows = sorted((r for r in csv.DictReader(stream) if r['MEP_system'] == 'Formamide'),
                      key=lambda r: int(r['Image_ID']))
    assert [int(r['Image_ID']) for r in rows] == list(range(20))
    reference = np.array([float(r['Reference_E(Hartree)']) for r in rows])
    centered_reference = reference - reference.mean()
    source = run_dir / 'training/result.h5'
    key = 'E_loc/mean/ewm' if kind == 'online' else 'E_loc/mean_elec'
    if kind == 'evaluation':
        config = yaml.safe_load((run_dir / 'training/config.yaml').read_text())
        assert config['args']['test'], 'Independent evaluation requires a run launched with --test.'
    with h5py.File(source, 'r', swmr=True, libver='v110') as stream:
        indices, energy = stream['metrics/mol_idx'], stream['metrics/' + key]
        indices.refresh()
        energy.refresh()
        count = min(len(indices), len(energy))
        if kind == 'online':
            errors = stream['metrics/E_loc/mean/ewm_err']
            errors.refresh()
            count = min(count, len(errors))
        idx, values = indices[:count].astype(int), energy[:count]
        ewm_errors = errors[:count] if kind == 'online' else None
    assert count > 0 and idx.shape == values.shape
    assert np.all((idx >= 0) & (idx < 20))
    output.mkdir(parents=True, exist_ok=True)
    result = dict(kind=kind, logged_global_steps=count, geometry_count=20,
                  unit='kcal/mol', source=str(source), reference=str(reference_path),
                  reference_sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
                  conversion_factor=627.5,
                  exported_at_utc=datetime.now(timezone.utc).isoformat())
    if kind == 'online':
        current = np.full(20, np.nan)
        current_error = np.full(20, np.nan)
        curve = []
        for step, (which, batch, uncertainty) in enumerate(zip(idx, values, ewm_errors, strict=True), 1):
            current[which] = batch
            current_error[which] = uncertainty
            if np.isfinite(current).all():
                error = current - current.mean() - centered_reference
                curve.append((step, float(np.abs(error).mean() * 627.5)))
        if not curve or curve[-1][0] != count:
            raise RuntimeError('All 20 geometries must have finite logged energies; a one-geometry smoke run is insufficient.')
        result.update(MARE_online_EWM_kcal_per_mol=curve[-1][1],
                      interpretation='Training EWM diagnostic; not independent fixed-checkpoint evaluation.',
                      refresh='Snapshot; --watch refreshes every 60 seconds.')
        write_csv(output / 'mare_online.csv', ['optimizer_updates', 'MARE_online_EWM_kcal_per_mol'], curve)
        write_csv(output / 'energy_online_latest.csv',
                  ['geometry_id', 'E_training_EWM_Ha', 'EWM_error_Ha',
                   'E_reference_Ha', 'centered_relative_error_kcal_per_mol'],
                  [(i, current[i], current_error[i], reference[i], error[i] * 627.5) for i in range(20)])
        summary = output / 'mare_online_latest.json'
    else:
        from scipy.optimize import minimize
        current = np.empty(20)
        samples = []
        for geometry in range(20):
            observed = values[idx == geometry]
            observed = observed[np.isfinite(observed)]
            assert len(observed), f'Geometry {geometry} has no finite evaluation values.'
            samples.append(len(observed))
            # Same Huber objective and delta=1 Ha as upstream robust_mean().
            def objective(x):
                residual = observed - x
                large = np.abs(residual) > 1.0
                return (0.5 * residual[~large] ** 2).sum() + (np.abs(residual[large]) - 0.5).sum()
            def derivative(x):
                return np.array([-np.clip(observed - x, -1.0, 1.0).sum()])
            optimum = minimize(objective, observed.mean(), jac=derivative)
            if not optimum.success:
                raise RuntimeError(f'Robust mean did not converge for geometry {geometry}: {optimum.message}')
            current[geometry] = optimum.x.item()
        error = current - current.mean() - centered_reference
        result.update(MARE_evaluation_kcal_per_mol=float(np.abs(error).mean() * 627.5),
                      finite_evaluation_steps_per_geometry=samples,
                      energy_estimator='Huber robust mean of logged per-geometry electron-batch means, delta=1 Ha')
        summary = output / 'mare_evaluation.json'
    if kind == 'evaluation':
        write_csv(output / 'energies_evaluation.csv',
                  ['geometry_id', 'E_Ha', 'reference_E_Ha', 'centered_relative_error_kcal_per_mol'],
                  [(i, current[i], reference[i], error[i] * 627.5) for i in range(20)])
    temporary = summary.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(summary)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--kind', choices=['online', 'evaluation'], default='online')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--watch', action='store_true', help='Refresh online outputs every 60 seconds.')
    args = parser.parse_args()
    if args.watch and args.kind != 'online':
        parser.error('--watch is only supported for online training diagnostics.')
    while True:
        try:
            export(args.run_dir, args.kind, args.output)
        except Exception as error:
            if not args.watch:
                raise
            print(f'Export retry: {type(error).__name__}: {error}', flush=True)
        if not args.watch:
            break
        time.sleep(60)


if __name__ == '__main__':
    main()
