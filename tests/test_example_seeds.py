"""Verify the public seed42 defaults and the labels attached to exported data."""
import csv
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('folder,extra', [
    ('h4_rectangle', []), ('formamide_scratch', ['--devices', '4']),
    ('ethane_scratch', ['--devices', '4']),
])
def test_example_seed_and_dry_run_metadata(folder, extra, tmp_path):
    output = tmp_path / folder
    result = subprocess.run([sys.executable, str(REPO / f'examples/{folder}/run.py'),
                             '--mode', 'train', '--output', str(output), '--dry-run', *extra],
                            capture_output=True, text=True, check=True)
    launch = json.loads(result.stdout.split('\n', 1)[1])
    assert launch['seed'] == launch['upstream_args']['seed'] == 42
    assert launch['upstream_args']['chkpt'] is None
    assert launch['upstream_args']['pretrain_steps'] == 0
    assert not output.exists()


def test_h4_independent_plot_preserves_absolute_energy_and_seed(tmp_path):
    training = tmp_path / 'training'
    training.mkdir()
    (tmp_path / 'runtime_status.json').write_text(json.dumps({'stage': 'COMPLETE'}))
    (training / 'config.yaml').write_text(yaml.safe_dump({'args': {'test': False, 'seed': 42}}))
    indices = np.array([[3, 0, 5, 1, 4, 2]] * 4)
    ordered = np.arange(6) * 0.01 - 2.0 + np.arange(4)[:, None] * -0.001
    with h5py.File(training / 'result.h5', 'w') as stream:
        stream['metrics/mol_idx'] = indices
        stream['metrics/E_loc/mean_elec'] = np.take_along_axis(ordered, indices, axis=1)
    subprocess.run([sys.executable, str(REPO / 'examples/h4_rectangle/plot_energy.py'),
                    '--run-dir', str(tmp_path), '--window', '2'], check=True, capture_output=True)
    with (tmp_path / 'analysis/h4_convergence.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    np.testing.assert_allclose([float(r['mean_training_energy_Ha']) for r in rows], ordered.mean(axis=1))
    np.testing.assert_allclose([float(r['mean_energy_trailing_2_Ha']) for r in rows[1:]],
                               np.convolve(ordered.mean(axis=1), [0.5, 0.5], 'valid'))
    metadata = json.loads((tmp_path / 'analysis/h4_convergence_metadata.json').read_text())
    assert metadata['seed'] == 42 and metadata['energy_shift_Ha'] == 0
