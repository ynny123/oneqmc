"""Check the 24-electron benchmark and complete-path relative-energy metric."""
import csv
import importlib.util
from pathlib import Path

import h5py
import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('formamide_mare', REPO / 'examples/formamide_scratch/mare.py')
mare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mare)


def reference_values():
    with (REPO / 'data/references/formamide_bbmep_reference_energy.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row['Image_ID']) for row in rows] == list(range(20))
    assert all(row['MEP_system'] == 'Formamide' and row['source'].startswith('https://github.com/microsoft/oneqmc/blob/') for row in rows)
    return np.array([float(row['Reference_E(Hartree)']) for row in rows])


def test_formamide_geometry_and_reference_alignment():
    data = sorted((REPO / 'data/bbmep/Formamide').glob('*.yaml'))
    assert [p.stem for p in data] == [f'{i:02d}' for i in range(20)]
    for path in data:
        mol = yaml.safe_load(path.read_text())
        assert sum(mol['charges']) - mol['charge'] == 24 and mol['spin'] == 0
        assert sorted(mol['charges']) == [1] * 3 + [6, 7, 8]
        assert np.asarray(mol['coords']).shape == (6, 3) and mol['unit'] == 'angstrom'
    first, last = [yaml.safe_load(data[i].read_text()) for i in (0, -1)]
    distances = [np.linalg.norm(np.array(m['coords'][2]) - m['coords'][4]) for m in (first, last)]
    np.testing.assert_allclose(distances, [1.0126306070, 5.9657800325], atol=1e-9)
    assert np.isfinite(reference_values()).all()


def test_online_mare_requires_coverage_and_ignores_global_energy_shift(tmp_path):
    ref = reference_values()
    indices = np.vstack([np.arange(20).reshape(5, 4), [0, 1, 2, 3]])
    energy = ref[indices] + 10.0
    energy[-1] += 0.01
    training = tmp_path / 'training'
    training.mkdir()
    source = training / 'result.h5'
    with h5py.File(source, 'w', libver='v110') as stream:
        stream['metrics/mol_idx'] = indices
        stream['metrics/E_loc/mean/ewm'] = energy
        stream['metrics/E_loc/mean/ewm_err'] = np.zeros_like(energy)
    result = mare.export(tmp_path)
    rows = list(csv.DictReader((tmp_path / 'analysis/mare_online.csv').open()))
    assert [int(row['optimizer_updates']) for row in rows] == [5, 6]
    assert float(rows[0]['MARE_online_EWM_kcal_per_mol']) == pytest.approx(0, abs=1e-9)
    # Four errors of +0.008 Ha and sixteen of -0.002 Ha after centering.
    assert result['MARE_online_EWM_kcal_per_mol'] == pytest.approx(2.008, abs=1e-9)
    with h5py.File(source, 'w', libver='v110') as stream:
        stream['metrics/mol_idx'] = indices[:1]
        stream['metrics/E_loc/mean/ewm'] = energy[:1]
        stream['metrics/E_loc/mean/ewm_err'] = np.zeros_like(energy[:1])
    with pytest.raises(RuntimeError, match='All 20 geometries'):
        mare.export(tmp_path)
