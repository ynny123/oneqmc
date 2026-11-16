"""Check the external benchmark's geometry convention and native loader units."""
import importlib.util
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('h4_generator', REPO / 'examples/h4_rectangle/generate.py')
generator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generator)


def test_rectangle_dataset_and_loader(tmp_path):
    from oneqmc.convert_geo import load_molecules
    hashes = generator.generate(tmp_path)
    assert len(hashes) == 6 and generator.generate(tmp_path) == hashes
    molecules = load_molecules(str(tmp_path))
    assert len(molecules) == 6
    for mol in molecules:
        assert mol.n_up == mol.n_down == 2 and mol.n_nuc == 4
        np.testing.assert_allclose(mol.coords.mean(0), 0, atol=1e-14)
        np.testing.assert_allclose(np.linalg.norm(mol.coords, axis=1), 3.2843, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(mol.coords[0] - mol.coords[3]), 6.5686, atol=1e-12)
        assert abs(np.dot(mol.coords[0] - mol.coords[1], mol.coords[0] - mol.coords[2])) < 1e-12
    # Values calculated from the author's theta=85, R=3.2843 convention, in bohr.
    first = yaml.safe_load((tmp_path / '00.yaml').read_text())
    np.testing.assert_allclose(first['coords'][0], [2.4214399572854903, 2.2188409188721128, 0], atol=1e-12)
    last = yaml.safe_load((tmp_path / '05.yaml').read_text())
    assert last['extras']['theta_deg'] == 89.999
    assert last['coords'][0][0] != last['coords'][0][1]


def test_generator_protects_existing_data(tmp_path):
    import pytest
    generator.generate(tmp_path)
    path = tmp_path / '00.yaml'
    path.write_text('different data\n')
    with pytest.raises(FileExistsError):
        generator.generate(tmp_path)
    assert path.read_text() == 'different data\n'
