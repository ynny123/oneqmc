import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'training_curves', Path(__file__).resolve().parents[1] / 'examples/plot_training_curves.py')
curves = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curves)


def test_energy_average_uses_geometry_ids_and_full_trailing_windows():
    ids = np.array([[2, 0, 1], [1, 2, 0], [0, 1, 2], [2, 1, 0]])
    energies = np.array([[30, 10, 20], [21, 31, 11], [12, 22, 32], [33, 23, 13]])
    ordered = curves.sorted_geometry_energies(ids, energies, 3)
    np.testing.assert_array_equal(ordered, [[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33]])
    mean = curves.trailing_mean(ordered.mean(axis=1), 2)
    assert np.isnan(mean[0])
    np.testing.assert_allclose(mean[1:], [20.5, 21.5, 22.5])


def test_incomplete_geometry_rows_and_invalid_smoothing_are_rejected():
    with pytest.raises(ValueError):
        curves.sorted_geometry_energies([[0, 0, 2]], [[1, 2, 3]], 3)
    with pytest.raises(ValueError):
        curves.trailing_mean([1.0, np.nan], 2)
    with pytest.raises(ValueError):
        curves.trailing_mean([1.0, 2.0], 3)
