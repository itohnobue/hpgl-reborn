"""Regression tests for sis.py fixes (s8-fix-py-sim).

Covers:
- II-18  SIS LVM equal-volume per-dimension marginal-prob shape validation
         (mirrors lvm_kriging R-13 guard, geo.py:1919-1927)
- III-13 SIS mask binary-semantics validation (Python counts mask!=0, C++
         gates mask==1 — non-binary masks must fail loudly)
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
    )
    from geo_bsd.sis import sis_simulation

    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


def _cov():
    return CovarianceModel(
        type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
    )


def _sis_data():
    return [
        {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
        {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
    ]


@pytest.mark.hpgl
class TestSisLvmShapeValidation:
    """II-18: SIS LVM marginal-prob per-dimension shape must match the grid.

    The C++ LVM provider consumes the marginal-prob buffer by flat node
    index, so a (2,2,2) marginal_probs[i] on a (1,2,4) grid (both volume 8)
    silently permutes the probability field. Mirror the lvm_kriging R-13
    guard (geo.py:1919-1927).
    """

    def _prop_grid(self):
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8").reshape(1, 2, 4)
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = IndProperty(data, mask, 2)
        return grid, prop

    def test_wrong_3d_marginal_shape_raises(self):
        grid, prop = self._prop_grid()
        wrong = np.full((2, 2, 2), 0.5, dtype="float32")
        with pytest.raises(ValueError, match="3D LVM marginal_probs"):
            sis_simulation(prop, grid, _sis_data(), seed=42,
                           marginal_probs=[wrong, wrong])

    def test_correct_3d_marginal_succeeds(self):
        grid, prop = self._prop_grid()
        ok = np.full((1, 2, 4), 0.5, dtype="float32")
        out = sis_simulation(prop, grid, _sis_data(), seed=42,
                             marginal_probs=[ok, ok])
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))

    def test_flat_same_volume_marginal_still_succeeds(self):
        """1D (flat) LVM vectors are covered by the size check and carry no
        per-dimension meaning — must not be rejected by the 3D guard."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8")
        mask = np.ones(8, dtype="uint8")
        prop = IndProperty(data, mask, 2)
        flat = np.full(8, 0.5, dtype="float32")
        out = sis_simulation(prop, grid, _sis_data(), seed=42,
                             marginal_probs=[flat, flat])
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))


@pytest.mark.hpgl
class TestSisMaskSemantics:
    """Mask semantics contract: non-zero = informed (got-20260803180153).

    The C++ SIS kernel gates simulation on mask[node] == 1
    (sequential_indicator_simulation.cpp:114) while the Python expected-cell
    count uses mask != 0. The centralized normalization
    (ffi_adapter.normalize_mask_binary) converts any non-zero mask to binary
    1 at the boundary so both sides agree after normalization.
    """

    def _prop_grid(self):
        grid = SugarboxGrid(x=2, y=2, z=2)
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8")
        mask = np.ones(8, dtype="uint8")
        mask[0] = 0
        return grid, IndProperty(data, mask, 2)

    def test_non_binary_mask_is_normalized_not_rejected(self):
        grid, prop = self._prop_grid()
        bad_mask = np.ones(8, dtype="uint8")
        bad_mask[1] = 2
        from geo_bsd.ffi_adapter import normalize_mask_binary

        normalized = normalize_mask_binary(bad_mask, "test")
        assert set(np.unique(normalized)) <= {0, 1}
        assert normalized[1] == 1
        out = sis_simulation(prop, grid, _sis_data(), seed=42,
                             marginal_probs=[0.5, 0.5], mask=bad_mask)
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))

    def test_binary_mask_still_succeeds(self):
        grid, prop = self._prop_grid()
        good_mask = np.ones(8, dtype="uint8")
        out = sis_simulation(prop, grid, _sis_data(), seed=42,
                             marginal_probs=[0.5, 0.5], mask=good_mask)
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))
