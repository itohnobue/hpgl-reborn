"""Regression tests for sgs.py fixes (s8-fix-py-sim).

Covers:
- II-18  SGS LVM equal-volume per-dimension mean shape validation (mirrors
         lvm_kriging R-13 guard, geo.py:1919-1927)
- III-13 SGS mask binary-semantics validation (Python counts mask!=0, C++
         gates mask==1 — non-binary masks must fail loudly)
- III-14 create_ubyte_array must preserve the caller's mask shape so the
         C++ per-dimension shape guard can fire
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.cdf import CdfData
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        SugarboxGrid,
        covariance,
    )
    from geo_bsd.sgs import sgs_simulation

    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


def _cov():
    return CovarianceModel(
        type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
    )


def _cdf():
    return CdfData(
        values=np.array([0.0, 0.5, 1.0], dtype="float32"),
        probs=np.array([0.3, 0.7, 1.0], dtype="float32"),
    )


@pytest.mark.hpgl
class TestSgsLvmShapeValidation:
    """II-18: SGS LVM mean per-dimension shape must match the grid.

    The C++ LVM provider consumes the mean buffer by flat node index, so a
    (2,2,2) mean on a (1,2,4) grid (both volume 8) silently permutes the
    mean field. Mirror the lvm_kriging R-13 guard (geo.py:1919-1927).
    """

    def test_wrong_3d_mean_shape_raises(self):
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_wrong = np.random.RandomState(2).rand(2, 2, 2).astype("float32")
        with pytest.raises(ValueError, match="3D LVM mean shape"):
            sgs_simulation(prop, grid, _cdf(), (1, 1, 1), 4, _cov(),
                           seed=42, mean=mean_wrong)

    def test_correct_3d_mean_succeeds(self):
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_ok = np.random.RandomState(4).rand(1, 2, 4).astype("float32")
        out = sgs_simulation(prop, grid, _cdf(), (1, 1, 1), 4, _cov(),
                             seed=42, mean=mean_ok)
        assert np.all(np.isfinite(out.data))

    def test_flat_same_volume_mean_still_succeeds(self):
        """1D (flat) LVM mean vectors are covered by the size check and carry
        no per-dimension meaning — must not be rejected by the 3D guard."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        prop = ContProperty(
            np.random.RandomState(0).rand(8).astype("float32"),
            np.ones(8, dtype="uint8"),
        )
        mean_flat = np.random.RandomState(5).rand(8).astype("float32")
        out = sgs_simulation(prop, grid, _cdf(), (1, 1, 1), 4, _cov(),
                             seed=42, mean=mean_flat)
        assert np.all(np.isfinite(out.data))


@pytest.mark.hpgl
class TestSgsMaskSemantics:
    """Mask semantics contract: non-zero = informed (got-20260803180153).

    The C++ SGS kernel gates simulation on mask[node] == 1
    (sequential_simulation.h:124) while the Python expected-cell count uses
    mask != 0. The centralized normalization (ffi_adapter.normalize_mask_binary)
    converts any non-zero mask to binary 1 at the boundary, so a mask value
    like 2 is counted as simulate by Python AND simulated by C++ — both sides
    agree after normalization instead of the old silent-skip or loud-reject.
    """

    def _prop_grid(self):
        grid = SugarboxGrid(x=2, y=2, z=2)
        data = np.random.RandomState(0).rand(8).astype("float32")
        mask = np.ones(8, dtype="uint8")
        mask[0] = 0  # one uninformed cell so SGS actually simulates
        return grid, ContProperty(data, mask)

    def test_non_binary_mask_is_normalized_not_rejected(self):
        grid, prop = self._prop_grid()
        bad_mask = np.ones(8, dtype="uint8")
        bad_mask[1] = 2  # non-binary value — C++ gates on == 1
        out = sgs_simulation(prop, grid, _cdf(), (1, 1, 1), 4, _cov(),
                             seed=42, mask=bad_mask)
        assert np.all(np.isfinite(out.data))
        # The normalized contract: a mask value of 2 is informed (non-zero)
        # and must be simulated, not silently skipped.
        from geo_bsd.ffi_adapter import normalize_mask_binary

        normalized = normalize_mask_binary(bad_mask, "test")
        assert set(np.unique(normalized)) <= {0, 1}
        assert normalized[1] == 1

    def test_binary_mask_still_succeeds(self):
        grid, prop = self._prop_grid()
        good_mask = np.ones(8, dtype="uint8")
        out = sgs_simulation(prop, grid, _cdf(), (1, 1, 1), 4, _cov(),
                             seed=42, mask=good_mask)
        assert np.all(np.isfinite(out.data))


@pytest.mark.hpgl
class TestSgsMaskShapePreserved:
    """III-14: create_ubyte_array must preserve the caller's 3D mask shape so
    the C++ per-dimension shape guard (validate_simulation_mask_shape_or_throw,
    api.cpp:197-217) can fire on an equal-volume per-dim mismatch."""

    def test_mask_struct_preserves_3d_shape(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        mask3d = np.ones((2, 8, 1), dtype="uint8", order="F")  # volume 16

        class _G:
            x, y, z = 4, 4, 1

        ub = create_ubyte_array(mask3d, _G())
        assert tuple(ub.shape.m_data) == (2, 8, 1), (
            "III-14: create_ubyte_array must preserve the caller's 3D mask "
            "shape so the C++ per-dim guard can fire"
        )

    def test_flat_mask_uses_grid_dims(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        flat = np.ones(16, dtype="uint8", order="F")

        class _G:
            x, y, z = 4, 4, 1

        ub = create_ubyte_array(flat, _G())
        assert tuple(ub.shape.m_data) == (4, 4, 1)

    def test_volume_mismatch_still_raises(self):
        from geo_bsd.ffi_adapter import create_ubyte_array

        small = np.ones(8, dtype="uint8", order="F")

        class _G:
            x, y, z = 4, 4, 1

        with pytest.raises(RuntimeError, match="Invalid data size"):
            create_ubyte_array(small, _G())
