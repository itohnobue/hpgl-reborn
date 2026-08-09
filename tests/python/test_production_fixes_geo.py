"""Regression tests for the geo.py production-fix pass (s6-fix-py-geo).

Covers the CONFIRMED geo.py findings from the stage-3/stage-5 synthesis
grids:

- H-2    simple_kriging/lvm_kriging equal-volume 3D shape validation
- M-9    median_ik/indicator_kriging singular-system RuntimeError
- M-16   slow parser ±1.0e21 sentinel window (fast/slow path agreement)
- M-19   simple_kriging_weights post-FFI isfinite validation
- M-26   documented exception contract (ValueError / CriticalValidationError)
- 2-M-15 property setters data/mask shape invariant
- 2-M-16 slow-parser line-amplification DoS hardening
- L-56   get_gslib_property ndarray type check
"""

import ast
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from geo_bsd import validation
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        covariance,
        get_gslib_property,
        indicator_kriging,
        load_cont_property,
        load_ind_property,
        lvm_kriging,
        median_ik,
        read_inc_file_byte,
        read_inc_file_float,
        simple_kriging,
        simple_kriging_weights,
        write_gslib_property,
        write_property,
    )

    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


def _cov():
    return CovarianceModel(
        type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
    )


# =============================================================================
# H-2: equal-volume per-dimension shape validation (simple_kriging / lvm_kriging)
# =============================================================================


@pytest.mark.hpgl
class TestEqualVolumeShapeValidation:
    """H-2: equal-volume shape mismatch must raise, not silently misread.

    A (2,2,2) prop with a (1,2,4) grid has equal volume (8) — pre-fix the
    C++ backend silently misread/miswrote the flat buffers with no error.
    """

    def test_simple_kriging_3d_shape_mismatch_raises(self):
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(0).rand(2, 2, 2).astype("float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        prop = ContProperty(data, mask)
        with pytest.raises(ValueError, match="3D data shape"):
            simple_kriging(prop, grid, (1, 1, 1), 4, _cov())

    def test_lvm_kriging_3d_shape_mismatch_raises(self):
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(1).rand(2, 2, 2).astype("float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.random.RandomState(2).rand(8).astype("float32")
        with pytest.raises(ValueError, match="3D data shape"):
            lvm_kriging(prop, grid, mean_data, (1, 1, 1), 4, _cov())

    def test_flat_same_volume_still_succeeds(self):
        """1D (flat) properties are unaffected — size check covers them."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        prop = ContProperty(
            np.random.RandomState(0).rand(8).astype("float32"),
            np.ones(8, dtype="uint8"),
        )
        out = simple_kriging(prop, grid, (1, 1, 1), 4, _cov())
        assert out.data.shape == (8,)
        assert np.all(np.isfinite(out.data))

    def test_matching_3d_shape_succeeds(self):
        """Control: a matching (2,2,2) prop with a (2,2,2) grid is accepted."""
        grid = SugarboxGrid(x=2, y=2, z=2)
        data = np.random.RandomState(1).rand(2, 2, 2).astype("float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        prop = ContProperty(data, mask)
        out = simple_kriging(prop, grid, (1, 1, 1), 4, _cov())
        assert out.data.shape == (2, 2, 2)


# =============================================================================
# R-13 (H-2 residual): lvm_kriging mean_data equal-volume 3D shape validation
# =============================================================================


@pytest.mark.hpgl
class TestLvmMeanDataShapeValidation:
    """R-13: mean_data in lvm_kriging is consumed by flat node index in C++
    (mean_provider.h:38-43), so an equal-volume 3D shape mismatch — e.g. a
    (2,2,2) mean_data with a (1,2,4) grid — must raise rather than silently
    permute the mean field. 1D (flat) mean vectors are covered by the size
    check and carry no per-dimension meaning."""

    def test_wrong_3d_mean_data_shape_raises(self):
        """An equal-volume 3D shape mismatch on mean_data raises (pre-fix:
        only the size was checked — (2,2,2) volume 8 passed the size check
        against a (1,2,4) grid and the mean field was permuted silently)."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.random.RandomState(2).rand(2, 2, 2).astype("float32")
        with pytest.raises(ValueError, match="3D mean_data shape"):
            lvm_kriging(prop, grid, mean_data, (1, 1, 1), 4, _cov())

    def test_correct_3d_mean_data_succeeds(self):
        """Control: a matching 3D mean_data is accepted and kriges."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.random.RandomState(4).rand(1, 2, 4).astype("float32")
        out = lvm_kriging(prop, grid, mean_data, (1, 1, 1), 4, _cov())
        assert out.data.shape == (1, 2, 4)
        assert np.all(np.isfinite(out.data))

    def test_flat_mean_data_succeeds(self):
        """Control: a flat 1D mean_data of the right size is accepted
        (the existing size check covers it)."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.random.RandomState(5).rand(8).astype("float32")
        out = lvm_kriging(prop, grid, mean_data, (1, 1, 1), 4, _cov())
        assert out.data.shape == (1, 2, 4)
        assert np.all(np.isfinite(out.data))


# =============================================================================
# II-18 (s8-fix-py-sim): SGS/SIS LVM per-dimension shape validation
# =============================================================================
#
# lvm_kriging's R-13 guard (above) validates per-dimension shape on
# mean_data; SGS/SIS LVM paths consumed their per-dim fields by flat node
# index with only a volume check, so an equal-volume (2,2,2) field on a
# (1,2,4) grid silently permuted the mean/probability field. The s8 fix
# replicates the R-13 guard at the Python boundary of sgs_simulation and
# sis_simulation.


@pytest.mark.hpgl
class TestSimulationLvmShapeValidation:
    """II-18: SGS/SIS LVM equal-volume per-dim shape mismatch must raise."""

    def _sgs_cdf(self):
        from geo_bsd.cdf import CdfData

        return CdfData(
            values=np.array([0.0, 0.5, 1.0], dtype="float32"),
            probs=np.array([0.3, 0.7, 1.0], dtype="float32"),
        )

    def test_sgs_lvm_3d_mean_shape_mismatch_raises(self):
        from geo_bsd.sgs import sgs_simulation

        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_wrong = np.random.RandomState(2).rand(2, 2, 2).astype("float32")
        with pytest.raises(ValueError, match="3D LVM mean shape"):
            sgs_simulation(prop, grid, self._sgs_cdf(), (1, 1, 1), 4, _cov(),
                           seed=42, mean=mean_wrong)

    def test_sis_lvm_3d_marginal_shape_mismatch_raises(self):
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8").reshape(1, 2, 4)
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = IndProperty(data, mask, 2)
        sis_data = [
            {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
            {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
        ]
        wrong = np.full((2, 2, 2), 0.5, dtype="float32")
        with pytest.raises(ValueError, match="3D LVM marginal_probs"):
            sis_simulation(prop, grid, sis_data, seed=42,
                           marginal_probs=[wrong, wrong])

    def test_sgs_lvm_3d_mean_shape_match_succeeds(self):
        from geo_bsd.sgs import sgs_simulation

        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.random.RandomState(3).rand(1, 2, 4).astype("float32")
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = ContProperty(data, mask)
        mean_ok = np.random.RandomState(4).rand(1, 2, 4).astype("float32")
        out = sgs_simulation(prop, grid, self._sgs_cdf(), (1, 1, 1), 4, _cov(),
                             seed=42, mean=mean_ok)
        assert np.all(np.isfinite(out.data))

    def test_sis_lvm_3d_marginal_shape_match_succeeds(self):
        from geo_bsd.sis import sis_simulation

        grid = SugarboxGrid(x=1, y=2, z=4)
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8").reshape(1, 2, 4)
        mask = np.ones((1, 2, 4), dtype="uint8")
        prop = IndProperty(data, mask, 2)
        sis_data = [
            {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
            {"cov_model": _cov(), "radiuses": (1, 1, 1), "max_neighbours": 4},
        ]
        ok = np.full((1, 2, 4), 0.5, dtype="float32")
        out = sis_simulation(prop, grid, sis_data, seed=42,
                             marginal_probs=[ok, ok])
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))


# =============================================================================
# M-9: median_ik / indicator_kriging singular-system RuntimeError
# =============================================================================


@pytest.mark.hpgl
class TestIndicatorKrigingStatsFinalize:
    """M-9: wrappers consume C++ stats so singular systems raise RuntimeError."""

    def test_median_ik_singularity_raises(self, monkeypatch):
        from geo_bsd import geo

        real_stats = {
            "points_calculated": 0,
            "points_without_neighbours": 0,
            "points_singularity": 5,
            "mean": 1.0,
            "speed_nps": 0.0,
        }
        monkeypatch.setattr(geo, "get_kriging_stats", lambda: real_stats)

        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = IndProperty(
            np.array([0, 1, 0, 1], dtype="uint8"),
            np.array([1, 1, 0, 0], dtype="uint8"),
            2,
        )
        with pytest.raises(RuntimeError, match="singular"):
            median_ik(prop, grid, (0.5, 0.5), (1, 1, 1), 4, _cov())

    def test_indicator_kriging_singularity_raises(self, monkeypatch):
        from geo_bsd import geo

        real_stats = {
            "points_calculated": 0,
            "points_without_neighbours": 0,
            "points_singularity": 3,
            "mean": 1.0,
            "speed_nps": 0.0,
        }
        monkeypatch.setattr(geo, "get_kriging_stats", lambda: real_stats)

        grid = SugarboxGrid(x=2, y=2, z=2)
        prop = IndProperty(
            np.array([0, 1, 2, 0, 1, 2, 0, 1], dtype="uint8"),
            np.ones(8, dtype="uint8"),
            3,
        )
        data = [
            {"radiuses": (1, 1, 1), "max_neighbours": 4, "cov_model": _cov()}
            for _ in range(3)
        ]
        with pytest.raises(RuntimeError, match="singular"):
            indicator_kriging(prop, grid, data, (0.3, 0.3, 0.4))

    def test_median_ik_clean_call_populates_stats(self):
        """A clean median_ik call populates _last_kriging_stats (M-9 wiring)."""
        from geo_bsd import geo

        geo._last_kriging_stats = None
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = IndProperty(
            np.array([0, 1, 0, 1], dtype="uint8"),
            np.array([1, 1, 0, 0], dtype="uint8"),
            2,
        )
        out = median_ik(prop, grid, (0.5, 0.5), (1, 1, 1), 4, _cov())
        assert isinstance(out, IndProperty)
        assert geo._last_kriging_stats is not None, (
            "median_ik should populate _last_kriging_stats via _finalize_kriging_stats"
        )


# =============================================================================
# M-16: slow parser ±1.0e21 sentinel window
# =============================================================================


@pytest.mark.hpgl
class TestSlowParserSentinelWindow:
    """M-16: slow path applies the ±1.0e21 window like the C++ fast reader."""

    def test_slow_and_fast_mask_out_of_window_sentinel(self, tmp_path):
        # -2.0e21 is strictly outside the ±1.0e21 window — masked by both paths.
        data = np.array([10.0, 20.0, -2.0e21, 40.0], dtype="float32")
        mask = np.ones(4, dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "sentinel_window.inc")
        write_property(prop_data, filename, "sentinel", -99.0, basedir=str(tmp_path))

        slow = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast = read_inc_file_float(filename, -99.0, 4, basedir=str(tmp_path))

        np.testing.assert_array_equal(slow.mask, fast.mask)
        assert slow.mask[2] == 0, f"slow parser should mask sentinel, got {slow.mask}"
        assert fast.mask[2] == 0, f"fast parser should mask sentinel, got {fast.mask}"

    def test_slow_and_fast_agree_on_undefined_equality(self, tmp_path):
        """Control: exact undefined_value matches stay masked on both paths."""
        data = np.array([1.0, -99.0, 3.0, 4.0], dtype="float32")
        mask = np.array([1, 0, 1, 1], dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "sentinel_undefined.inc")
        write_property(prop_data, filename, "undef", -99.0, basedir=str(tmp_path))

        slow = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast = read_inc_file_float(filename, -99.0, 4, basedir=str(tmp_path))

        np.testing.assert_array_equal(slow.mask, fast.mask)
        np.testing.assert_array_equal(slow.mask, np.array([1, 0, 1, 1], dtype="uint8"))

    def test_float32_sentinel_edge_both_stacks_post_p01(self, tmp_path):
        """A-01: float32 1.0e21f (token 1.000000020E+21) is SENTINEL on BOTH stacks.

        P-01 (read_inc_file.cpp:395-401) moved the C++ fast reader's sentinel
        window comparison to float64. Pre-fix, the writer's own %.9E output
        "1.000000020E+21" for float32 1.0e21f was classified DATA by the C++
        reader (float32 window = 1.0000000200408773e21) but SENTINEL by every
        Python float64 reader. This pin asserts the POST-P-01 parity: the
        exact float32 edge masks on both stacks, and the in-window 9.99e20
        stays DATA on both. Reverting P-01 makes fast.mask[1] return 1
        (DATA) while slow.mask[1] stays 0 — the parity assert fails.
        """
        # float32(1.0e21) = 1.0000000200408773e21; the C++ %.9E writer emits
        # "1.000000020E+21" for it — the exact token that exposed P-01.
        data = np.array([10.0, 1.0e21, 9.99e20, 40.0], dtype="float32")
        mask = np.ones(4, dtype="uint8")
        prop_data = ContProperty(data, mask)

        filename = str(tmp_path / "sentinel_edge.inc")
        write_property(prop_data, filename, "edge", -99.0, basedir=str(tmp_path))

        # The C++ %.9E writer must emit the exact-edge token for float32
        # 1.0e21f (a %E-6 writer change would emit "1.000000E+21" and
        # silently re-mask the divergence — this pins the writer precision).
        contents = Path(filename).read_text(encoding="utf-8")
        assert "1.000000020E+21" in contents

        slow = load_cont_property(filename, -99.0, basedir=str(tmp_path))
        fast = read_inc_file_float(filename, -99.0, 4, basedir=str(tmp_path))

        np.testing.assert_array_equal(slow.mask, fast.mask)
        # float32 1.0e21f must classify SENTINEL (mask 0) on BOTH stacks.
        assert slow.mask[1] == 0, (
            f"slow parser should mask float32 1.0e21f, got {slow.mask}"
        )
        assert fast.mask[1] == 0, (
            f"fast parser should mask float32 1.0e21f, got {fast.mask}"
        )
        # In-window edge 9.99e20 stays DATA (mask 1) on both.
        assert slow.mask[2] == 1, f"slow parser should keep 9.99e20 DATA, got {slow.mask}"
        assert fast.mask[2] == 1, f"fast parser should keep 9.99e20 DATA, got {fast.mask}"


# =============================================================================
# M-19: simple_kriging_weights post-FFI isfinite validation
# =============================================================================


@pytest.mark.hpgl
class TestSimpleKrigingWeightsFiniteCheck:
    """M-19: near-singular dpotrs_ success with NaN weights must raise."""

    def test_nan_weights_raise(self, monkeypatch):
        from geo_bsd import geo

        def fake_ffi(center_point, n_x, n_y, n_z, count, cov_params_struct, weights):
            # Simulate a near-singular solve that returns rc == 0 but NaN weights.
            weights[:] = np.nan
            return 0

        monkeypatch.setattr(geo, "call_simple_kriging_weights", fake_ffi)

        with pytest.raises(RuntimeError, match="weights contain NaN or Inf"):
            simple_kriging_weights((0.0, 0.0, 0.0), [1.0], [1.0], [1.0])

    def test_finite_weights_returned(self, monkeypatch):
        from geo_bsd import geo

        def fake_ffi(center_point, n_x, n_y, n_z, count, cov_params_struct, weights):
            weights[:] = [0.25, 0.75]
            return 0

        monkeypatch.setattr(geo, "call_simple_kriging_weights", fake_ffi)

        weights = simple_kriging_weights(
            (0.0, 0.0, 0.0), [1.0, 2.0], [1.0, 2.0], [1.0, 2.0]
        )
        np.testing.assert_allclose(weights, [0.25, 0.75])


# =============================================================================
# M-26: kriging wrappers raise the documented exception types
# =============================================================================


@pytest.mark.hpgl
class TestKrigingExceptionContract:
    """M-26: docstrings now match actual behavior.

    Data-validation errors (empty data, size/shape mismatch, NaN) raise
    ValueError; parameter-validation errors raise CriticalValidationError;
    C++ computation failures raise RuntimeError.
    """

    def test_simple_kriging_empty_data_valueerror(self):
        grid = SugarboxGrid(x=2, y=2, z=1)
        empty = ContProperty(np.array([], dtype="float32"), np.array([], dtype="uint8"))
        with pytest.raises(ValueError, match="prop.data is empty"):
            simple_kriging(empty, grid, (1, 1, 1), 4, _cov())

    def test_simple_kriging_nan_data_valueerror(self):
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(np.arange(4, dtype="float32"), np.ones(4, dtype="uint8"))
        # NaN is rejected by the constructor, so inject via the setter.
        prop.data = np.array([1.0, np.nan, 3.0, 4.0], dtype="float32")
        with pytest.raises(ValueError, match="contains NaN or Inf"):
            simple_kriging(prop, grid, (1, 1, 1), 4, _cov())

    def test_simple_kriging_invalid_radius_critical(self):
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(np.arange(4, dtype="float32"), np.ones(4, dtype="uint8"))
        with pytest.raises(validation.CriticalValidationError):
            simple_kriging(prop, grid, (0, 0, 0), 4, _cov())

    def test_lvm_kriging_mean_data_not_ndarray_valueerror(self):
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(np.arange(4, dtype="float32"), np.ones(4, dtype="uint8"))
        with pytest.raises(ValueError, match="mean_data must be a numpy array"):
            lvm_kriging(prop, grid, [1.0, 2.0, 3.0, 4.0], (1, 1, 1), 4, _cov())


# =============================================================================
# 2-M-15: property setters data/mask shape invariant
# =============================================================================


@pytest.mark.hpgl
class TestPropertySetterShapeInvariant:
    """2-M-15: setters reject data/mask desynchronization (C++ mask OOB)."""

    def test_cont_data_setter_shape_mismatch_raises(self):
        prop = ContProperty(np.arange(8, dtype="float32"), np.ones(8, dtype="uint8"))
        with pytest.raises(ValueError, match="does not match mask shape"):
            prop.data = np.arange(6, dtype="float32")

    def test_cont_mask_setter_shape_mismatch_raises(self):
        prop = ContProperty(np.arange(8, dtype="float32"), np.ones(8, dtype="uint8"))
        with pytest.raises(ValueError, match="does not match data shape"):
            prop.mask = np.ones(6, dtype="uint8")

    def test_cont_setter_same_shape_succeeds(self):
        prop = ContProperty(np.arange(8, dtype="float32"), np.ones(8, dtype="uint8"))
        new_data = np.full(8, 5.0, dtype="float32")
        prop.data = new_data
        np.testing.assert_array_equal(prop.data, new_data)
        new_mask = np.zeros(8, dtype="uint8")
        prop.mask = new_mask
        np.testing.assert_array_equal(prop.mask, new_mask)

    def test_ind_data_setter_shape_mismatch_raises(self):
        prop = IndProperty(
            np.array([0, 1, 2, 0], dtype="uint8"), np.ones(4, dtype="uint8"), 3
        )
        with pytest.raises(ValueError, match="does not match mask shape"):
            prop.data = np.array([0, 1], dtype="uint8")

    def test_ind_mask_setter_shape_mismatch_raises(self):
        prop = IndProperty(
            np.array([0, 1, 2, 0], dtype="uint8"), np.ones(4, dtype="uint8"), 3
        )
        with pytest.raises(ValueError, match="does not match data shape"):
            prop.mask = np.ones(5, dtype="uint8")

    def test_fix_shape_preserves_invariant(self):
        """Internal reshape paths bypass the setters and keep shapes in sync."""
        grid = SugarboxGrid(x=1, y=2, z=4)
        prop = ContProperty(np.arange(8, dtype="float32"), np.ones(8, dtype="uint8"))
        prop.fix_shape(grid)
        assert prop.data.shape == (1, 2, 4)
        assert prop.mask.shape == (1, 2, 4)


# =============================================================================
# 2-M-16: slow-parser line-amplification DoS hardening
# =============================================================================


@pytest.mark.hpgl
class TestSlowParserLineDoSHardening:
    """2-M-16: crafted oversized lines abort with a clear error, not OOM."""

    def test_oversized_line_token_cap_rejected(self, monkeypatch, tmp_path):
        from geo_bsd import geo as geo_mod

        # Shrink the cap so the test stays fast; a single newline-free line
        # with more tokens than the cap must be rejected incrementally.
        monkeypatch.setattr(geo_mod, "_MAX_SLOW_PARSER_ELEMENTS", 100)
        filename = tmp_path / "oversized_line.inc"
        with open(filename, "w") as fh:
            fh.write("prop\n")
            fh.write(" ".join(["1.0"] * 500) + "\n")
            fh.write("/\n")

        with pytest.raises(MemoryError, match="exceeds"):
            geo_mod.load_cont_property(str(filename), -99.0, basedir=str(tmp_path))

    def test_oversized_line_bytes_rejected(self, monkeypatch, tmp_path):
        from geo_bsd import geo as geo_mod

        monkeypatch.setattr(geo_mod, "_MAX_SLOW_PARSER_LINE_BYTES", 100)
        filename = tmp_path / "oversized_line_bytes.inc"
        with open(filename, "w") as fh:
            fh.write("prop\n")
            fh.write(" ".join(["1.0"] * 40) + "\n")  # ~160 bytes > 100-byte cap

        with pytest.raises(MemoryError, match="line exceeds"):
            geo_mod.load_cont_property(str(filename), -99.0, basedir=str(tmp_path))

    def test_normal_line_still_parses(self, tmp_path):
        """Control: a normal multi-token line is unaffected."""
        filename = tmp_path / "normal_line.inc"
        with open(filename, "w") as fh:
            fh.write("prop\n")
            fh.write("1.0 2.0 3.0 4.0\n")
            fh.write("/\n")
        result = load_cont_property(str(filename), -99.0, basedir=str(tmp_path))
        np.testing.assert_array_equal(result.data, np.array([1.0, 2.0, 3.0, 4.0], dtype="float32"))
        np.testing.assert_array_equal(result.mask, np.ones(4, dtype="uint8"))


# =============================================================================
# L-56: get_gslib_property ndarray type check
# =============================================================================


@pytest.mark.hpgl
class TestGetGslibPropertyTypeCheck:
    """L-56: non-ndarray property values raise a clear TypeError."""

    def test_list_value_raises_type_error(self):
        with pytest.raises(TypeError, match="must be a numpy array"):
            get_gslib_property({"a": [1.0, 2.0]}, "a", -99.0)

    def test_ndarray_value_ok(self):
        arr = np.array([1.0, -99.0, 3.0], dtype="float32")
        prop, mask = get_gslib_property({"a": arr}, "a", -99.0)
        np.testing.assert_array_equal(prop, arr)
        np.testing.assert_array_equal(mask, np.array([1, 0, 1], dtype="uint8"))


# =============================================================================
# III-24 — Python 3.9 import guard (geo.py PEP 604 module annotation)
# =============================================================================


@pytest.mark.hpgl
class TestPy39ImportGuard:
    """III-24: geo.py must stay importable on Python 3.9 (declared-supported,
    requires-python >=3.9).

    Pre-fix, geo.py:163 carried a module-level PEP 604 union
    (``_last_kriging_stats: dict | None = None``) with NO
    ``from __future__ import annotations`` — the annotation is evaluated at
    import time on 3.9 (``type.__or__`` landed in 3.10) and ``import
    geo_bsd`` raised TypeError. There is no CI; this test is the durable
    guard the build smoke (build.sh wheel/dev gates) wires in.
    """

    def _geo_source(self):
        from geo_bsd import geo

        return Path(geo.__file__).read_text(encoding="utf-8")

    def test_future_annotations_present(self):
        """Postponed evaluation must be enabled so module-level PEP 604
        annotations are never evaluated at import time on 3.9."""
        import ast

        source = self._geo_source()
        tree = ast.parse(source)
        future = [
            n
            for n in tree.body
            if isinstance(n, ast.ImportFrom)
            and n.module == "__future__"
            and any(a.name == "annotations" for a in n.names)
        ]
        assert future, (
            "geo.py must import 'annotations' from __future__ — a module-level "
            "PEP 604 union (e.g. _last_kriging_stats: dict | None) is evaluated "
            "at import time on Python 3.9 and crashes the import"
        )

    def test_module_parses_under_py39_grammar(self):
        """The module must parse under the Python 3.9 grammar (guards against
        match statements / other 3.10+ syntax regressions that no 3.13-only
        test run would catch)."""
        import ast

        source = self._geo_source()
        ast.parse(source, filename="geo.py", feature_version=(3, 9))


# =============================================================================
# III-11 — ContProperty float64->float32 downcast overflow guard
# =============================================================================


@pytest.mark.hpgl
class TestContPropertyDowncastOverflow:
    """III-11: the ctor isfinite check must run on the STORED float32 array.

    Pre-fix the check ran on the float64 view, so 1e300 passed and was
    silently stored as inf after the float32 downcast; calc_mean then
    returned NaN with no error.
    """

    def test_ctor_rejects_float64_overflow_to_inf(self):
        """1e300 is finite in float64 but overflows to inf in float32 —
        must be rejected by the constructor (pre-fix: accepted silently)."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(
                np.array([1e300, 2.0, 3.0]),
                np.ones(3, dtype="uint8"),
            )

    def test_ctor_rejects_direct_inf(self):
        """Control: an explicit inf is still rejected (existing behavior)."""
        with pytest.raises(ValueError, match="NaN or Inf"):
            ContProperty(
                np.array([1.0, np.inf, 3.0]),
                np.ones(3, dtype="uint8"),
            )

    def test_normal_float64_downcast_accepted(self):
        """Control: values that survive the float32 downcast are accepted
        and stored as float32."""
        prop = ContProperty(np.array([1.5, 2.0, 3.0]), np.ones(3, dtype="uint8"))
        assert prop.data.dtype == np.float32
        np.testing.assert_allclose(prop.data, [1.5, 2.0, 3.0])

    def test_calc_mean_rejects_non_finite(self):
        """calc_mean must raise on non-finite informed data instead of
        returning NaN (pre-fix: returned nan with only a warning)."""
        from geo_bsd import geo

        prop = ContProperty(np.arange(4, dtype="float32"), np.ones(4, dtype="uint8"))
        # The ctor rejects NaN, so inject via the setter (the documented
        # escape hatch used by the kriging wrappers' own validation tests).
        prop.data = np.array([1.0, np.nan, 3.0, 4.0], dtype="float32")
        with pytest.raises(ValueError, match="calc_mean: prop.data contains NaN or Inf"):
            geo.calc_mean(prop)

    def test_calc_mean_still_returns_finite_mean(self):
        """Control: a finite property still computes its mean."""
        from geo_bsd import geo

        prop = ContProperty(np.arange(4, dtype="float32"), np.ones(4, dtype="uint8"))
        assert geo.calc_mean(prop) == 1.5


# =============================================================================
# F-23 + II-37 + II-36 — IndProperty setter invariant re-validation
# =============================================================================


@pytest.mark.hpgl
class TestIndPropertySetterInvariants:
    """F-23/II-37: the data setter must enforce the constructor's full
    validation (range + integrality + indicator-range invariant). II-36: the
    mask setter must re-validate the indicator-range invariant for
    newly-informed cells.

    Pre-fix the setters accepted out-of-range data ([0,1,5,1] with count=2),
    silently truncated fractional values (1.5 -> 1), silently wrapped
    out-of-range values (300 -> 44), and allowed unmasking a 255-sentinel
    cell into an informed out-of-range state the constructor rejects.
    """

    @staticmethod
    def _prop(count=2):
        return IndProperty(
            np.array([0, 1, 0, 1], dtype="uint8"),
            np.ones(4, dtype="uint8"),
            count,
        )

    def test_data_setter_rejects_out_of_range(self):
        """F-23: an informed out-of-range value must be rejected by the
        setter (pre-fix: accepted, then silently dropped by C++ equality
        matching)."""
        prop = self._prop(count=2)
        with pytest.raises(RuntimeError, match="outside of \\[0\\.\\.1\\] range"):
            prop.data = np.array([0, 1, 5, 1], dtype="uint8")

    def test_data_setter_rejects_fractional(self):
        """II-37: fractional values must be rejected BEFORE the uint8
        conversion (pre-fix: 1.5 silently truncated to 1)."""
        prop = self._prop()
        with pytest.raises(ValueError, match="must contain integer values"):
            prop.data = np.array([0.0, 1.5, 0.0, 1.0])

    def test_data_setter_rejects_overflow_wrap(self):
        """II-37: out-of-range floats must be rejected, not wrapped mod 256
        (pre-fix: 300 -> 44)."""
        prop = self._prop()
        with pytest.raises(ValueError, match="must be in \\[0, 255\\]"):
            prop.data = np.array([0, 300, 0, 1])

    def test_mask_setter_rejects_unmasking_sentinel(self):
        """II-36: unmasking a masked cell holding the 255 sentinel must be
        rejected (pre-fix: accepted silently, producing an informed
        out-of-range value)."""
        prop = IndProperty(
            np.array([0, 255, 0, 1], dtype="uint8"),
            np.array([1, 0, 1, 1], dtype="uint8"),
            2,
        )
        with pytest.raises(RuntimeError, match="outside of \\[0\\.\\.1\\] range"):
            prop.mask = np.ones(4, dtype="uint8")

    def test_setter_rejects_state_constructor_rejects(self):
        """The setter must reject exactly the state the constructor rejects —
        enforcement must be consistent across both entry points."""
        prop = self._prop(count=2)
        # Constructor rejects the identical final state (e19aa84 check).
        with pytest.raises(RuntimeError, match="outside of"):
            IndProperty(
                np.array([0, 1, 5, 1], dtype="uint8"),
                np.ones(4, dtype="uint8"),
                2,
            )
        # And so must the setter.
        with pytest.raises(RuntimeError, match="outside of"):
            prop.data = np.array([0, 1, 5, 1], dtype="uint8")

    def test_legal_setter_ops_still_work(self):
        """Control: legal in-range integral data and legal masks are
        unaffected."""
        prop = self._prop()
        prop.data = np.array([1, 0, 1, 0], dtype="uint8")
        np.testing.assert_array_equal(prop.data, [1, 0, 1, 0])
        prop.mask = np.array([1, 1, 0, 0], dtype="uint8")
        np.testing.assert_array_equal(prop.mask, [1, 1, 0, 0])


# =============================================================================
# II-38 — undefined_value must not collide with a mapped indicator value
# =============================================================================


@pytest.mark.hpgl
class TestUndefinedValueCollision:
    """II-38: writing an IndProperty with undefined_value == an indicator
    value must raise — informed cells written with that value read back as
    MASKED (silent round-trip data loss). Pre-fix both byte writers accepted
    the collision (live probe: 2 of 5 cells lost)."""

    def _prop(self):
        return IndProperty(
            np.array([0, 1, 0, 2], dtype="uint8"),
            np.ones(4, dtype="uint8"),
            3,
        )

    def test_write_property_rejects_collision(self, tmp_path):
        with pytest.raises(ValueError, match="also an indicator value"):
            write_property(
                self._prop(),
                str(tmp_path / "collide.inc"),
                "col",
                20,
                indicator_values=[10, 20, 30],
                basedir=str(tmp_path),
            )

    def test_write_gslib_property_rejects_collision(self, tmp_path):
        with pytest.raises(ValueError, match="also an indicator value"):
            write_gslib_property(
                self._prop(),
                str(tmp_path / "collide.gslib"),
                "col",
                20,
                indicator_values=[10, 20, 30],
                basedir=str(tmp_path),
            )

    def test_non_colliding_roundtrip_ok(self, tmp_path):
        """Control: a non-colliding undefined_value writes and reads back
        with all informed cells preserved."""
        filename = str(tmp_path / "ok.inc")
        write_property(
            self._prop(),
            filename,
            "col",
            99,
            indicator_values=[10, 20, 30],
            basedir=str(tmp_path),
        )
        loaded = read_inc_file_byte(
            filename, 99, 4, [10, 20, 30], basedir=str(tmp_path)
        )
        np.testing.assert_array_equal(loaded.mask, np.ones(4, dtype="uint8"))
        np.testing.assert_array_equal(loaded.data, [0, 1, 0, 2])


# =============================================================================
# II-35 — I/O wrappers must hold _hpgl_call_lock around the FFI call
# =============================================================================


@pytest.mark.hpgl
class TestIoWrappersHoldHpglCallLock:
    """II-35: the four file-I/O wrappers (write_property,
    write_gslib_property, read_inc_file_float, read_inc_file_byte) must hold
    _hpgl_call_lock during their FFI call + error-message read. The C++
    locale_keeper sets a process-global locale for the parse window; without
    serialization, concurrent I/O on a non-glibc platform (macOS) with a
    non-C locale silently mis-parses numbers AND permanently corrupts the
    process locale (live race reproduced).

    The probe monkeypatches the FFI call and checks that a DIFFERENT thread
    cannot acquire the lock while the FFI call runs (RLock reentrancy makes
    an in-thread acquire check meaningless).
    """

    @staticmethod
    def _lock_blocked_from_other_thread(geo_mod):
        """Return True if _hpgl_call_lock is held by the calling thread
        (a worker thread cannot acquire it)."""
        acquired = []

        def _try():
            ok = geo_mod._hpgl_call_lock.acquire(blocking=False)
            acquired.append(ok)
            if ok:
                geo_mod._hpgl_call_lock.release()

        t = threading.Thread(target=_try)
        t.start()
        t.join()
        return not acquired[0]

    def test_read_inc_file_float_holds_lock(self, monkeypatch, tmp_path):
        import geo_bsd.geo as geo_mod

        fpath = tmp_path / "lock.inc"
        fpath.write_text("col\n1.0 2.0 3.0 4.0\n/\n")
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_read_inc_file_float", spy)
        geo_mod.read_inc_file_float(str(fpath), -99.0, 4, basedir=str(tmp_path))
        assert seen.get("held") is True, (
            "read_inc_file_float FFI call must hold _hpgl_call_lock (II-35)"
        )

    def test_write_property_holds_lock(self, monkeypatch, tmp_path):
        import geo_bsd.geo as geo_mod

        prop = IndProperty(
            np.array([0, 1, 0, 2], dtype="uint8"),
            np.ones(4, dtype="uint8"),
            3,
        )
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_write_inc_file_byte", spy)
        geo_mod.write_property(
            prop,
            str(tmp_path / "lock2.inc"),
            "col",
            99,
            indicator_values=[10, 20, 30],
            basedir=str(tmp_path),
        )
        assert seen.get("held") is True, (
            "write_property (byte) FFI call must hold _hpgl_call_lock (II-35)"
        )

    def test_write_gslib_property_holds_lock(self, monkeypatch, tmp_path):
        import geo_bsd.geo as geo_mod

        prop = ContProperty(
            np.arange(4, dtype="float32"), np.ones(4, dtype="uint8")
        )
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_write_gslib_cont_property", spy)
        geo_mod.write_gslib_property(
            prop,
            str(tmp_path / "lock3.gslib"),
            "col",
            -99.0,
            basedir=str(tmp_path),
        )
        assert seen.get("held") is True, (
            "write_gslib_property (cont) FFI call must hold _hpgl_call_lock (II-35)"
        )

    def test_write_property_float_path_holds_lock(self, monkeypatch, tmp_path):
        """G-4: write_property's FLOAT path (ContProperty →
        call_write_inc_file_float, lock at geo.py:1079) must hold the lock —
        the II-35 pin family previously covered only the byte path."""
        import geo_bsd.geo as geo_mod

        prop = ContProperty(
            np.arange(4, dtype="float32"), np.ones(4, dtype="uint8")
        )
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_write_inc_file_float", spy)
        geo_mod.write_property(
            prop,
            str(tmp_path / "lock4.inc"),
            "col",
            -99.0,
            basedir=str(tmp_path),
        )
        assert seen.get("held") is True, (
            "write_property (float) FFI call must hold _hpgl_call_lock (II-35)"
        )

    def test_write_gslib_byte_path_holds_lock(self, monkeypatch, tmp_path):
        """G-4: write_gslib_property's BYTE path (IndProperty →
        call_write_gslib_byte_property, lock at geo.py:1175) must hold the
        lock — previously only the cont path was pinned."""
        import geo_bsd.geo as geo_mod

        prop = IndProperty(
            np.array([0, 1, 0, 2], dtype="uint8"),
            np.ones(4, dtype="uint8"),
            3,
        )
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_write_gslib_byte_property", spy)
        geo_mod.write_gslib_property(
            prop,
            str(tmp_path / "lock5.gslib"),
            "col",
            99,
            indicator_values=[10, 20, 30],
            basedir=str(tmp_path),
        )
        assert seen.get("held") is True, (
            "write_gslib_property (byte) FFI call must hold _hpgl_call_lock (II-35)"
        )

    def test_read_inc_file_byte_holds_lock(self, monkeypatch, tmp_path):
        """G-4: read_inc_file_byte (lock at geo.py:1443) must hold the lock —
        previously only the float reader was pinned."""
        import geo_bsd.geo as geo_mod

        fpath = tmp_path / "lockbyte.inc"
        fpath.write_text("col\n0 1 0 2\n/\n")
        seen = {}

        def spy(*args, **kwargs):
            seen["held"] = self._lock_blocked_from_other_thread(geo_mod)
            return 0

        monkeypatch.setattr(geo_mod, "call_read_inc_file_byte", spy)
        geo_mod.read_inc_file_byte(
            str(fpath), 99, 4, [10, 20, 30], basedir=str(tmp_path)
        )
        assert seen.get("held") is True, (
            "read_inc_file_byte FFI call must hold _hpgl_call_lock (II-35)"
        )


# =============================================================================
# III-12 — indicator_kriging 2-cat redirect honors the "data[1] ignored" contract
# =============================================================================


@pytest.mark.hpgl
class TestIndicatorKrigingMinimalData1:
    """III-12: the 2-category median_ik redirect dereferenced
    data[1]["radiuses"]/["cov_model"] in the log warning even though the
    validation above deliberately skips data[1] (validate_entries =
    data[:1]) — a minimal data[1] dict crashed with KeyError. The redirect
    must honor the documented "data[1] is ignored" contract."""

    def test_minimal_data1_dict_no_keyerror(self):
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = IndProperty(
            np.array([0, 1, 0, 1], dtype="uint8"),
            np.array([1, 1, 0, 0], dtype="uint8"),
            2,
        )
        cov = _cov()
        data_list = [
            {"cov_model": cov, "radiuses": (1, 1, 1), "max_neighbours": 4},
            {"cov_model": cov},  # minimal — no radiuses key
        ]
        # Pre-fix: KeyError 'radiuses' at the logger.warning dereference.
        out = indicator_kriging(prop, grid, data_list, (0.5, 0.5))
        assert out.indicator_count == 2

    def test_full_data1_redirect_still_works(self):
        """Control: a full data[1] dict behaves as before."""
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = IndProperty(
            np.array([0, 1, 0, 1], dtype="uint8"),
            np.array([1, 1, 0, 0], dtype="uint8"),
            2,
        )
        cov = _cov()
        data_list = [
            {"cov_model": cov, "radiuses": (1, 1, 1), "max_neighbours": 4},
            {"cov_model": cov, "radiuses": (1, 1, 1), "max_neighbours": 4},
        ]
        out = indicator_kriging(prop, grid, data_list, (0.5, 0.5))
        assert out.indicator_count == 2


# =============================================================================
# R-06 (F-54): sample-script TEST_DATA_DIR must resolve without a literal '..'
# =============================================================================
#
# F-54's fix built TEST_DATA_DIR as os.path.join(os.path.dirname(__file__),
# '..', 'tests', 'python', 'test_data') — a literal '..' component that
# geo_bsd's PathValidator rejects BEFORE normalization (validation.py:204-213,
# "Path traversal detected"). All 8 data-loading scripts crashed on their
# first load. R-06 wraps the construction in os.path.abspath() so the
# resolved path is absolute with no '..' parts.


@pytest.mark.hpgl
class TestR06SampleScriptDataPaths:
    """R-06: every sample data-loading script's TEST_DATA_DIR expression is
    evaluated (AST-extracted from the ACTUAL script source, so a reverted
    edit — removing abspath — fails the assertions) and must resolve to an
    absolute, '..'-free directory whose referenced data file loads."""

    SCRIPTS = [
        "gtsim_test.py",
        "gtsimk_test.py",
        "gtsimk_test1.py",
        "test_gtsimk.py",
        "test_gtsimk1.py",
        "test_prop2array.py",
        "ntg_calc_hist.py",
        "mean_calc_hist.py",
    ]

    # (loader kind, data file, dims, undefined value) for each script's
    # first load from TEST_DATA_DIR — mirrors the exact call in the script.
    LOADERS = {
        "gtsim_test.py": ("cont", "BIG_SOFT_DATA_160_141_20.INC", (166, 141, 20), -99),
        "gtsimk_test.py": ("cont", "BIG_SOFT_DATA_160_141_20.INC", (166, 141, 20), -99),
        "gtsimk_test1.py": ("cont", "BIG_SOFT_DATA_160_141_20.INC", (166, 141, 20), -99),
        "test_gtsimk.py": ("cont", "BIG_SOFT_DATA_160_141_20.INC", (166, 141, 20), -99),
        "test_gtsimk1.py": ("cont", "BIG_SOFT_DATA_160_141_20.INC", (166, 141, 20), -99),
        "test_prop2array.py": ("ind", "NEW_TEST_PROP_01.INC", (286, 10, 1), -99),
        "ntg_calc_hist.py": ("ind", "NEW_TEST_PROP_01.INC", (286, 10, 1), -99),
        "mean_calc_hist.py": ("cont", "BIG_SOFT_DATA_CON_160_141_20.INC", (166, 141, 20), -99),
    }

    @staticmethod
    def _script_test_data_dir(script):
        """Evaluate the script's OWN TEST_DATA_DIR assignment from source."""
        script_path = REPO_ROOT / "sample-scripts" / script
        tree = ast.parse(script_path.read_text(encoding="utf-8"))
        ns = {"os": os, "__file__": str(script_path)}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "TEST_DATA_DIR"
            ):
                expr = ast.Expression(node.value)
                ast.fix_missing_locations(expr)
                return eval(compile(expr, str(script_path), "eval"), ns)
        raise AssertionError(f"{script}: no TEST_DATA_DIR assignment found")

    @pytest.mark.parametrize("script", SCRIPTS)
    def test_test_data_dir_absolute_without_dotdot(self, script):
        test_data_dir = self._script_test_data_dir(script)
        assert os.path.isabs(test_data_dir), (
            f"{script}: TEST_DATA_DIR must be absolute, got {test_data_dir!r}"
        )
        parts = str(test_data_dir).replace("\\", "/").split("/")
        assert ".." not in parts, (
            f"{script}: TEST_DATA_DIR still contains a literal '..' — "
            "PathValidator rejects it before normalization"
        )
        assert os.path.isdir(test_data_dir), (
            f"{script}: TEST_DATA_DIR does not resolve to a real directory"
        )

    @pytest.mark.parametrize("script", SCRIPTS)
    @pytest.mark.slow  # H-02: 6 of 8 cases load the 468,120-cell big fixtures
    def test_script_data_file_exists_and_loads(self, script):
        kind, fname, dims, undef = self.LOADERS[script]
        test_data_dir = self._script_test_data_dir(script)
        fpath = os.path.join(test_data_dir, fname)
        assert os.path.isfile(fpath), (
            f"{script}: {fname} missing under TEST_DATA_DIR {test_data_dir}"
        )
        if kind == "cont":
            prop = load_cont_property(fpath, undef, dims)
        else:
            prop = load_ind_property(fpath, undef, [0, 1], dims)
        assert prop.data.size == np.prod(dims), (
            f"{script}: load of {fname} returned {prop.data.size} cells, "
            f"expected {np.prod(dims)}"
        )


# =============================================================================
# R-07 (III-20): gtsimk siblings must raise clean ValueError on user-pk flow
# =============================================================================
#
# gtsim.py's III-20 guard (if sgs_params is None: sgs_params = sk_params;
# if STILL None: raise a clean ValueError) was added to gtsim.py only. The
# siblings gtsimk.py / gtsimk_const_prob.py kept only the first half, so the
# documented user-pk flow (pk_prop provided, no sk_params/sgs_params)
# crashed with a raw `**None` TypeError at the sgs_simulation call.


@pytest.mark.hpgl
class TestGtsimkSiblingSgsParamsGuard:
    """R-07: the user-pk flow must raise a clean ValueError in both siblings
    instead of a `**None` TypeError."""

    @staticmethod
    def _load_script(name, module_name):
        import importlib.util

        sys.path.insert(0, str(REPO_ROOT / "sample-scripts"))
        script = REPO_ROOT / "sample-scripts" / name
        spec = importlib.util.spec_from_file_location(module_name, str(script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_gtsimk_user_pk_flow_raises_clean_value_error(self):
        gtsimk = self._load_script("gtsimk.py", "sample_gtsimk_r07")
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(
            np.array([0, 1, 0, 1], dtype="float32"), np.ones(4, dtype="uint8")
        )
        pk = [
            ContProperty(
                np.array([0.3, 0.7, 0.4, 0.6], dtype="float32"),
                np.ones(4, dtype="uint8"),
            ),
            ContProperty(
                np.array([0.6, 0.4, 0.5, 0.5], dtype="float32"),
                np.ones(4, dtype="uint8"),
            ),
        ]
        with pytest.raises(ValueError, match="sgs_params"):
            gtsimk.gtsim_Kind(
                grid, prop, 2, sk_params=None, pk_prop=pk, sgs_params=None
            )

    def test_gtsimk_const_prob_user_pk_flow_raises_clean_value_error(
        self, monkeypatch
    ):
        gtsimk_cp = self._load_script(
            "gtsimk_const_prob.py", "sample_gtsimk_cp_r07"
        )
        # The const-prob pseudo_gaussian_transform writes
        # results/GTSIM_TRANSFORMED_PROP.INC to the CWD before the SGS step —
        # redirect to a no-op so the test has no CWD side effects.
        monkeypatch.setattr(gtsimk_cp, "write_property", lambda *a, **k: None)
        grid = SugarboxGrid(x=2, y=2, z=1)
        prop = ContProperty(
            np.array([0, 1, 2, 0], dtype="float32"), np.ones(4, dtype="uint8")
        )
        with pytest.raises(ValueError, match="sgs_params"):
            gtsimk_cp.gtsim_Kind_const_prop(
                grid, prop, 3, sk_params=None, pk_prop=[0.3, 0.3], sgs_params=None
            )


# =============================================================================
# R-08 (II-01): cdf_transform must be invertible at the max datum
# =============================================================================
#
# cdf_transform computes cum_probs[-1] == 1.0 exactly (sum of counts ==
# defined_values_count). inverse_normal_score(1.0) clamps to +3.0,
# gaussian_cdf(+3.0) = 0.99865, and back_cdf_transform returns the
# 99.865th-percentile value — NOT the max (empirical pre-fix error: 7.5%
# on this fixture, 27.5% on the finding's). The R-08 clamp (cum_probs[-1] =
# np.nextafter(1.0, 0.0), mirroring src/geo_bsd/cdf.py:218-226) makes the
# max datum round-trip exactly.


@pytest.mark.hpgl
class TestCdfTransformMaxDatumInvertible:
    """R-08: max-datum round-trip through cdf_transform →
    back_cdf_transform must be exact (or near-exact)."""

    @staticmethod
    def _load_book_module():
        import importlib.util

        mod_path = (
            REPO_ROOT / "solved_problems_book" / "shared" / "gaussian_cdf.py"
        )
        spec = importlib.util.spec_from_file_location(
            "book_gaussian_cdf_r08", str(mod_path)
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_max_datum_round_trips_exactly(self):
        book = self._load_book_module()
        rng = np.random.RandomState(0)
        arr = rng.uniform(0.5, 3.02, size=(200, 1, 1)).astype("float32")
        arr[-1, 0, 0] = 4.168
        orig = arr.copy()
        work = arr.copy()
        props, values = book.cdf_transform(work, -99.0)
        book.back_cdf_transform(work, props, values, -99.0)
        assert abs(work.max() - orig.max()) < 1e-3, (
            f"max datum must round-trip: got {work.max()!r}, "
            f"expected {orig.max()!r}"
        )

    def test_inner_values_round_trip_close(self):
        """Control: non-max values invert approximately through the table."""
        book = self._load_book_module()
        rng = np.random.RandomState(1)
        arr = rng.uniform(0.5, 3.02, size=(200, 1, 1)).astype("float32")
        arr[-1, 0, 0] = 4.168
        orig = arr.copy()
        work = arr.copy()
        props, values = book.cdf_transform(work, -99.0)
        book.back_cdf_transform(work, props, values, -99.0)
        np.testing.assert_allclose(work.ravel(), orig.ravel(), rtol=1e-2, atol=1e-2)


# =============================================================================
# P-02 — gtsim_2ind hard-data facies preservation at the clamp boundary
# =============================================================================


@pytest.mark.hpgl
class TestGtsimHardDataFaciesPreserved:
    """P-02: sample-scripts/gtsim.py truncation must preserve hard-data
    facies at the clamp-boundary equality case.

    At hard cells SK is an exact interpolator → pk = 0.0/1.0 exactly →
    tk = inverse_normal_score(pk) = ∓10 (gaussian_cdf.py:17-20 clamp) → the
    pseudo-gaussian transform draws a DEGENERATE uniform(-10,-10)/(10,10) →
    SGS reproduces the conditioning value exactly → prop_sgs == tk at the
    clamp. The strict ``<`` truncation mapped that EQUALITY case to facies
    1, so every facies-0 hard cell was corrupted to 1 (gtsim_test.py
    hard-data check: 2540/3002 errors, where 2540 == facies-0 count
    exactly; verified pre-existing since restore 761f9d5, not a pass-2
    regression). Hard cells must keep their original facies.

    Pre-fix this fixture reports 20/20 hard-data errors; post-fix 0/20.
    """

    @staticmethod
    def _load_gtsim():
        import importlib.util

        sys.path.insert(0, str(REPO_ROOT / "sample-scripts"))
        script = REPO_ROOT / "sample-scripts" / "gtsim.py"
        spec = importlib.util.spec_from_file_location("sample_gtsim_p02", str(script))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_hard_data_facies_preserved_at_clamp_boundary(self, monkeypatch):
        gtsim = self._load_gtsim()
        # The script writes results/GTSIM_*.INC into the CWD — redirect to a
        # no-op so the test has no filesystem side effects (R-07 pattern).
        monkeypatch.setattr(gtsim, "write_property", lambda *a, **k: None)

        np.random.seed(42)
        grid = SugarboxGrid(x=5, y=5, z=2)
        size = 50
        data = np.full(size, -99.0, dtype="float32")
        mask = np.zeros(size, dtype="uint8")
        # 20 hard-data cells: 10 facies-0, 10 facies-1 — every hard cell
        # lands exactly on the ±10 clamp boundary.
        data[:10] = 0.0
        mask[:10] = 1
        data[10:20] = 1.0
        mask[10:20] = 1
        prop = ContProperty(data, mask)

        cov = CovarianceModel(
            type=covariance.exponential, ranges=(10, 10, 10), sill=1
        )
        params = {"radiuses": (20, 20, 20), "max_neighbours": 12, "cov_model": cov}
        result = gtsim.gtsim_2ind(grid, prop, params, params)

        hard = np.where(mask == 1)[0]
        np.testing.assert_array_equal(
            result.data.flat[hard],
            prop.data.flat[hard],
            err_msg=(
                "hard-data facies must be preserved at the clamp-boundary "
                "equality case (P-02)"
            ),
        )

