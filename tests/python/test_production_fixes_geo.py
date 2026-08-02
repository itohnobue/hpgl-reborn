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

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

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
        lvm_kriging,
        median_ik,
        read_inc_file_float,
        simple_kriging,
        simple_kriging_weights,
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
