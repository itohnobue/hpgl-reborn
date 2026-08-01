"""
Regression tests for production-check fixes in routines.py, validation.py
and config.py (F-26, F-27, F-28, F-31, F-32, F-34, F-45, I2-05, I2-06).

Each test fails against the pre-fix code and passes against the fixed code.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.config import SGSConfig, SISConfig
    from geo_bsd.routines import (
        Cubes2PointSet,
        GetCubicalMask,
        GetEllipseMask,
        LoadGslibFile,
        MeanCalc,
        MovingAverage3D,
        SaveGSLIBCubes,
        SaveGSLIBPointSet,
    )
    from geo_bsd.validation import (
        CriticalValidationError,
        ParameterValidator,
        ValidationConstants,
        validate_kriging_params,
    )

    FIXES_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError):
    FIXES_AVAILABLE = False


def _write_gslib(path, header_lines, data_lines):
    """Write a GSLIB file with explicit header + data rows."""
    path.write_text("\n".join(header_lines) + "\n" + "\n".join(data_lines) + "\n")


# =============================================================================
# F-31 — LoadGslibFile rejects duplicate property names
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestLoadGslibFileDuplicateNames:
    """F-31: duplicate property names silently corrupt data (regression)."""

    def test_duplicate_names_raise(self, tmp_path):
        """Header with duplicate property name 'A' must raise ValueError.

        Pre-fix: returns {'A': [2.0, 6.0, 4.0, 8.0]} (interleaved columns,
        silent data corruption). Post-fix: ValueError.
        """
        fpath = tmp_path / "dup.gslib"
        _write_gslib(fpath, ["cap", "2", "A", "A"], ["1 2", "3 4", "5 6", "7 8"])
        with pytest.raises(ValueError, match="duplicate"):
            LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))

    def test_unique_names_still_load(self, tmp_path):
        """A well-formed file with unique names loads correctly."""
        fpath = tmp_path / "ok.gslib"
        _write_gslib(fpath, ["cap", "2", "A", "B"], ["1 2", "3 4", "5 6", "7 8"])
        result = LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))
        np.testing.assert_array_equal(result["A"].ravel(order="F"), [1.0, 3.0, 5.0, 7.0])
        np.testing.assert_array_equal(result["B"].ravel(order="F"), [2.0, 4.0, 6.0, 8.0])


# =============================================================================
# F-32 — writers reject NaN/Inf (C++ writer contract parity)
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestGslibWritersRejectNonFinite:
    """F-32: Python GSLIB writers reject NaN/Inf at write time."""

    def test_pointset_nan_raises(self, tmp_path):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.0, np.nan], dtype="float32")}
        with pytest.raises(ValueError, match="non-finite"):
            SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        assert not os.path.exists(fpath), "no file should be written on rejection"

    def test_pointset_inf_raises(self, tmp_path):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.0, np.inf], dtype="float32")}
        with pytest.raises(ValueError, match="non-finite"):
            SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))

    def test_cubes_nan_raises(self, tmp_path):
        fpath = str(tmp_path / "cubes.inc")
        cubes = {"P": np.full((2, 2, 2), np.nan, dtype="float32")}
        with pytest.raises(ValueError, match="non-finite"):
            SaveGSLIBCubes(cubes, fpath, "cap", basedir=str(tmp_path))

    def test_cubes_finite_write_succeeds(self, tmp_path):
        fpath = str(tmp_path / "cubes.inc")
        cubes = {"P": np.ones((2, 2, 2), dtype="float32")}
        SaveGSLIBCubes(cubes, fpath, "cap", basedir=str(tmp_path))
        assert os.path.exists(fpath)


# =============================================================================
# F-45 — Cubes2PointSet rejects property names colliding with X/Y/Z
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestCubes2PointSetCoordinateCollision:
    """F-45: a property named X/Y/Z must not silently clobber coordinates."""

    def test_property_named_x_raises(self):
        cubes = {
            "X": np.arange(8, dtype="float32").reshape((2, 2, 2), order="F"),
            "prop": np.zeros((2, 2, 2), dtype="float32"),
        }
        mask = np.ones((2, 2, 2), dtype="uint8")
        with pytest.raises(ValueError, match="collides"):
            Cubes2PointSet(cubes, mask)

    def test_property_named_z_raises(self):
        cubes = {
            "Z": np.arange(8, dtype="float32").reshape((2, 2, 2), order="F"),
            "prop": np.zeros((2, 2, 2), dtype="float32"),
        }
        mask = np.ones((2, 2, 2), dtype="uint8")
        with pytest.raises(ValueError, match="collides"):
            Cubes2PointSet(cubes, mask)

    def test_normal_properties_work(self):
        cube = np.ones((2, 2, 2), dtype="float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        result = Cubes2PointSet({"prop": cube}, mask)
        assert "X" in result and "Y" in result and "Z" in result
        assert "prop" in result
        assert len(result["prop"]) == int(mask.sum())


# =============================================================================
# F-28 — trusted basedir instead of self-referential basedir
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestTrustedBasedir:
    """F-28: writers/loaders must use a trusted base, not the file's own dir."""

    def test_write_outside_default_basedir_rejected(self, tmp_path):
        """Writing to a path outside DEFAULT_BASE_DIR must raise when no
        explicit basedir is given.

        Pre-fix: the self-referential basedir (dirname of the file) always
        passes containment, so the write succeeds. Post-fix: the default is
        DEFAULT_BASE_DIR (process cwd); tmp_path lies outside it, so the
        write is rejected.
        """
        fpath = str(tmp_path / "escape.inc")
        ps = {"P": np.array([1.0, 2.0], dtype="float32")}
        with pytest.raises(CriticalValidationError, match="outside allowed base"):
            SaveGSLIBPointSet(ps, fpath, "cap")

    def test_explicit_basedir_allows_write(self, tmp_path):
        """Passing an explicit trusted base permits writes inside it."""
        fpath = str(tmp_path / "ok.inc")
        ps = {"P": np.array([1.0, 2.0], dtype="float32")}
        SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        assert os.path.exists(fpath)

    def test_symlink_escape_rejected(self, tmp_path):
        """A symlink inside the trusted base pointing outside must be rejected.

        Pre-fix: basedir = dirname(filename) = the symlink directory itself,
        so the resolved (escaped) path is still 'inside' it. Post-fix with a
        trusted base, the resolved path outside the base is rejected.
        """
        base = tmp_path / "base"
        base.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()
        link = base / "link"
        link.symlink_to(outside, target_is_directory=True)

        fpath = str(link / "out.inc")
        ps = {"P": np.array([1.0, 2.0], dtype="float32")}
        with pytest.raises(CriticalValidationError, match="outside allowed base"):
            SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(base))


# =============================================================================
# I2-05 — LoadGslibFile total-value cap
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestLoadGslibFileValueCap:
    """I2-05: pure-Python GSLIB loader rejects absurd value counts."""

    def test_grid_prop_product_over_cap_raises(self, tmp_path):
        """grid_size x num_p above MAX_GSLIB_VALUES must raise before parsing.

        The header is tiny; the claimed grid (1e9 cells x 2 props) would
        require 2e9 values, which the cap rejects before reading data.
        """
        fpath = tmp_path / "big.gslib"
        _write_gslib(fpath, ["cap", "2", "A", "B"], ["1 2"])
        with pytest.raises(ValueError, match="exceeding the maximum allowed"):
            LoadGslibFile(str(fpath), property_size=(1_000_000, 1_000, 1), basedir=str(tmp_path))

    def test_normal_load_under_cap(self, tmp_path):
        """A normal file is unaffected by the cap."""
        fpath = tmp_path / "ok.gslib"
        _write_gslib(fpath, ["cap", "1", "A"], ["1", "2", "3", "4"])
        result = LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))
        assert result["A"].size == 4


# =============================================================================
# I2-06 — MovingAverage3D vectorized cubical path + ellipse cap
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestMovingAverage3DVectorized:
    """I2-06: cubical-mask path vectorized; ellipse path volume-capped."""

    @staticmethod
    def _reference_loop(Cube, Mask, Radiuses, undefined_value):
        """Reference per-cell implementation (the pre-fix algorithm)."""
        MeanMask = GetCubicalMask(Radiuses)
        out = Cube.copy()
        for i in range(Cube.shape[0]):
            for j in range(Cube.shape[1]):
                for k in range(Cube.shape[2]):
                    out[i, j, k] = MeanCalc(
                        Cube, Mask, Radiuses, MeanMask, (i, j, k), undefined_value
                    )
        return out

    def test_cubical_matches_reference_loop(self):
        """Vectorized cubical result equals the reference per-cell loop."""
        rng = np.random.RandomState(0)
        Cube = rng.rand(5, 5, 3).astype("float32") * 10
        Mask = rng.randint(0, 2, (5, 5, 3)).astype("uint8")
        Mask[0, 0, 0] = 0
        fast = MovingAverage3D((Cube, Mask), (1, 1, 1), -999.0, GetCubicalMask)
        ref = self._reference_loop(Cube, Mask, (1, 1, 1), -999.0)
        assert fast.shape == ref.shape
        assert fast.dtype == ref.dtype
        np.testing.assert_allclose(fast, ref, atol=1e-5)

    def test_cubical_undefined_fill_matches(self):
        """Fully-masked regions fall back to undefined_value in both paths."""
        Cube = np.ones((4, 4, 4), dtype="float32")
        Mask = np.zeros((4, 4, 4), dtype="uint8")  # no informed cells
        fast = MovingAverage3D((Cube, Mask), (1, 1, 1), -999.0, GetCubicalMask)
        ref = self._reference_loop(Cube, Mask, (1, 1, 1), -999.0)
        np.testing.assert_array_equal(fast, ref)
        assert np.all(fast == -999.0)

    def test_ellipse_path_volume_cap(self, monkeypatch):
        """Non-cubical (ellipse) path rejects grids over the volume cap.

        Pre-fix: no cap exists, so the per-cell Python loop runs unbounded.
        Post-fix: a grid over MAX_MOVING_AVERAGE_VOLUME raises ValueError.
        """
        monkeypatch.setattr(ValidationConstants, "MAX_MOVING_AVERAGE_VOLUME", 8)
        Cube = np.ones((3, 3, 3), dtype="float32")  # volume 27 > 8
        Mask = np.ones((3, 3, 3), dtype="uint8")
        with pytest.raises(ValueError, match="exceeds the maximum"):
            MovingAverage3D((Cube, Mask), (1, 1, 1), -999.0, GetEllipseMask)

    def test_ellipse_path_under_cap_runs(self, monkeypatch):
        """A small ellipse-mask grid still runs through the loop path."""
        monkeypatch.setattr(ValidationConstants, "MAX_MOVING_AVERAGE_VOLUME", 1000)
        Cube = np.ones((3, 3, 3), dtype="float32")
        Mask = np.ones((3, 3, 3), dtype="uint8")
        out = MovingAverage3D((Cube, Mask), (1, 1, 1), -999.0, GetEllipseMask)
        assert out.shape == (3, 3, 3)


# =============================================================================
# F-26 — config dataclasses reject fractional radiuses
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestConfigFractionalRadiuses:
    """F-26: construction-time parity with validate_radius."""

    def test_sgs_fractional_radius_raises(self):
        with pytest.raises(ValueError, match="integer"):
            SGSConfig(radiuses=(5.5, 5, 3))

    def test_sis_fractional_radius_raises(self):
        with pytest.raises(ValueError, match="integer"):
            SISConfig(radiuses=(5, 5.5, 3))

    def test_zero_radius_still_accepted(self):
        """Radius 0 remains valid for SGS/SIS (documented CDF-draw path)."""
        assert SGSConfig(radiuses=(0, 0, 0)).radiuses == (0, 0, 0)
        assert SISConfig(radiuses=(0, 0, 0)).radiuses == (0, 0, 0)

    def test_integer_radiuses_accepted(self):
        assert SGSConfig(radiuses=(5, 5, 3)).radiuses == (5, 5, 3)
        assert SISConfig(radiuses=(10, 20, 30)).radiuses == (10, 20, 30)


# =============================================================================
# F-27 — validate_max_neighbors type gate
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestValidateMaxNeighborsTypeGate:
    """F-27: float/bool must fail fast instead of reaching raw ctypes."""

    def test_float_raises_typeerror(self):
        with pytest.raises(TypeError, match="must be an int"):
            ParameterValidator.validate_max_neighbors(5.5)

    def test_bool_raises_typeerror(self):
        with pytest.raises(TypeError, match="must be an int"):
            ParameterValidator.validate_max_neighbors(True)

    def test_numpy_int_accepted(self):
        assert ParameterValidator.validate_max_neighbors(np.int32(12)) is None

    def test_valid_int_accepted(self):
        assert ParameterValidator.validate_max_neighbors(12) is None


# =============================================================================
# F-34 — kriging-path positive minimum radius
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestKrigingMinimumRadius:
    """F-34: the kriging path rejects zero radius without breaking SGS."""

    def test_zero_radius_default_still_accepted(self):
        """Global default (SGS zero-radius CDF-draw) keeps working."""
        assert ParameterValidator.validate_radius((0, 0, 0)) == (0, 0, 0)

    def test_zero_radius_kriging_policy_rejected(self):
        """With MIN_KRIGING_RADIUS, zero radius is rejected (pre-fix: no such
        parameter, so a TypeError would surface instead of this policy)."""
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            ParameterValidator.validate_radius(
                (0, 0, 0), min_radius=ValidationConstants.MIN_KRIGING_RADIUS
            )

    def test_positive_radius_kriging_policy_accepted(self):
        assert ParameterValidator.validate_radius(
            (2, 2, 1), min_radius=ValidationConstants.MIN_KRIGING_RADIUS
        ) == (2, 2, 1)

    def test_validate_kriging_params_min_radius_threaded(self):
        """validate_kriging_params forwards min_radius to validate_radius."""
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            validate_kriging_params(
                None,
                (0, 0, 0),
                None,
                None,
                min_radius=ValidationConstants.MIN_KRIGING_RADIUS,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
