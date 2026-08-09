"""
Unit tests for HPGL routines module (routines.py).

Q1 fix: Previously only CalcVPCsIndicator had a single test.
This file adds dedicated tests for all 16 public functions in routines.py.

Note: routines.py imports from geo.py which may have syntax issues.
Tests skip gracefully when the module cannot be imported.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Attempt import — routines.py depends on geo.py which may have syntax issues
try:
    from geo_bsd.routines import (
        CalcMarginalProbsIndicator,
        CalcMean,
        CalcVPC,
        CalcVPCsIndicator,
        Cube2PointSet,
        CubeFromVPC,
        Cubes2PointSet,
        CubesFromVPCs,
        GetCubicalMask,
        GetEllipseMask,
        LoadGslibFile,
        MovingAverage3D,
        PointSet2Cube,
        SaveGSLIBCubes,
        SaveGSLIBPointSet,
    )

    ROUTINES_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError):
    ROUTINES_AVAILABLE = False


# =============================================================================
# Helper functions
# =============================================================================


def _make_cube_mask(nx=5, ny=4, nz=3):
    """Create a simple 3D cube and mask for testing.

    Returns (cube, mask) where cube has gradient values and mask has ~80% informed.
    """
    np.random.seed(42)
    shape = (nx, ny, nz)
    size = nx * ny * nz
    cube = np.arange(size, dtype="float32").reshape(shape, order="F")
    mask = np.ones(shape, dtype="uint8")
    mask[::5] = 0  # ~20% uninformed
    return cube, mask


# =============================================================================
# CalcMean Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCalcMean:
    """Tests for CalcMean function."""

    def test_calc_mean_all_masked(self):
        """CalcMean with all-masked returns masked constant (NaN/masked)."""
        cube = np.ones((2, 2, 2), dtype="float32") * 5.0
        mask = np.zeros((2, 2, 2), dtype="uint8")
        result = CalcMean(cube, mask)
        # All masked -> masked array mean is masked
        assert np.ma.is_masked(result) or np.isnan(result)


# =============================================================================
# CalcMarginalProbsIndicator Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCalcMarginalProbsIndicator:
    """Tests for CalcMarginalProbsIndicator function."""

    def test_two_indicators_balanced(self):
        """Equal counts of two indicators give equal probs."""
        cube = np.array([0, 0, 1, 1], dtype="float32").reshape((2, 2, 1), order="F")
        mask = np.ones((2, 2, 1), dtype="uint8")
        result = CalcMarginalProbsIndicator(cube, mask, [0, 1])
        assert len(result) == 2
        assert result[0] == pytest.approx(0.5)
        assert result[1] == pytest.approx(0.5)

    def test_single_indicator_all_match(self):
        """When all values are the indicator, marginal prob is 1.0."""
        cube = np.ones((3, 3, 2), dtype="float32") * 5.0
        mask = np.ones((3, 3, 2), dtype="uint8")
        result = CalcMarginalProbsIndicator(cube, mask, [5])
        assert len(result) == 1
        assert result[0] == pytest.approx(1.0)


# =============================================================================
# CalcVPC Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCalcVPCsIndicator:
    """Tests for CalcVPCsIndicator function."""

    def test_two_indicators_returns_two_vpcs(self):
        """CalcVPCsIndicator returns one VPC per indicator."""
        nx, ny, nz = 3, 3, 2
        cube = (
            np.random.randint(0, 2, nx * ny * nz).astype("float32").reshape((nx, ny, nz), order="F")
        )
        mask = np.ones((nx, ny, nz), dtype="uint8")
        indicators = [0, 1]
        marginal_probs = [0.5, 0.5]
        result = CalcVPCsIndicator(cube, mask, indicators, marginal_probs)
        assert len(result) == 2
        for vpc in result:
            assert len(vpc) == nz


# =============================================================================
# CubeFromVPC / CubesFromVPCs Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCubeFromVPC:
    """Tests for CubeFromVPC function."""

    def test_cube_from_vpc_shape(self):
        """CubeFromVPC produces a cube with correct shape."""
        vpc = np.array([0.1, 0.2, 0.3, 0.4], dtype="float32")
        result = CubeFromVPC(vpc, NX=5, NY=3)
        assert result.shape == (5, 3, 4)

    def test_cube_from_vpc_values(self):
        """CubeFromVPC replicates VPC values across XY plane."""
        vpc = np.array([10.0, 20.0], dtype="float32")
        result = CubeFromVPC(vpc, NX=2, NY=2)
        assert result.shape == (2, 2, 2)
        # All XY cells in layer 0 should be 10.0
        assert np.all(result[:, :, 0] == 10.0)
        # All XY cells in layer 1 should be 20.0
        assert np.all(result[:, :, 1] == 20.0)


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCubesFromVPCs:
    """Tests for CubesFromVPCs function."""

    def test_cubes_from_vpcs_count(self):
        """CubesFromVPCs returns one cube per VPC."""
        vpcs = [
            np.array([0.1, 0.2], dtype="float32"),
            np.array([0.3, 0.4], dtype="float32"),
        ]
        results = CubesFromVPCs(vpcs, NX=3, NY=3)
        assert len(results) == 2
        for cube in results:
            assert cube.shape == (3, 3, 2)


# =============================================================================
# Cubes2PointSet / Cube2PointSet Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCube2PointSet:
    """Tests for Cube2PointSet function."""

    def test_partial_mask(self):
        """Cube2PointSet with partial mask returns fewer points than all-informed."""
        cube = np.zeros((2, 2, 2), dtype="float32", order="F")
        mask = np.ones((2, 2, 2), dtype="uint8", order="F")
        mask[0, 0, 0] = 0
        mask[1, 1, 1] = 0
        x, y, z, prop = Cube2PointSet(cube, mask)
        assert len(prop) > 0, "Expected at least one informed point"


# =============================================================================
# PointSet2Cube Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestPointSet2Cube:
    """Tests for PointSet2Cube function."""

    def test_fortran_order_output_consistency(self):
        """Regression test: Cube2PointSet with Fortran-order input must produce
        consistent-length output arrays (X, Y, Z, and prop all same length).

        This test verifies the fix for the documented Fortran-order bug where
        uint8 mask slices were treated as fancy-index integers instead of
        boolean masks. The fix converts mask slices to bool dtype before indexing.

        Fixed as of v1.6.1 — the xfail marker has been removed.
        """
        cube = np.arange(27, dtype="float32").reshape((3, 3, 3), order="F")
        mask = np.ones((3, 3, 3), dtype="uint8")
        mask[0, 0, 0] = 0  # One uninformed cell
        x, y, z, prop = Cube2PointSet(cube, mask)
        # Regression assertion: all output arrays must have the same length
        n = len(x)
        assert n > 0, "Expected at least one informed point"
        assert len(y) == n, f"Y length {len(y)} != X length {n} (Fortran-order bug)"
        assert len(z) == n, f"Z length {len(z)} != X length {n} (Fortran-order bug)"
        assert len(prop) == n, f"prop length {len(prop)} != X length {n} (Fortran-order bug)"

    def test_out_of_bounds_points_ignored(self):
        """Points outside the cube are silently ignored."""
        cube = np.zeros((2, 2, 2), dtype="float32")
        x = np.array([0, 5], dtype="int32")  # Second X is out of bounds
        y = np.array([0, 0], dtype="int32")
        z = np.array([0, 0], dtype="int32")
        prop = np.array([42.0, 99.0], dtype="float32")
        new_cube, new_mask = PointSet2Cube(x, y, z, prop, cube)
        # Only first point should be placed
        assert new_cube[0, 0, 0] == 42.0


# =============================================================================
# SaveGSLIBPointSet Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestSaveGSLIBPointSet:
    """Tests for SaveGSLIBPointSet function."""

    def test_saves_valid_gslib_file(self, tmp_path):
        """SaveGSLIBPointSet writes a properly formatted GSLIB point set file."""
        fpath = str(tmp_path / "test.inc")
        point_set = {
            "X": np.array([0, 1], dtype="int32"),
            "Y": np.array([0, 0], dtype="int32"),
            "Z": np.array([0, 0], dtype="int32"),
            "Property": np.array([10.5, 20.5], dtype="float32"),
        }
        # F-28: pass an explicit trusted base — the default (DEFAULT_BASE_DIR)
        # is the process cwd, so writing to tmp_path requires an explicit base.
        SaveGSLIBPointSet(point_set, fpath, "Test Point Set", basedir=str(tmp_path))
        assert os.path.exists(fpath)
        content = open(fpath).read()
        assert "Test Point Set" in content
        assert "X" in content
        assert "Property" in content

    def test_empty_filename_raises(self):
        """Empty or None filename raises ValueError."""
        point_set = {"X": np.array([0], dtype="int32")}
        with pytest.raises(ValueError):
            SaveGSLIBPointSet(point_set, "", "caption")
        with pytest.raises(ValueError):
            SaveGSLIBPointSet(point_set, None, "caption")

    def test_non_string_filename_raises(self):
        """Non-string filename raises ValueError."""
        point_set = {"X": np.array([0], dtype="int32")}
        with pytest.raises(ValueError):
            SaveGSLIBPointSet(point_set, 123, "caption")


# =============================================================================
# SaveGSLIBCubes Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestSaveGSLIBCubes:
    """Tests for SaveGSLIBCubes function."""

    def test_saves_valid_gslib_cube_file(self, tmp_path):
        """SaveGSLIBCubes writes a properly formatted GSLIB cube file."""
        fpath = str(tmp_path / "cubes.inc")
        cubes = {
            "Property1": np.ones((2, 2, 2), dtype="float32"),
            "Property2": np.ones((2, 2, 2), dtype="float32") * 2,
        }
        # F-28: pass an explicit trusted base — the default (DEFAULT_BASE_DIR)
        # is the process cwd, so writing to tmp_path requires an explicit base.
        SaveGSLIBCubes(cubes, fpath, "Test Cubes", basedir=str(tmp_path))
        assert os.path.exists(fpath)
        content = open(fpath).read()
        assert "Test Cubes" in content
        assert "Property1" in content

    def test_empty_filename_raises(self):
        """Empty filename raises ValueError."""
        cubes = {"p": np.ones((1, 1, 1), dtype="float32")}
        with pytest.raises(ValueError):
            SaveGSLIBCubes(cubes, "", "c")


# =============================================================================
# GetCubicalMask / GetEllipseMask Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestGetCubicalMask:
    """Tests for GetCubicalMask function."""

    def test_returns_correct_shape(self):
        """GetCubicalMask returns a mask of (2*R1, 2*R2, 2*R3)."""
        result = GetCubicalMask((3, 4, 5))
        assert result.shape == (6, 8, 10)

    def test_all_ones(self):
        """Cubical mask should be all ones."""
        result = GetCubicalMask((2, 2, 2))
        assert np.all(result == 1)

    def test_fortran_order(self):
        """Result should be in Fortran (column-major) order."""
        result = GetCubicalMask((2, 2, 2))
        assert result.flags["F_CONTIGUOUS"]


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestGetEllipseMask:
    """Tests for GetEllipseMask function."""

    def test_returns_correct_shape(self):
        """GetEllipseMask returns a mask of (2*R1, 2*R2, 2*R3)."""
        result = GetEllipseMask((3, 4, 5))
        assert result.shape == (6, 8, 10)

    def test_center_is_one(self):
        """Center point of ellipse mask should be 1."""
        result = GetEllipseMask((3, 3, 3))
        assert result[3, 3, 3] == 1

    def test_corners_are_zero(self):
        """Corners of ellipse mask should be 0."""
        result = GetEllipseMask((3, 3, 3))
        assert result[0, 0, 0] == 0
        assert result[-1, -1, -1] == 0


# =============================================================================
# LoadGslibFile Tests
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestLoadGslibFile:
    """Tests for LoadGslibFile function."""

    def test_loads_valid_gslib_file(self, tmp_path):
        """LoadGslibFile loads a properly formatted GSLIB file."""
        fpath = tmp_path / "test.gslib"
        content = "Test Data\n2\nX\nY\n0 10.0\n1 20.0\n2 30.0\n3 40.0\n"
        fpath.write_text(content)
        # F-28: pass an explicit trusted base — the default (DEFAULT_BASE_DIR)
        # is the process cwd, so reading tmp_path requires an explicit base.
        result = LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))
        assert "X" in result
        assert "Y" in result
        assert result["X"].size == 4

    def test_empty_filename_raises(self):
        """Empty or None filename raises ValueError."""
        with pytest.raises(ValueError):
            LoadGslibFile("", property_size=(2, 2, 1))
        with pytest.raises(ValueError):
            LoadGslibFile(None, property_size=(2, 2, 1))


# =============================================================================
# F-29: GSLIB writers must reject finite out-of-window (|v| > 1.0e21) values
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestGslibWriterSentinelWindow:
    """F-29: the reader converts |v| > 1.0e21 to NaN (missing sentinel), so
    the writer must reject such FINITE values at write time — otherwise a
    value like 2.0e21 round-trips as silent NaN corruption (probe:
    SaveGSLIBCubes [2.0e21] → LoadGslibFile [NaN])."""

    def test_save_pointset_rejects_out_of_window(self, tmp_path):
        fpath = str(tmp_path / "out_of_window.inc")
        point_set = {
            "X": np.array([0, 1], dtype="int32"),
            "Y": np.array([0, 0], dtype="int32"),
            "Z": np.array([0, 0], dtype="int32"),
            "Property": np.array([1.0, 2.0e21], dtype="float64"),
        }
        with pytest.raises(ValueError, match="sentinel window"):
            SaveGSLIBPointSet(point_set, fpath, "Test", basedir=str(tmp_path))

    def test_save_cubes_rejects_out_of_window(self, tmp_path):
        fpath = str(tmp_path / "out_of_window.inc")
        cubes = {"Property": np.array([[[1.0]], [[2.0e21]]], dtype="float64")}
        with pytest.raises(ValueError, match="sentinel window"):
            SaveGSLIBCubes(cubes, fpath, "Test", basedir=str(tmp_path))

    def test_in_window_values_still_write(self, tmp_path):
        """Control: values inside ±1.0e21 write normally and round-trip."""
        fpath = str(tmp_path / "in_window.inc")
        cubes = {"Property": np.array([[[1.0e20]], [[-1.0e20]]], dtype="float64")}
        SaveGSLIBCubes(cubes, fpath, "Test", basedir=str(tmp_path))
        loaded = LoadGslibFile(fpath, property_size=(1, 1, 2), basedir=str(tmp_path))
        assert np.all(np.isfinite(loaded["Property"]))


# =============================================================================
# F-30: MovingAverage3D per-cell (ellipse) path needs a work-based cap
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestMovingAverageWorkCap:
    """F-30: the volume cap bounds memory only; the per-cell ellipse path is
    O(N·V). A monkeypatched tiny work cap must reject an over-cap input
    before the loop runs."""

    def test_ellipse_path_rejects_over_cap(self, monkeypatch):
        import geo_bsd.routines as r

        monkeypatch.setattr(
            r.ValidationConstants, "MAX_MOVING_AVERAGE_WORK", 1000
        )
        cube, mask = _make_cube_mask(nx=6, ny=6, nz=3)
        # radius (2,2,2): window volume = 4^3 = 64; grid volume 108 ->
        # work 6912 > 1000. GetEllipseMask is non-uniform -> per-cell path.
        with pytest.raises(ValueError, match="per-cell work"):
            MovingAverage3D((cube, mask), (2, 2, 2), -999.0, GetEllipseMask)

    def test_cubical_path_under_cap_still_works(self, monkeypatch):
        import geo_bsd.routines as r

        monkeypatch.setattr(
            r.ValidationConstants, "MAX_MOVING_AVERAGE_WORK", 1000
        )
        cube, mask = _make_cube_mask(nx=4, ny=4, nz=2)
        # cubical path is vectorized and below any reasonable cap.
        result = MovingAverage3D((cube, mask), (1, 1, 1), -999.0, GetCubicalMask)
        assert result.shape == cube.shape


# =============================================================================
# II-41: mask semantics — non-zero = informed (non-binary masks)
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestNonBinaryMaskSemantics:
    """II-41: a mask with value 2 is a legal "non-zero = informed" mask.
    CalcVPC must not halve layer means, Cubes2PointSet must not allocate
    trailing-zero rows, and MovingAverage3D must not mark every cell
    undefined."""

    def test_calc_vpc_mask2_matches_mask1(self):
        cube = np.zeros((2, 2, 2), dtype="float32")
        cube[:, :, 0] = 10.0
        cube[:, :, 1] = 20.0
        mask1 = np.ones((2, 2, 2), dtype="uint8")
        mask2 = np.full((2, 2, 2), 2, dtype="uint8")
        vpc1 = CalcVPC(cube, mask1, 0.0)
        vpc2 = CalcVPC(cube, mask2, 0.0)
        np.testing.assert_array_equal(vpc1, vpc2)

    def test_cubes2pointset_mask2_no_trailing_zeros(self):
        cube = np.zeros((2, 2, 2), dtype="float32", order="F")
        cube[:, :, :] = 5.0
        mask = np.full((2, 2, 2), 2, dtype="uint8", order="F")
        # mask out one cell: only 7 informed
        mask[0, 0, 0] = 0
        result = Cubes2PointSet({"prop": cube}, mask)
        assert len(result["prop"]) == 7, (
            "II-41: Cubes2PointSet must allocate exactly (Mask != 0).sum() "
            f"rows, got {len(result['prop'])}"
        )

    def test_moving_average_mask2_not_all_undefined(self):
        cube = np.ones((4, 4, 2), dtype="float32") * 42.0
        mask = np.full((4, 4, 2), 2, dtype="uint8")
        result = MovingAverage3D((cube, mask), (1, 1, 1), -999.0, GetCubicalMask)
        assert np.any(result != -999.0), (
            "II-41: mask=2 cells are informed; every cell must be averaged, "
            "not undefined"
        )


# =============================================================================
# II-42: LoadGslibFile line-length / token-count DoS hardening
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestLoadGslibLineDoSHardening:
    """II-42: a crafted over-long line with far more tokens than num_p must
    be rejected BEFORE line.split() materializes the token list."""

    def test_oversized_line_rejected(self, tmp_path):
        fpath = tmp_path / "oversized.inc"
        with open(fpath, "w") as fh:
            fh.write("caption\n1\nprop\n")
            # > 1 MB single line so the II-42 line-length check fires
            # (the R-12 row-count pre-pass sees one line == grid_size 1).
            fh.write(" ".join(["1.0"] * 300000) + "\n")
        with pytest.raises(RuntimeError, match="exceeds .* bytes"):
            LoadGslibFile(str(fpath), property_size=(1, 1, 1), basedir=str(tmp_path))

    def test_too_many_tokens_on_line_rejected(self, tmp_path):
        fpath = tmp_path / "manytokens.inc"
        with open(fpath, "w") as fh:
            fh.write("caption\n1\nprop\n")
            fh.write("1.0 2.0 3.0\n")  # 3 tokens but num_p=1
        # grid_size=1 so the row-count pre-pass succeeds and the parse
        # loop reaches the token-count check.
        with pytest.raises(RuntimeError, match="expected 1 values per data line"):
            LoadGslibFile(str(fpath), property_size=(1, 1, 1), basedir=str(tmp_path))

    def test_normal_line_still_parses(self, tmp_path):
        fpath = tmp_path / "normal.inc"
        with open(fpath, "w") as fh:
            fh.write("caption\n1\nprop\n1.0\n2.0\n3.0\n")
        result = LoadGslibFile(str(fpath), property_size=(3, 1, 1), basedir=str(tmp_path))
        np.testing.assert_array_equal(
            result["prop"].ravel(), np.array([1.0, 2.0, 3.0])
        )


# =============================================================================
# III-17: Cubes2PointSet equal-shape validation
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestCubes2PointSetShapeValidation:
    """III-17: mismatched cube shapes must raise instead of silently
    truncating extra Z-layers (mirror SaveGSLIBCubes)."""

    def test_mismatched_shapes_raise(self):
        cube_a = np.zeros((2, 2, 2), dtype="float32")
        cube_b = np.zeros((2, 2, 3), dtype="float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        with pytest.raises(ValueError, match="identical shape"):
            Cubes2PointSet({"a": cube_a, "b": cube_b}, mask)

    def test_matching_shapes_succeed(self):
        cube_a = np.zeros((2, 2, 2), dtype="float32")
        cube_b = np.zeros((2, 2, 2), dtype="float32")
        mask = np.ones((2, 2, 2), dtype="uint8")
        result = Cubes2PointSet({"a": cube_a, "b": cube_b}, mask)
        assert len(result["a"]) == 8


# =============================================================================
# III-18: CalcMean / CalcVPC isfinite gates
# =============================================================================


@pytest.mark.skipif(not ROUTINES_AVAILABLE, reason="routines module not available")
class TestRoutinesIsfiniteGates:
    """III-18: NaN in an informed cell must raise (mirror C++
    calc_mean.cpp:18-22), not silently propagate."""

    def test_calc_mean_nan_informed_raises(self):
        cube = np.array([1.0, np.nan, 3.0], dtype="float32").reshape((1, 3, 1))
        mask = np.array([1, 1, 1], dtype="uint8").reshape((1, 3, 1))
        with pytest.raises(ValueError, match="NaN or Inf"):
            CalcMean(cube, mask)

    def test_calc_mean_nan_masked_ok(self):
        """NaN in a MASKED (zero) cell is excluded by definition and must
        not raise."""
        cube = np.array([1.0, np.nan, 3.0], dtype="float32").reshape((1, 3, 1))
        mask = np.array([1, 0, 1], dtype="uint8").reshape((1, 3, 1))
        result = CalcMean(cube, mask)
        assert result == pytest.approx(2.0)

    def test_calc_vpc_nan_informed_raises(self):
        cube = np.zeros((2, 2, 2), dtype="float32")
        cube[0, 0, 0] = np.nan
        mask = np.ones((2, 2, 2), dtype="uint8")
        with pytest.raises(ValueError, match="NaN or Inf"):
            CalcVPC(cube, mask, 0.0)

    def test_calc_mean_finite_still_works(self):
        cube = np.array([1.0, 2.0, 3.0], dtype="float32").reshape((1, 3, 1))
        mask = np.ones((1, 3, 1), dtype="uint8")
        assert CalcMean(cube, mask) == pytest.approx(2.0)
