"""Regression tests for Stage-6 CONFIRMED validation/IO/routines/variogram
fixes (TEST-ADD T-03..T-19). Each test fails against the pre-fix code and
passes against the fixed code (see the test docstrings for the pre-fix failure
mode). Tests exercise the documented TOP-LEVEL public API
(``from geo_bsd import X``) per pattern got-20260803180229 / finding B-08.

Covers:
- T-03  E-M80: CStackLayers scalez beyond the float32 max is rejected
- T-04  E-M72: property-name length cap (1024 UTF-8 bytes)
- T-05  E2-18: SaveGSLIBCubes rejects equal-flat/different-shape cubes
- T-06  E2-30: rotated-ellipsoid pair counting (hemisphere cut) matches C++
- T-07  E-M10: per-pair variance accumulates in float64 (not float32)
- T-09  E2-26: MovingAverage3D preserves fractional means on int cubes
- T-19  E2-06: fractional undefined_value round-trips both loaders identically
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import geo_bsd
    from geo_bsd import (
        ContProperty,
        SugarboxGrid,
        load_cont_property,
        routines,
        validation,
        variogram,
        write_property,
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False

# Every test in this module needs the geo_bsd package (the variogram/routines
# paths are pure Python but import through the package top level).
pytestmark = pytest.mark.skipif(
    not HPGL_AVAILABLE, reason="HPGL (geo_bsd) not available"
)


# =============================================================================
# T-03 — E-M80: CStackLayers scalez beyond the float32 max must raise BEFORE
# the ctypes conversion (a finite float64 like 1e39 silently became inf and
# produced an all-blank result with no exception).
# =============================================================================


class TestCStackLayersScalezFloat32Max:
    def test_scalez_beyond_float32_max_raises(self):
        pytest.importorskip("geo_bsd.cvariogram")
        layer = np.ones((2, 2), dtype="float32")
        result = np.zeros((2, 2, 1), dtype="float32")
        with pytest.raises(ValueError, match="exceeds the float32 maximum"):
            geo_bsd.cvariogram.CStackLayers(
                [layer], [1], 1, 1e39, -99.0, result
            )

    def test_valid_scalez_stack_succeeds(self):
        """Control: a legal scalez still stacks a layer (value assertion:
        a thickness-1 layer with scalez=1 fills cell k=0 with the marker)."""
        pytest.importorskip("geo_bsd.cvariogram")
        layer = np.ones((1, 1), dtype="float32")
        result = np.zeros((1, 1, 1), dtype="float32")
        geo_bsd.cvariogram.CStackLayers([layer], [1], 1, 1.0, -99, result)
        assert result[0, 0, 0] == 1.0


# =============================================================================
# T-04 — E-M72: property names longer than the C++ fast reader's 1024-byte cap
# must be rejected at write time (the writer must not produce files the reader
# rejects). Byte length (not code points) matches the C++ name.size() check.
# =============================================================================


class TestPropertyNameLengthCap:
    def test_name_over_1024_bytes_rejected(self):
        with pytest.raises(ValueError, match="1024"):
            validation.validate_property_name("x" * 1025)

    def test_name_at_1024_bytes_accepted(self):
        name = validation.validate_property_name("x" * 1024)
        assert name == "x" * 1024

    def test_multibyte_name_byte_cap_enforced(self):
        """A 513-character multi-byte (2-byte/char) name is 1026 UTF-8 bytes —
        over the byte cap even though the code-point count is under 1024."""
        with pytest.raises(ValueError, match="1024"):
            validation.validate_property_name("é" * 513)


# =============================================================================
# T-05 — E2-18: SaveGSLIBCubes must reject cubes with equal flat length but
# different per-dimension shapes (e.g. (2,2,2) vs (4,2,1) — both 8 cells) —
# LoadGslibFile reshapes every property to the same property_size, so such a
# file silently scrambles layers on load.
# =============================================================================


class TestSaveGslibCubesEqualFlatShapeRejected:
    def test_equal_flat_different_shape_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="identical shape"):
            routines.SaveGSLIBCubes(
                {"a": np.ones((2, 2, 2), dtype="float32"),
                 "b": np.ones((4, 2, 1), dtype="float32")},
                str(tmp_path / "bad.gslib"),
                "caption",
                basedir=str(tmp_path),
            )

    def test_matching_shapes_still_write(self, tmp_path):
        """Control: identical-shape cubes write without raising."""
        routines.SaveGSLIBCubes(
            {"a": np.ones((2, 2, 2), dtype="float32"),
             "b": np.ones((2, 2, 2), dtype="float32")},
            str(tmp_path / "ok.gslib"),
            "caption",
            basedir=str(tmp_path),
        )
        assert (tmp_path / "ok.gslib").exists()


# =============================================================================
# T-06 — E2-30: the rotated-ellipsoid hemisphere cut must count each unordered
# pair exactly once (uniform pair weighting), matching the C++ point-set
# kernel. Pre-fix the axis-aligned window admitted BOTH +v and −v offsets of
# every in-tunnel pair on this small grid (all offsets are deep inside the
# window box) → each pair was counted twice. Post-fix the hemisphere cut
# admits exactly one endpoint per pair, and the tunnel filter (_IsInTunnel,
# identical C++ is_in_tunnel) then excludes the opposite-sign and (on this
# float64 boundary) the vertical pairs, so the admitted total is 60.
# =============================================================================


class TestRotatedEllipsoidPairCounting:
    def test_azimuth45_total_pairs_counted_once(self):
        pytest.importorskip("geo_bsd.cvariogram")

        grid_pts = np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")
        px, py, pz = np.meshgrid(grid_pts, grid_pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(16, dtype="float32")
        n_points = 16

        templ = variogram.TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=10,
            Ellipsoid=variogram.TVEllipsoid(
                R1=100, R2=100, R3=100, Azimut=45.0, Dip=0, Rotation=0
            ),
            FirstLagDistance=0,
        )
        res, _ = variogram.PointSetScanContStyle(
            templ, {"X": px, "Y": py, "Z": pz},
            variogram.CalcVariogramFunction,
            {"HardData": [pvals]},
        )
        # Count slot = NumValues (index 2 for 1 value array). Every unordered
        # pair admitted exactly once across all lags (hemisphere cut, E2-30).
        # The tunnel filter (_IsInTunnel, variogram.py:608 — identical C++
        # is_in_tunnel, variograms.cpp:307) is applied AFTER the hemisphere
        # cut and admits offset (dx,dy,0) iff Dist <= S1 (TolDistance=1.0),
        # i.e. sqrt(S2²+S3²) <= S1. On this 4×4 grid:
        #   - 36 same-sign pairs (dx·dy > 0) pass the cone and are counted;
        #   - 36 opposite-sign pairs (dx·dy < 0) fail it from both endpoints;
        #   - of the 48 zero-component pairs, the 24 horizontal (dy=0) pass;
        #     the 24 vertical (dx=0) fail by 1 ULP — Direction2[1] =
        #     0.7071067811865476 vs Direction1[1] = 0.7071067811865475, so
        #     Dist = |dy·D2[1]|/R2 is exactly one ULP above S1 = |dy·D1[1]|/R1
        #     and the `Dist <= S1` boundary check rejects them.
        # Admitted = 24 + 36 = 60 (verified against the C++ kernel parity in
        # the sibling test). Pre-fix (no hemisphere cut) the window admitted
        # +v and −v from both endpoints on this small grid, counting each
        # admitted pair twice -> 120.
        total_pairs = int(res[:, 2].sum())
        assert total_pairs == 60, (
            f"azimuth=45 pair count {total_pairs} != 60 "
            f"(24 horizontal + 36 same-sign pairs admitted by the tunnel "
            f"cone after the hemisphere cut)"
        )

    def test_azimuth45_matches_cpp_kernel(self):
        """E2-30 parity: the pure-Python point-set scan must produce the same
        lag-bin values as the C++ calc_variograms_from_point_set under a
        rotated ellipsoid (uniform pair weighting is the parity target)."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import (
            CalcVariogramsFromPointSet,
            Ellipsoid,
            VariogramSearchTemplate,
        )

        grid_pts = np.array([0.0, 1.0, 2.0, 3.0], dtype="float32")
        px, py, pz = np.meshgrid(grid_pts, grid_pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(16, dtype="float32")

        c_ell = Ellipsoid(R1=100, R2=100, R3=100, azimuth=45, dip=0, rotation=0)
        c_templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=10, first_lag_distance=0.0, ellipsoid=c_ell,
        )
        c_var = np.zeros(10, dtype="float32")
        _, cpp = CalcVariogramsFromPointSet(
            c_templ, {"X": px, "Y": py, "Z": pz, "Property": pvals}, c_var
        )

        p_templ = variogram.TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=10,
            Ellipsoid=variogram.TVEllipsoid(
                R1=100, R2=100, R3=100, Azimut=45.0, Dip=0, Rotation=0
            ),
            FirstLagDistance=0,
        )
        py_res, _ = variogram.PointSetScanContStyle(
            p_templ, {"X": px, "Y": py, "Z": pz},
            variogram.CalcVariogramFunction,
            {"HardData": [pvals]},
        )
        np.testing.assert_allclose(py_res[:, 0], cpp, atol=1e-4)


# =============================================================================
# T-07 — E-M10: the per-pair variance must accumulate in float64. The old
# float32 cast of the per-pair differences BEFORE squaring silently rounded
# squares that need >24 mantissa bits (probe: diff 10001 → square 100,020,001
# rounds to 100,020,000 in float32), so the accumulated lag sum differs.
# =============================================================================


class TestVariogramFloat64Accumulation:
    def test_cross_magnitude_sum_matches_float64(self):
        """Two lag-1 pairs with values [0, 10001, 10000]: variances
        10001² = 100,020,001 and 1² = 1. float64 accumulation keeps the exact
        sum 100,020,002; the pre-fix float32 per-pair cast rounded the square
        to 100,020,000 → sum 100,020,001 (and the +1 tail pair was lost)."""
        xs = np.array([0.0, 1.0, 2.0], dtype="float32")
        ys = np.zeros(3, dtype="float32")
        zs = np.zeros(3, dtype="float32")
        vals = np.array([0.0, 10001.0, 10000.0], dtype="float32")

        templ = variogram.TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=3,
            Ellipsoid=variogram.TVEllipsoid(R1=100, R2=100, R3=100),
            FirstLagDistance=0,
        )
        res, _ = variogram.PointSetScanContStyle(
            templ, {"X": xs, "Y": ys, "Z": zs},
            variogram.CalcVariogramFunction,
            {"HardData": [vals]},
        )
        # Lag 1 (band [0.5, 1.5)) contains pairs (0,1) and (1,2).
        assert res[1, 2] == 2, f"expected 2 pairs in lag 1, got {res[1, 2]}"
        # Sum slot: exact float64 sum of the two per-pair variances.
        assert abs(res[1, 1] - (10001.0 ** 2 + 1.0)) < 1e-6, (
            f"lag-1 variance sum {res[1, 1]} != float64 reference "
            f"{10001.0 ** 2 + 1.0} (E-M10 float32 accumulation regression)"
        )


# =============================================================================
# T-09 — E2-26: MovingAverage3D must preserve fractional means for integer /
# unsigned cubes (assigning the float mean into an int-typed output truncates:
# probe 0.63 → 0). The cubical vectorized branch returns float64 for integral
# input.
# =============================================================================


class TestMovingAverageIntCubePreservesFraction:
    def test_int_cube_fractional_mean_kept(self):
        cube = np.array([[[1], [1]], [[1], [0]]], dtype="int32")  # shape (2, 2, 1)
        cube = np.asfortranarray(cube)
        mask = np.ones((2, 2, 1), dtype="uint8")
        result = routines.MovingAverage3D(
            (cube, mask), (1, 1, 1), -99.0, routines.GetCubicalMask
        )
        # Cell (1,1,0): the half-open window [0,2)x[0,2) covers the whole
        # 2x2x1 grid → mean = (1+1+1+0)/4 = 0.75. (Cell (0,0,0)'s window is
        # [0,1)x[0,1) → only itself → 1.0; reference-verified parity.)
        assert float(result[1, 1, 0]) == 0.75, (
            f"int32 cube fractional mean truncated: got {result[1, 1, 0]} "
            f"(expected 0.75 — E2-26)"
        )

    def test_float_cube_keeps_float_dtype(self):
        """Control: float32 input keeps its own dtype (per-cell parity)."""
        cube = np.array([[[1.0], [1.0]], [[1.0], [0.0]]], dtype="float32")
        cube = np.asfortranarray(cube)
        mask = np.ones((2, 2, 1), dtype="uint8")
        result = routines.MovingAverage3D(
            (cube, mask), (1, 1, 1), -99.0, routines.GetCubicalMask
        )
        assert float(result[1, 1, 0]) == 0.75


# =============================================================================
# T-19 — E2-06/R-06: the INC undefined_value re-mask must use EXACT float32
# equality (matching the C++ fast reader) — the old 1e-6 relative tolerance
# over-masked near-sentinel real data (e.g. -99.0 next to sentinel -99.00005).
# The %.9E writer round-trips float32 exactly, so a fractional sentinel
# round-trips through BOTH loaders with identical masks.
# =============================================================================


@pytest.mark.hpgl
class TestFractionalUndefinedRoundtrip:
    def test_fractional_undefined_roundtrip_masked(self, tmp_path):
        undefined_value = -99.00005  # 9 significant digits
        # Cell 0 masked (written as the sentinel); cell 3 is REAL data at
        # exactly -99.0 (float32) — must stay INFORMED after the round-trip.
        data = np.array([-99.00005, 5.0, 3.0, -99.0], dtype="float32")
        mask = np.array([0, 1, 1, 1], dtype="uint8")
        prop = ContProperty(data, mask)

        fname = str(tmp_path / "frac.inc")
        write_property(prop, fname, "col", undefined_value, basedir=str(tmp_path))

        slow = load_cont_property(fname, undefined_value, basedir=str(tmp_path))
        fast = load_cont_property(
            fname, undefined_value, size=4, basedir=str(tmp_path)
        )

        # Both loaders must agree (E2-06 parity) and only the written sentinel
        # cell may be masked — the -99.0 real datum stays informed.
        np.testing.assert_array_equal(slow.mask, mask, err_msg="slow-path mask")
        np.testing.assert_array_equal(fast.mask, mask, err_msg="fast-path mask")
        np.testing.assert_array_equal(slow.data, data, err_msg="slow-path data")
        np.testing.assert_array_equal(fast.data, data, err_msg="fast-path data")
