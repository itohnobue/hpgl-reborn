"""Regression tests for stage-5/6 CONFIRMED Python findings (FIX pass).

Each test fails against the pre-fix code and passes against the fixed code
(see test docstrings for the pre-fix failure mode).

Covers:
- M-23: __version__ tracks installed metadata (hardcoded fallback only on
  PackageNotFoundError, not unconditionally)
- M-10: SGSConfig/SISConfig radiuses and GTSIMConfig tk_mean/tk_std_dev type
  gates accept numpy scalars and exclude bool (matching sibling gates and
  validate_radius)
- M-20: max_neighbours hard cap aligned with the C++ engine (100000) in
  validation and config classes; warn at 1000 retained
- M-24: gtsim_2ind never mutates the caller's pk_prop.data buffer (both
  overshoot and no-overshoot paths)
- 2-M-12: gtsim_2ind never mutates the caller's prop object (no attribute
  rebind, no in-place transform)
- M-25: sis_simulation config path copies the caller's data dicts (no
  in-place injection of radiuses/max_neighbours)
- 2-M-10: LoadGslibFile streams the parse (bounded intermediate memory,
  identical output on a large synthetic file)
- 2-M-11: validate_seed raises CriticalValidationError for negative seeds
- 2-M-1: sgs_simulation rejects unknown kriging_type with a clear error
- 2-M-13: Python lag-binning uses the C++ projection metric (point-set and
  grid paths match the C-bound kernels)
- M-21: PointSetScanGridStyle skips self-pairs, matching
  PointSetScanContStyle
- 2-M-3: CalcVariograms plumbs an optional seed to the seeded C++ kernel
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


# =============================================================================
# M-23 — __version__ must track installed metadata
# =============================================================================


class TestVersionMetadata:
    def test_version_matches_installed_metadata(self):
        """M-23: __version__ must equal importlib.metadata.version("hpgl")
        when the package is installed.

        Pre-fix: __init__.py:66 unconditionally overwrote __version__ with
        the hardcoded "2.0.1" after the metadata lookup (fallback dedented
        out of the except block), so a version bump in pyproject.toml never
        surfaced and the installed metadata was ignored.
        """
        from importlib.metadata import PackageNotFoundError, version

        import geo_bsd

        try:
            meta = version("hpgl")
        except PackageNotFoundError:
            # Not installed in this environment — the fallback assertion in
            # test_version_fallback_only_on_not_found covers the path.
            pytest.skip("hpgl not installed via importlib.metadata")
        assert geo_bsd.__version__ == meta, (
            f"geo_bsd.__version__ ({geo_bsd.__version__!r}) must match "
            f"importlib.metadata.version('hpgl') ({meta!r})"
        )

    def test_version_fallback_only_on_not_found(self, monkeypatch):
        """M-23: the hardcoded fallback applies ONLY when the metadata lookup
        raises PackageNotFoundError — never unconditionally."""
        import importlib.metadata

        import geo_bsd

        def _raise_not_found(name):
            raise importlib.metadata.PackageNotFoundError(name)

        # Force the fallback path: re-import the package with a metadata
        # lookup that always raises. importlib.reload re-executes __init__.py
        # (submodules are cached, so only the version block re-runs).
        monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
        importlib.reload(geo_bsd)
        assert geo_bsd.__version__ == "2.0.2"
        # Restore the live lookup so subsequent tests see the metadata version.
        monkeypatch.undo()
        importlib.reload(geo_bsd)
        assert isinstance(geo_bsd.__version__, str) and geo_bsd.__version__


# =============================================================================
# M-10 — radiuses/tk type gates accept numpy scalars, exclude bool
# =============================================================================


class TestConfigNumpyRadiusGates:
    def test_sgs_config_numpy_int64_radius_accepted(self):
        from geo_bsd.config import SGSConfig

        cfg = SGSConfig(radiuses=(np.int64(5), 5, 3))
        assert cfg.radiuses == (5, 5, 3)

    def test_sgs_config_numpy_float32_radius_accepted(self):
        from geo_bsd.config import SGSConfig

        cfg = SGSConfig(radiuses=(5.0, np.float32(3.0), 3))
        assert cfg.radiuses[1] == 3.0

    def test_sis_config_numpy_radius_accepted(self):
        from geo_bsd.config import SISConfig

        cfg = SISConfig(radiuses=(np.int64(5), np.float32(3), 3))
        assert cfg.radiuses[0] == 5

    def test_sgs_config_bool_radius_rejected(self):
        """M-10: True is a bool, not a radius value — pre-fix it was accepted
        as radius 1 (bool is an int subclass)."""
        from geo_bsd.config import SGSConfig

        with pytest.raises(TypeError, match="radiuses\\[0\\] must be a number"):
            SGSConfig(radiuses=(True, 5, 3))

    def test_sis_config_bool_radius_rejected(self):
        from geo_bsd.config import SISConfig

        with pytest.raises(TypeError, match="radiuses\\[0\\] must be a number"):
            SISConfig(radiuses=(True, 5, 3))

    def test_gtsim_config_numpy_tk_mean_accepted(self):
        from geo_bsd.config import GTSIMConfig

        cfg = GTSIMConfig(tk_mean=np.float32(0.5), tk_std_dev=np.float32(2.0))
        assert float(cfg.tk_mean) == 0.5
        assert float(cfg.tk_std_dev) == 2.0

    def test_gtsim_config_bool_tk_rejected(self):
        """M-10: True must not be accepted as tk_mean/tk_std_dev."""
        from geo_bsd.config import GTSIMConfig

        with pytest.raises(TypeError, match="tk_mean must be a number"):
            GTSIMConfig(tk_mean=True)
        with pytest.raises(TypeError, match="tk_std_dev must be a number"):
            GTSIMConfig(tk_std_dev=True)

    def test_validate_radius_bool_rejected(self):
        """M-10: validate_radius must reject True (bool) like the config gates."""
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        with pytest.raises(CriticalValidationError, match="Radius"):
            ParameterValidator.validate_radius(True)

    def test_validate_radius_numpy_scalar_accepted(self):
        from geo_bsd.validation import ParameterValidator

        assert ParameterValidator.validate_radius(np.int64(5)) == (5, 5, 5)
        assert ParameterValidator.validate_radius(np.float32(3.0)) == (3, 3, 3)


# =============================================================================
# M-20 — max_neighbours hard cap aligned with the C++ engine (100000)
# =============================================================================


class TestMaxNeighboursHardCap:
    def test_validation_hard_rejects_above_100000(self):
        """M-20: values above the C++ engine bound must be hard-rejected,
        not warn-only (pre-fix: MAX_NEIGHBORS=1000 warned but accepted)."""
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        with pytest.raises(CriticalValidationError, match="maximum allowed"):
            ParameterValidator.validate_max_neighbors(100001)

    def test_validation_accepts_up_to_100000(self):
        from geo_bsd.validation import ParameterValidator

        # Boundary value accepted (may warn — that is the documented
        # performance guidance).
        ParameterValidator.validate_max_neighbors(100000)

    def test_validation_still_warns_above_1000(self):
        """M-20: the 1000 threshold remains a performance-guidance warning."""
        from geo_bsd.validation import ParameterValidator

        with pytest.warns(UserWarning, match="recommended maximum"):
            ParameterValidator.validate_max_neighbors(2000)

    def test_sgs_config_hard_rejects_huge(self):
        """M-20: SGSConfig(max_neighbours=1e9) previously constructed; the
        C++ engine then hard-rejected it with a late RuntimeError."""
        from geo_bsd.config import SGSConfig

        with pytest.raises(ValueError, match="maximum allowed"):
            SGSConfig(max_neighbours=10**9)

    def test_sis_config_hard_rejects_huge(self):
        from geo_bsd.config import SISConfig

        with pytest.raises(ValueError, match="maximum allowed"):
            SISConfig(max_neighbours=100001)

    def test_config_boundary_100000_accepted(self):
        from geo_bsd.config import SGSConfig

        cfg = SGSConfig(max_neighbours=100000)
        assert cfg.max_neighbours == 100000


# =============================================================================
# 2-M-11 — validate_seed raises CriticalValidationError
# =============================================================================


class TestValidateSeedCritical:
    def test_negative_seed_raises_critical(self):
        """2-M-11: validate_seed must raise CriticalValidationError, matching
        the sgs/sis docstring contract and sibling validators. Pre-fix it
        raised the base ValidationError, so callers catching only
        CriticalValidationError missed negative seeds."""
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        with pytest.raises(CriticalValidationError, match="negative"):
            ParameterValidator.validate_seed(-1)

    def test_negative_seed_not_base_only(self):
        """The raised exception must be CriticalValidationError specifically
        (the base-class match would pass pre-fix too)."""
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        try:
            ParameterValidator.validate_seed(-5)
        except CriticalValidationError:
            pass  # expected
        else:
            raise AssertionError("validate_seed(-5) did not raise CriticalValidationError")


# =============================================================================
# M-24 / 2-M-12 — gtsim_2ind never mutates caller-owned data
# =============================================================================


@pytest.mark.hpgl
class TestGtsim2IndNoMutation:
    """M-24 + 2-M-12: gtsim_2ind must not mutate the caller's pk_prop buffer
    or the caller's prop object on EITHER the overshoot or no-overshoot path."""

    def _make_grid_prop(self, x=5, y=5, z=2):
        import geo_bsd.geo as geo

        np.random.seed(42)
        grid = geo.SugarboxGrid(x=x, y=y, z=z)
        size = x * y * z
        data = np.where(np.random.rand(size) < 0.6, 0.0, 1.0).astype("float32")
        mask = np.ones(size, dtype="uint8")
        prop = geo.ContProperty(data, mask)
        return grid, prop

    def _make_sk_params(self):
        import geo_bsd.geo as geo

        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        return {"radiuses": (3, 3, 2), "max_neighbours": 8, "cov_model": cov_model}

    def test_pk_prop_data_unchanged_no_overshoot(self):
        """M-24: with all pk probabilities inside [0,1] (no-overshoot path),
        the caller's pk_prop.data buffer must be unchanged after gtsim_2ind.

        Pre-fix: tk_calculation wrote the inverse-CDF thresholds into the
        caller's array in place via the ravel('K') view; the attribute
        restore hid the damage from pk_prop but the external buffer was
        corrupted."""
        from geo_bsd.geo import ContProperty
        from geo_bsd.gtsim import gtsim_2ind

        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        orig = pk_data.copy()
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        assert isinstance(result, ContProperty)
        np.testing.assert_array_equal(pk_data, orig, err_msg="no-overshoot buffer corrupted")
        # The caller's pk_prop.data attribute must not have been rebound either.
        assert pk_prop.data is pk_data

    def test_pk_prop_data_unchanged_overshoot(self):
        """M-24: with legitimate kriging overshoots (values outside [0,1]),
        the caller's pk_prop.data buffer must be unchanged."""
        from geo_bsd.geo import ContProperty
        from geo_bsd.gtsim import gtsim_2ind

        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        pk_data = np.full(prop.data.size, 0.5, dtype="float32")
        pk_data[0] = -0.05
        pk_data[1] = 1.05
        orig = pk_data.copy()
        pk_prop = ContProperty(pk_data, np.ones(prop.data.size, dtype="uint8"))

        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        assert isinstance(result, ContProperty)
        np.testing.assert_array_equal(pk_data, orig, err_msg="overshoot buffer corrupted")
        assert pk_prop.data is pk_data

    def test_caller_prop_object_unchanged(self):
        """2-M-12: the caller's prop object must be untouched — its .data
        attribute must not be REBOUND and its content must not be transformed.

        Pre-fix: `prop.data = prop.data.copy()` rebound the caller's
        attribute and pseudo_gaussian_transform transformed the rebound array
        in place, despite a comment claiming "avoid mutating caller's data"."""
        from geo_bsd.geo import ContProperty
        from geo_bsd.gtsim import gtsim_2ind

        grid, prop = self._make_grid_prop()
        sk_params = self._make_sk_params()

        orig_data = prop.data.copy()
        orig_ref = prop.data

        pk_prop = ContProperty(
            np.full(prop.data.size, 0.5, dtype="float32"),
            np.ones(prop.data.size, dtype="uint8"),
        )
        result = gtsim_2ind(grid, prop, sk_params, do_sk=False, pk_prop=pk_prop, seed=42)
        assert isinstance(result, ContProperty)
        # Attribute must not have been rebound.
        assert prop.data is orig_ref, "caller's prop.data attribute was rebound"
        # Content must be untouched (pre-fix: transformed to uniform values).
        np.testing.assert_array_equal(prop.data, orig_data)


# =============================================================================
# M-25 — sis_simulation config path copies the caller's data dicts
# =============================================================================


@pytest.mark.hpgl
class TestSisConfigNoDictMutation:
    def test_config_path_does_not_mutate_caller_dicts(self):
        """M-25: with a SISConfig, sis_simulation must NOT inject
        radiuses/max_neighbours into the caller's data dicts in place.

        Pre-fix: the injection loop wrote config values into the caller-owned
        dicts, leaving stale values behind for dict reuse."""
        import geo_bsd.geo as geo
        from geo_bsd.config import SISConfig
        from geo_bsd.sis import sis_simulation

        grid = geo.SugarboxGrid(x=8, y=8, z=2)
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(7)
        indicator_count = 2
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[::3] = 0  # sparse informed lattice
        prop = geo.IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        # Caller dicts WITHOUT radiuses/max_neighbours — the config is
        # expected to supply them (on copies).
        caller_data = [
            {"cov_model": cov_model},
            {"cov_model": cov_model},
        ]
        before = [dict(d) for d in caller_data]

        config = SISConfig(seed=42, radiuses=(3, 3, 2), max_neighbours=12)
        result = sis_simulation(
            prop=prop, grid=grid, data=caller_data,
            seed=42, marginal_probs=[0.5, 0.5], config=config,
        )
        assert result.indicator_count == 2
        # The caller's dicts must be unchanged (no injected keys).
        assert caller_data == before, (
            "sis_simulation mutated the caller's data dicts in place"
        )
        for ikd in caller_data:
            assert "radiuses" not in ikd
            assert "max_neighbours" not in ikd

    def test_config_path_without_config_no_mutation(self):
        """Control: without a config, sis_simulation must not touch the
        caller's dicts either."""
        import geo_bsd.geo as geo
        from geo_bsd.sis import sis_simulation

        grid = geo.SugarboxGrid(x=8, y=8, z=2)
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(7)
        indicator_count = 2
        data = rng.randint(0, indicator_count, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[::3] = 0
        prop = geo.IndProperty(data, mask, indicator_count)
        prop.fix_shape(grid)

        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )
        caller_data = [
            {"cov_model": cov_model, "radiuses": (3, 3, 2), "max_neighbours": 12},
            {"cov_model": cov_model, "radiuses": (3, 3, 2), "max_neighbours": 12},
        ]
        before = [dict(d) for d in caller_data]
        result = sis_simulation(
            prop=prop, grid=grid, data=caller_data,
            seed=42, marginal_probs=[0.5, 0.5],
        )
        assert result.indicator_count == 2
        assert caller_data == before


# =============================================================================
# 2-M-10 — LoadGslibFile streams the parse (bounded memory)
# =============================================================================


class TestLoadGslibFileStreaming:
    def _write_gslib(self, path, num_props, names, rows):
        lines = ["Test caption", str(num_props)] + names
        for row in rows:
            lines.append(" ".join(str(v) for v in row))
        path.write_text("\n".join(lines) + "\n")

    def test_large_file_matches_reference(self, tmp_path):
        """2-M-10: a moderately large synthetic file must load with the same
        output as the pre-fix list-of-lists conversion, without OOM."""
        from geo_bsd.routines import LoadGslibFile

        nx, ny, nz = 200, 100, 5  # 100k cells
        grid_size = nx * ny * nz
        num_props = 4
        rng = np.random.RandomState(0)
        data = rng.rand(grid_size, num_props) * 100.0

        fpath = tmp_path / "large.gslib"
        self._write_gslib(fpath, num_props, ["P1", "P2", "P3", "P4"], data)

        result = LoadGslibFile(str(fpath), property_size=(nx, ny, nz), basedir=str(tmp_path))
        assert set(result.keys()) == {"P1", "P2", "P3", "P4"}
        for j, name in enumerate(["P1", "P2", "P3", "P4"]):
            expected = data[:, j].reshape((nx, ny, nz), order="F")
            np.testing.assert_allclose(result[name], expected, rtol=1e-6)

    def test_streaming_preserves_error_semantics(self, tmp_path):
        """2-M-10: streaming must preserve the existing row-count error paths."""
        from geo_bsd.routines import LoadGslibFile

        fpath = tmp_path / "trunc.gslib"
        fpath.write_text(
            "caption\n2\nA\nB\n1.0 2.0\n3.0 4.0\n"
        )  # 2 values, expects 4 for a 2x2x1 grid
        with pytest.raises(RuntimeError, match="expected 4"):
            LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))


# =============================================================================
# 2-M-1 — sgs_simulation rejects unknown kriging_type
# =============================================================================


@pytest.mark.hpgl
class TestSgsKrigingTypeValidation:
    def test_unknown_kriging_type_rejected(self):
        """2-M-1 (Python part): unknown kriging_type values are rejected with
        a clear ValueError instead of silently running ordinary kriging."""
        import geo_bsd.geo as geo
        from geo_bsd.sgs import sgs_simulation

        grid = geo.SugarboxGrid(x=4, y=4, z=2)
        size = grid.x * grid.y * grid.z
        prop = geo.ContProperty(
            np.full(size, 0.5, dtype="float32"), np.ones(size, dtype="uint8")
        )
        prop.fix_shape(grid)
        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(2.0, 2.0, 2.0), sill=1.0, nugget=0.1
        )
        with pytest.raises(ValueError, match="invalid kriging_type"):
            sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(2, 2, 2), max_neighbours=8, cov_model=cov_model,
                seed=42, kriging_type="lvm",
            )

    def test_valid_kriging_types_accepted(self):
        import geo_bsd.geo as geo
        from geo_bsd.sgs import sgs_simulation

        grid = geo.SugarboxGrid(x=4, y=4, z=2)
        size = grid.x * grid.y * grid.z
        prop = geo.ContProperty(
            np.full(size, 0.5, dtype="float32"), np.ones(size, dtype="uint8")
        )
        prop.fix_shape(grid)
        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(2.0, 2.0, 2.0), sill=1.0, nugget=0.1
        )
        # sk and ok both construct the SGS params structure without error.
        for kt in ("sk", "ok"):
            result = sgs_simulation(
                prop=prop, grid=grid, cdf_data=None,
                radiuses=(2, 2, 2), max_neighbours=8, cov_model=cov_model,
                seed=42, kriging_type=kt,
            )
            assert np.all(np.isfinite(result.data))


# =============================================================================
# 2-M-13 — Python lag-binning matches the C++ projection metric
# =============================================================================


@pytest.mark.hpgl
class TestVariogramProjectionMetric:
    """2-M-13: the Python pure-Python scans must bin lags by the directional
    projection onto the principal anisotropy axis (variograms.cpp:647, 805),
    not by raw Euclidean distance."""

    def _aniso_diagonal(self):
        """Diagonal points (0,0),(1,1),(2,2),(3,3): projection onto X
        (|dx|) differs from Euclidean distance (sqrt(2)*|dx|) — this
        discriminates the two metrics."""
        xs = np.array([0, 1, 2, 3], dtype="float32")
        ys = np.array([0, 1, 2, 3], dtype="float32")
        zs = np.zeros(4, dtype="float32")
        vals = np.array([10.0, 20.0, 30.0, 40.0], dtype="float32")
        return xs, ys, zs, vals

    def test_point_set_cont_style_matches_cpp_projection(self):
        """2-M-13 (point-set path): PointSetScanContStyle must produce the
        same lag-bin values as the C++ calc_variograms_from_point_set for
        data where Euclidean distance and projection diverge.

        Pre-fix (Euclidean): [0, 50, 0, 200]; C++ (projection): [0, 50, 200,
        450] — the (2,2) and (3,3) pairs binned into the wrong lags / were
        dropped."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import (
            CalcVariogramsFromPointSet,
            Ellipsoid,
            VariogramSearchTemplate,
        )
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanContStyle,
            TVEllipsoid,
            TVVariogramSearchTemplate,
        )

        xs, ys, zs, vals = self._aniso_diagonal()

        c_ell = Ellipsoid(R1=100, R2=100, R3=100, azimuth=0, dip=0, rotation=0)
        c_templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=4, first_lag_distance=0.0, ellipsoid=c_ell,
        )
        c_var = np.zeros(4, dtype="float32")
        _, cpp = CalcVariogramsFromPointSet(
            c_templ, {"X": xs, "Y": ys, "Z": zs, "Property": vals}, c_var
        )

        p_ell = TVEllipsoid(R1=100, R2=100, R3=100)
        p_templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=4, Ellipsoid=p_ell, FirstLagDistance=0,
        )
        py_res, _ = PointSetScanContStyle(
            p_templ, {"X": xs, "Y": ys, "Z": zs}, CalcVariogramFunction,
            {"HardData": [vals]},
        )

        np.testing.assert_allclose(py_res[:, 0], cpp, atol=1e-4)
        # Pin the projection-metric values — the pre-fix Euclidean scan gave
        # [0, 50, 0, 200] here.
        np.testing.assert_allclose(py_res[:, 0], [0.0, 50.0, 200.0, 450.0], atol=1e-4)

    def test_grid_scan_matches_cpp_grid_kernel(self):
        """2-M-13 (grid path): CubeScan must produce the same lag-bin values
        as the C++ calc_variograms grid kernel for the same template/data.

        Pre-fix (Euclidean) lag 2 diverged from the C++ projection kernel
        (202.0 vs 319.9) on a 5x5x5 flat-index grid."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import CalcVariograms, Ellipsoid, VariogramSearchTemplate
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            CubeScan,
            TVEllipsoid,
            TVVariogramSearchTemplate,
        )

        nx = ny = nz = 5
        data3 = np.arange(nx * ny * nz, dtype="float32").reshape(nx, ny, nz, order="F")
        mask3 = np.ones((nx, ny, nz), dtype="uint8")

        c_ell = Ellipsoid(R1=5, R2=5, R3=5, azimuth=0, dip=0, rotation=0)
        c_templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=c_ell,
        )
        _, cpp = CalcVariograms(c_templ, [data3, mask3])

        p_ell = TVEllipsoid(R1=5, R2=5, R3=5)
        p_templ = TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=3, Ellipsoid=p_ell, FirstLagDistance=0,
        )
        py_res, _ = CubeScan(p_templ, mask3, CalcVariogramFunction, {"HardData": [data3]})

        np.testing.assert_allclose(py_res[:, 0], cpp, atol=1e-3)


# =============================================================================
# M-21 — PointSetScanGridStyle skips self-pairs like PointSetScanContStyle
# =============================================================================


class TestGridStyleSelfPairSkip:
    def test_grid_style_matches_cont_style_counts(self):
        """M-21: after skipping self-pairs, PointSetScanGridStyle must
        produce identical lag-0/lag-1 counts to PointSetScanContStyle for the
        same grid-aligned data.

        Pre-fix: GridStyle counted each point's self-pair in lag 0, diluting
        the lag-0 variogram (22.22 vs 50.0 in the F-M24 reproduction)."""
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanContStyle,
            PointSetScanGridStyle,
            TVEllipsoid,
            TVVariogramSearchTemplate,
        )

        grid_pts = np.array([0.0, 1.0, 2.0], dtype="float32")
        px, py, pz = np.meshgrid(grid_pts, grid_pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(9, dtype="float32")

        templ = TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=1.0, TolDistance=1.0, NumLags=3,
            Ellipsoid=TVEllipsoid(R1=100, R2=100, R3=100), FirstLagDistance=0,
        )

        cont, _ = PointSetScanContStyle(
            templ, {"X": px, "Y": py, "Z": pz}, CalcVariogramFunction,
            {"HardData": [pvals]},
        )
        grid_style, _ = PointSetScanGridStyle(
            templ, (px, py, pz), CalcVariogramFunction, {"HardData": [pvals]},
        )

        # Identical pair counts per lag (count slot index 2*NumValues = 2).
        np.testing.assert_array_equal(grid_style[:, 2], cont[:, 2])
        np.testing.assert_array_equal(grid_style[:, 0], cont[:, 0])
        # The self-pair must NOT inflate lag 0: with 9 points, lag 0 has
        # zero real pairs (all offsets non-zero have projection >= 1 > 0.5).
        assert grid_style[0, 2] == 0, "lag 0 must not count self-pairs"
        assert grid_style[1, 2] > 0


# =============================================================================
# 2-M-3 — CalcVariograms optional seed plumbing
# =============================================================================


class TestCalcVariogramsSeedPlumbing:
    def test_seed_none_preserves_current_behavior(self):
        """2-M-3: seed=None (default) keeps the existing non-seeded path."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import CalcVariograms, Ellipsoid, VariogramSearchTemplate

        data = np.ones((4, 4, 4), dtype="float32") * 3.0
        mask = np.ones((4, 4, 4), dtype="uint8")
        ell = Ellipsoid(R1=4, R2=4, R3=4, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        lags, variogram = CalcVariograms(templ, [data, mask], percent=100)
        assert len(lags) == 3
        assert np.all(np.isfinite(variogram))

    def test_seed_type_and_range_validated(self):
        """2-M-3: an invalid seed is rejected before reaching C++."""
        pytest.importorskip("geo_bsd.cvariogram")
        from geo_bsd.cvariogram import CalcVariograms, Ellipsoid, VariogramSearchTemplate

        data = np.ones((4, 4, 4), dtype="float32") * 3.0
        mask = np.ones((4, 4, 4), dtype="uint8")
        ell = Ellipsoid(R1=4, R2=4, R3=4, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        with pytest.raises(TypeError, match="seed must be an int"):
            CalcVariograms(templ, [data, mask], seed="not-an-int")
        with pytest.raises(ValueError, match="non-negative"):
            CalcVariograms(templ, [data, mask], seed=-1)

    def test_seed_requires_seeded_cpp_kernel(self):
        """2-M-3: passing a seed requires the _cvariogram library to export
        the seeded C++ kernel (calc_variograms_seeded). Until the C++ agent
        lands that symbol, the plumbing reports a clear RuntimeError instead
        of silently ignoring the seed."""
        pytest.importorskip("geo_bsd.cvariogram")
        import geo_bsd.cvariogram as cv
        from geo_bsd.cvariogram import CalcVariograms, Ellipsoid, VariogramSearchTemplate

        data = np.ones((4, 4, 4), dtype="float32") * 3.0
        mask = np.ones((4, 4, 4), dtype="uint8")
        ell = Ellipsoid(R1=4, R2=4, R3=4, azimuth=0, dip=0, rotation=0)
        templ = VariogramSearchTemplate(
            lag_width=1.0, lag_separation=1.0, tol_distance=1.0,
            num_lags=3, first_lag_distance=0.0, ellipsoid=ell,
        )
        if hasattr(cv.cvar, "calc_variograms_seeded"):
            # Seeded kernel available: the call must succeed and be
            # deterministic for an identical seed.
            lags1, v1 = CalcVariograms(templ, [data, mask], percent=100, seed=12345)
            lags2, v2 = CalcVariograms(templ, [data, mask], percent=100, seed=12345)
            np.testing.assert_array_equal(v1, v2)
        else:
            with pytest.raises(RuntimeError, match="calc_variograms_seeded"):
                CalcVariograms(templ, [data, mask], percent=100, seed=12345)


# =============================================================================
# R-1 (H-1) — pure-Python point-set scans carry a total pair-lag work cap
# =============================================================================


class TestPointSetPairLagWorkCap:
    """R-1 (H-1): PointSetScanContStyle / PointSetScanGridStyle must reject
    an input whose estimated pair-lag work (n^2 * NumLags) exceeds
    MAX_TOTAL_PAIR_LAG_WORK before running the O(n^2) scan. The Python cap
    is 1e8 — deliberately lower than the C++ kernel's 1e12 (variograms.cpp:40)
    so pure-Python worst-case runtime stays ~2.5 s instead of hours-days.
    Pre-fix only MAX_POINT_SET_SIZE was checked, so a legal 1e6-point set
    with a large window ran up to 1e16 pair-lag Python iterations
    (effectively-infinite loop)."""

    @staticmethod
    def _template(num_lags):
        from geo_bsd.variogram import TVEllipsoid, TVVariogramSearchTemplate

        return TVVariogramSearchTemplate(
            LagWidth=2.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=num_lags,
            Ellipsoid=TVEllipsoid(R1=100, R2=100, R3=100), FirstLagDistance=0,
        )

    def test_cont_style_rejects_legal_large_work(self):
        """A LEGAL input (10001 points <= MAX_POINT_SET_SIZE, 1 lag <=
        MAX_NUM_LAGS) whose pair-lag work 10001^2*1 = 1.0002e8 just exceeds
        the pure-Python cap (MAX_TOTAL_PAIR_LAG_WORK = 1e8) must raise
        instead of hanging."""
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanContStyle,
        )

        n = 10001
        xs = np.zeros(n, dtype="float32")
        ys = np.zeros(n, dtype="float32")
        zs = np.zeros(n, dtype="float32")
        vals = np.zeros(n, dtype="float32")
        with pytest.raises(ValueError, match="pair-lag work"):
            PointSetScanContStyle(
                self._template(1),
                {"X": xs, "Y": ys, "Z": zs},
                CalcVariogramFunction,
                {"HardData": [vals]},
            )

    def test_grid_style_rejects_over_cap(self, monkeypatch):
        """GridStyle rejects the same work estimate (monkeypatched small cap
        keeps the test fast; the estimate is n^2 * NumLags)."""
        import geo_bsd.variogram as v
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanGridStyle,
        )

        monkeypatch.setattr(v, "MAX_TOTAL_PAIR_LAG_WORK", 1000.0)
        # 10x10 grid-aligned points = 100 pts; 100^2 * 1 lag = 10000 > 1000.
        pts = np.arange(10, dtype="float32")
        px, py, pz = np.meshgrid(pts, pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(100, dtype="float32")
        with pytest.raises(ValueError, match="pair-lag work"):
            PointSetScanGridStyle(
                self._template(1), (px, py, pz), CalcVariogramFunction,
                {"HardData": [pvals]},
            )

    def test_grid_style_rejects_real_cap(self):
        """GridStyle with the REAL pure-Python cap (1e8): a LEGAL 10100-point
        grid (<= MAX_POINT_SET_SIZE) with 1 lag gives pair-lag work
        10100^2*1 = 1.0201e8 > 1e8 and must raise instead of hanging."""
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanGridStyle,
        )

        px_pts = np.arange(100, dtype="float32")
        py_pts = np.arange(101, dtype="float32")
        px, py, pz = np.meshgrid(px_pts, py_pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(10100, dtype="float32")
        with pytest.raises(ValueError, match="pair-lag work"):
            PointSetScanGridStyle(
                self._template(1), (px, py, pz), CalcVariogramFunction,
                {"HardData": [pvals]},
            )

    def test_cont_style_below_cap_still_works(self):
        """Control: a small point set below the cap still runs and produces
        a finite variogram."""
        from geo_bsd.variogram import (
            CalcVariogramFunction,
            PointSetScanContStyle,
        )

        grid_pts = np.array([0.0, 1.0, 2.0], dtype="float32")
        px, py, pz = np.meshgrid(grid_pts, grid_pts, np.array([0.0], dtype="float32"))
        px = px.ravel().astype("float32")
        py = py.ravel().astype("float32")
        pz = pz.ravel().astype("float32")
        pvals = np.arange(9, dtype="float32")
        res, _ = PointSetScanContStyle(
            self._template(3), {"X": px, "Y": py, "Z": pz},
            CalcVariogramFunction, {"HardData": [pvals]},
        )
        assert np.all(np.isfinite(res))


# =============================================================================
# R-2 — sgs_simulation "ok" kriging honors the user mean on the fallback
# =============================================================================


@pytest.mark.hpgl
class TestSgsOkMeanHonored:
    """R-2: with kriging_type="ok", the user-supplied mean IS applied on the
    failure fallback — nodes that cannot be kriged draw from N(mean, 1.0)
    (GSLIB sgsim `cmean = gmean; cstdev = 1.0`), so the OK-mode output
    depends on the mean. Pre-fix the docstring claimed "an explicit mean has
    no effect on the OK branch" — empirically false (M-29 + api.cpp:1026 +
    sequential_simulation.h:181-182)."""

    def test_ok_mode_mean_affects_output(self):
        import geo_bsd.geo as geo
        from geo_bsd.sgs import sgs_simulation

        # Sparse 8x8x1 grid, 2 informed cells at opposite corners, tiny
        # radius -> most cells cannot be kriged and hit the fallback.
        grid = geo.SugarboxGrid(x=8, y=8, z=1)
        n_total = grid.x * grid.y * grid.z
        data = np.zeros(n_total, dtype="float32")
        mask = np.zeros(n_total, dtype="uint8")
        data[0] = 10.0
        mask[0] = 1
        data[63] = 20.0
        mask[63] = 1
        prop = geo.ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        geo._last_kriging_stats = None
        out_p = sgs_simulation(
            prop=prop, grid=grid, cdf_data=None, radiuses=(1, 1, 1),
            max_neighbours=4, cov_model=cov_model, seed=12345,
            kriging_type="ok", mean=100.0, use_harddata=True,
        )
        stats = geo._last_kriging_stats or {}
        assert stats.get("points_without_neighbours", 0) > 0, (
            "test precondition: the fallback must fire for the mean to act"
        )

        out_m = sgs_simulation(
            prop=prop, grid=grid, cdf_data=None, radiuses=(1, 1, 1),
            max_neighbours=4, cov_model=cov_model, seed=12345,
            kriging_type="ok", mean=-100.0, use_harddata=True,
        )

        # The user mean shifts the fallback draws -> the OK-mode output mean
        # differs substantially between mean=+100 and mean=-100 (empirically
        # ~134 on this fixture; assert a wide margin to stay robust).
        assert not np.array_equal(out_p.data, out_m.data)
        assert float(np.mean(out_p.data)) > float(np.mean(out_m.data)) + 50.0


# =============================================================================
# R-12 — LoadGslibFile truncation path must not allocate the full grid
# =============================================================================


class TestLoadGslibFileTruncationMemory:
    """R-12: the truncation error path must fail BEFORE allocating the full
    output array. Pre-fix, numpy.empty((grid_size, num_p)) was allocated up
    front, so a truncated file with a large declared grid allocated 1.6-8 GB
    and then raised."""

    def test_truncated_large_grid_raises_without_huge_allocation(self, tmp_path):
        import tracemalloc

        from geo_bsd.routines import LoadGslibFile

        nx, ny, nz = 1000, 1000, 100  # 1e8 cells declared
        fpath = tmp_path / "trunc_large.gslib"
        fpath.write_text("caption\n2\nA\nB\n1.0 2.0\n3.0 4.0\n")

        tracemalloc.start()
        try:
            with pytest.raises(RuntimeError, match="expected 100000000"):
                LoadGslibFile(str(fpath), property_size=(nx, ny, nz), basedir=str(tmp_path))
        finally:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

        # Pre-fix peak was ~1.6 GB (the full (1e8, 2) float64 array); the
        # fix counts rows first, so the error fires at KBs of memory.
        assert peak < 16 * 1024 * 1024, (
            f"truncation path allocated {peak / 1e6:.1f} MB (expected < 16 MB)"
        )

    def test_truncated_small_grid_still_raises(self, tmp_path):
        """The existing row-count error message is preserved."""
        from geo_bsd.routines import LoadGslibFile

        fpath = tmp_path / "trunc_small.gslib"
        fpath.write_text("caption\n1\nA\n1.0\n2.0\n")
        with pytest.raises(RuntimeError, match="expected 3"):
            LoadGslibFile(str(fpath), property_size=(1, 1, 3), basedir=str(tmp_path))

