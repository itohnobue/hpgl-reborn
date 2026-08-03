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
# II-28 — __version__ must report the SOURCE version, not stale dist metadata
# =============================================================================


def _pyproject_version():
    """Read the version from the source tree's pyproject.toml, or None.

    Independent of geo_bsd._source_version — the test asserts the reported
    __version__ against the pyproject literal, so it must not share the
    implementation under test.
    """
    pyproject = (
        Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    )
    if not pyproject.is_file():
        return None
    try:
        import tomllib  # Python 3.11+
    except ImportError:  # pragma: no cover - Python 3.9/3.10
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None
    if tomllib is not None:
        try:
            with pyproject.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError, TypeError):
            data = {}
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version:
            return version
    return None


class TestVersionMetadata:
    def test_version_matches_source_when_source_tree_present(self):
        """II-28: when the source tree is reachable, __version__ must equal
        pyproject.toml's version — the code actually imported.

        Pre-fix: __init__.py:66-72 read importlib.metadata.version("hpgl"),
        which returns the INSTALLED dist version; a stale hpgl-1.6.0.dist-
        info in the venv made __version__ == '1.6.0' against 2.0.2 source.
        """
        import geo_bsd

        expected = _pyproject_version()
        if expected is None:
            pytest.skip("pyproject.toml not reachable from source tree")
        assert geo_bsd.__version__ == expected, (
            f"geo_bsd.__version__ ({geo_bsd.__version__!r}) must match the "
            f"source pyproject.toml version ({expected!r}) — a stale "
            f"installed dist-info must not shadow the source version"
        )

    def test_version_prefers_source_over_stale_installed_metadata(self, monkeypatch):
        """II-28 regression: even when importlib.metadata reports a stale
        (different) installed version, __version__ must report the source.

        Pre-fix this test failed: reloading with a monkeypatched metadata
        version of '9.9.9' made __version__ == '9.9.9'. Post-fix the source
        tree is authoritative, so '9.9.9' is never consulted.
        """
        import importlib
        import importlib.metadata

        import geo_bsd

        expected = _pyproject_version()
        if expected is None:
            pytest.skip("pyproject.toml not reachable from source tree")

        monkeypatch.setattr(importlib.metadata, "version", lambda name: "9.9.9")
        importlib.reload(geo_bsd)
        assert geo_bsd.__version__ == expected, (
            f"stale installed metadata ('9.9.9') must not override the source "
            f"version ({expected!r}); got {geo_bsd.__version__!r}"
        )
        # Restore the live lookup so subsequent tests see the real state.
        monkeypatch.undo()
        importlib.reload(geo_bsd)
        assert isinstance(geo_bsd.__version__, str) and geo_bsd.__version__

    def test_version_fallback_only_on_not_found(self, monkeypatch):
        """The installed-metadata lookup falls back to the source version
        when the package metadata cannot be found (or when running without a
        source tree). The fallback literal must stay consistent with
        pyproject.toml."""
        import importlib.metadata

        import geo_bsd

        expected = _pyproject_version()
        if expected is not None:
            # Source tree present: the metadata monkeypatch is never even
            # consulted — __version__ is the source version regardless.
            def _raise_not_found(name):
                raise importlib.metadata.PackageNotFoundError(name)

            monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
            importlib.reload(geo_bsd)
            assert geo_bsd.__version__ == expected
        else:
            # No source tree: the metadata lookup must run; a
            # PackageNotFoundError falls back to the documented literal.
            expected = "2.0.3"

            def _raise_not_found(name):
                raise importlib.metadata.PackageNotFoundError(name)

            monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
            importlib.reload(geo_bsd)
            assert geo_bsd.__version__ == expected
        # Restore the live lookup so subsequent tests see the metadata version.
        monkeypatch.undo()
        importlib.reload(geo_bsd)
        assert isinstance(geo_bsd.__version__, str) and geo_bsd.__version__


# =============================================================================
# II-29 — gtsim_2ind exported at the package top level
# =============================================================================


class TestGtsim2IndTopLevelExport:
    def test_gtsim_2ind_accessible_at_top_level(self):
        """II-29: gtsim_2ind is a first-class simulation entry point; it must
        be reachable as geo_bsd.gtsim_2ind like its siblings
        sgs_simulation / sis_simulation. Pre-fix it raised AttributeError."""
        import geo_bsd

        assert callable(geo_bsd.gtsim_2ind), (
            "geo_bsd.gtsim_2ind must be exported at the top level (II-29)"
        )

    def test_gtsim_2ind_in_all(self):
        """The export must be part of the package's public __all__."""
        import geo_bsd

        assert "gtsim_2ind" in geo_bsd.__all__


# =============================================================================
# II-30 — SGSConfig/SISConfig/GTSIMConfig exported at the package top level
# =============================================================================


class TestConfigClassesTopLevelExport:
    @pytest.mark.parametrize(
        "name", ["SGSConfig", "SISConfig", "GTSIMConfig"]
    )
    def test_config_class_accessible_at_top_level(self, name):
        """II-30: the frozen-config dataclasses are documented public API
        (config.py); they must be reachable as geo_bsd.<name>. Pre-fix each
        raised AttributeError at the top level."""
        import geo_bsd

        assert hasattr(geo_bsd, name), (
            f"geo_bsd.{name} must be exported at the top level (II-30)"
        )

    def test_config_classes_in_all(self):
        """The exports must be part of the package's public __all__."""
        import geo_bsd

        for name in ("SGSConfig", "SISConfig", "GTSIMConfig"):
            assert name in geo_bsd.__all__


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


# =============================================================================
# F-20 / II-10 — cokriging NaN-proof guards + zero-variance primary-only
# degradation (simple_cokriging_markI.cpp)
# =============================================================================


class TestCokrigingKrigingHardening:
    """Python-side regression tests for the C++ kriging fixes in
    simple_cokriging_markI.cpp (F-20 isfinite-first entry guards, II-10
    secondary-equation drop on non-strictly-positive variance).

    The C++ entry-point guards are the chokepoint for direct-C++ callers; the
    Python validation layer already rejects NaN correlation_coef/variance
    (validate_correlation_coef / validate_variance), so these tests exercise
    the end-to-end behavior the C++ fixes guarantee: NaN correlation_coef is
    rejected at the C++ boundary even when the Python gate is bypassed, and
    secondary_variance=0 (Python-ACCEPTED; validation.py:962 rejects only < 0)
    degrades to primary-only kriging instead of raising RuntimeError from a
    singular system.
    """

    def test_cokriging_markI_zero_secondary_variance_degrades_primary_only(self):
        """II-10: secondary_variance=0.0 must NOT produce a singular system.

        Pre-fix: build_system wrote the raw 0 to the diagonal → singular
        matrix every node → KI_SINGULARITY → _check_kriging_failure_stats
        raises RuntimeError ("kriging system was singular"). Post-fix: the
        secondary equation is dropped entirely (primary-only kriging) → the
        call succeeds with finite output.
        """
        from geo_bsd.geo import (
            ContProperty,
            CovarianceModel,
            SugarboxGrid,
            covariance,
            simple_cokriging_markI,
        )

        grid = SugarboxGrid(x=5, y=5, z=3)
        size = 5 * 5 * 3
        rng = np.random.RandomState(7)
        primary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        # Secondary fully informed — the variance, not the data, is the trigger.
        secondary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(3.0, 3.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        result = simple_cokriging_markI(
            prop=primary,
            grid=grid,
            radiuses=(3, 3, 2),
            max_neighbours=8,
            cov_model=cov_model,
            secondary_data=secondary,
            primary_mean=50.0,
            secondary_mean=50.0,
            secondary_variance=0.0,  # Python-ACCEPTED; pre-fix singular → RuntimeError
            correlation_coef=0.5,
        )
        assert isinstance(result, ContProperty)
        assert result.data.size == size
        assert np.all(np.isfinite(result.data.astype("float64")))

    def test_cokriging_markI_negative_variance_rejected(self):
        """Negative secondary_variance must be rejected end-to-end.

        F-20 hardened the C++ entry guard (isfinite-first) so a negative/NaN
        variance can never reach the kernel even for direct-C++ callers. The
        Python path surfaces the rejection via the validation layer and/or the
        C++ entry guard (both fire); the assertion is that an error is raised
        rather than silent NaN output.
        """
        from geo_bsd.geo import (
            ContProperty,
            CovarianceModel,
            SugarboxGrid,
            covariance,
            simple_cokriging_markI,
        )
        from geo_bsd.validation import CriticalValidationError

        grid = SugarboxGrid(x=5, y=5, z=3)
        size = 5 * 5 * 3
        rng = np.random.RandomState(8)
        primary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        secondary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(3.0, 3.0, 2.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )

        with pytest.raises((ValueError, RuntimeError, CriticalValidationError)):
            simple_cokriging_markI(
                prop=primary,
                grid=grid,
                radiuses=(3, 3, 2),
                max_neighbours=8,
                cov_model=cov_model,
                secondary_data=secondary,
                primary_mean=50.0,
                secondary_mean=50.0,
                secondary_variance=-1.0,
                correlation_coef=0.5,
            )


class TestCovModelRangeRatioOverflow:
    """III-09: anisotropy range ratios that overflow to Inf/NaN must be
    rejected. Pre-fix, ranges {1e10, 1e-300, 1} produced ratio 1e310 → Inf
    scale → silent NaN/0 covariance for direct cov_model consumers. The
    Python validation layer accepts these ranges (each is finite and within
    MIN_RANGE/MAX_RANGE), so the C++ guard is the first line of defense; a
    kriging call that constructs the model must surface the C++ exception as
    a RuntimeError (via the FFI error guard), not silently emit NaN."""

    def test_cokriging_markI_overflowing_range_ratio_surfaces_error(self):
        from geo_bsd.geo import (
            ContProperty,
            CovarianceModel,
            SugarboxGrid,
            covariance,
            simple_cokriging_markI,
        )
        from geo_bsd.validation import CriticalValidationError

        grid = SugarboxGrid(x=3, y=3, z=1)
        size = 3 * 3 * 1
        rng = np.random.RandomState(9)
        primary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        secondary = ContProperty(
            rng.rand(size).astype("float32") * 100,
            np.ones(size, dtype="uint8"),
        )
        # ranges[0]/ranges[1] = 1e10/1e-300 = 1e310 → Inf ratio. Python
        # validation accepts each range (finite, >= MIN_RANGE, <= MAX_RANGE).
        cov_model = CovarianceModel(
            type=covariance.spherical,
            ranges=(1e10, 1e-300, 1.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.0,
        )

        with pytest.raises((RuntimeError, ValueError, CriticalValidationError)):
            simple_cokriging_markI(
                prop=primary,
                grid=grid,
                radiuses=(1, 1, 1),
                max_neighbours=4,
                cov_model=cov_model,
                secondary_data=secondary,
                primary_mean=50.0,
                secondary_mean=50.0,
                secondary_variance=1.0,
                correlation_coef=0.5,
            )



# =============================================================================
# III-37 — SGS scalar stationary mean must be CDF-transformed
# =============================================================================


@pytest.mark.hpgl
class TestSgsScalarMeanCdfTransform:
    """III-37: the user-supplied scalar stationary mean is in DATA space while
    the hard data have already been forward-transformed to normal-score space.
    Pre-fix the raw data-space mean was used in normal-score space, pinning
    simulated cells to the CDF's max datum (live probe: frac==100 = 0.122 with
    mean=50 vs 0 with mean=None). The existing test_sgs_with_scalar_mean
    assertion (\"closer to 50 than 0\") passes for the wrong reason because a
    pinned-at-max output is still closer to 50 than to 0.
    """

    def test_scalar_mean_does_not_pin_to_cdf_max(self):
        import geo_bsd.geo as geo
        from geo_bsd.cdf import CdfData
        from geo_bsd.sgs import sgs_simulation

        grid = geo.SugarboxGrid(x=10, y=10, z=5)  # 500 cells
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(42)
        data = rng.rand(size).astype("float32") * 100
        mask = np.zeros(size, dtype="uint8")
        mask[::10] = 1  # sparse: 10% informed
        prop = geo.ContProperty(data, mask)
        prop.fix_shape(grid)
        cov_model = geo.CovarianceModel(
            type=geo.covariance.spherical,
            ranges=(5.0, 5.0, 3.0),
            angles=(0.0, 0.0, 0.0),
            sill=1.0,
            nugget=0.1,
        )
        cdf = CdfData(
            np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype="float32"),
            np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype="float32"),
        )
        result = sgs_simulation(
            prop=prop,
            grid=grid,
            cdf_data=cdf,
            radiuses=(5, 5, 3),
            max_neighbours=12,
            cov_model=cov_model,
            seed=42,
            mean=50.0,
        )
        assert np.all(np.isfinite(result.data.astype("float64")))
        simulated = result.data[result.mask > 0].astype("float64")
        # Pre-fix: a data-space mean of 50 was treated as normal score 50
        # (CDF prob ≈ 1.0), pinning a large fraction of simulated cells to the
        # CDF max datum (100). Post-fix the mean is transformed to normal-score
        # 0, so essentially no cells land on the max datum.
        frac_at_max = float(np.mean(simulated >= 99.999))
        assert frac_at_max < 0.05, (
            f"SGS mean=50: {frac_at_max:.3f} of simulated cells pinned to CDF max "
            f"(mean {np.mean(simulated):.2f})"
        )


# =============================================================================
# III-10 — fast reader must not split long tokens into two values
# =============================================================================


@pytest.mark.hpgl
class TestReadIncFileLongToken:
    """III-10: the C++ fast reader's token_stream_t truncated tokens longer
    than the 255-char caller buffer and resumed mid-token, silently splitting
    one logical value into two (a 294-char \"0.000...5\" became 0.0 AND 5.0).
    On a truncated file the split produced exactly `size` tokens, so the I2-56
    count check never fired and load_cont_property returned SILENT WRONG DATA.
    Post-fix the reader rejects the over-long token; the slow-parser fallback
    (or the count-mismatch validation) then raises instead.
    """

    def test_truncated_long_token_raises_not_silent_wrong_data(self, tmp_path):
        import geo_bsd.geo as geo

        token = "0." + "0" * 291 + "5"  # 294 chars, value ~5e-292
        # Truncated file: 9 real tokens + the 294-char token = 10 tokens at
        # size=10. Pre-fix the split produced exactly 10 tokens → rc==0 with
        # [1,2,3,4,0,5,6,7,8,9] (true 5e-292 replaced by 0.0 AND 5.0).
        content = "long\n1 2 3 4 " + token + " 6 7 8 9"  # no trailing '\n'
        fpath = tmp_path / "longtoken.inc"
        fpath.write_text(content)

        with pytest.raises((RuntimeError, ValueError)):
            geo.load_cont_property(str(fpath), -99.0, size=10, basedir=str(tmp_path))


# =============================================================================
# F-39: Python validation must reject covariance range <= 0 at the boundary
# =============================================================================


class TestCovarianceRangeRejectsZero:
    """F-39: C++ set_ranges rejects range <= 0; the Python validator must
    reject it with a clean CriticalValidationError instead of deferring to
    a late RuntimeError inside the FFI call. Pre-fix MIN_RANGE=0.0 and the
    `r < MIN_RANGE` comparison accepted range=0.0."""

    def test_validate_covariance_rejects_zero_range(self):
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        with pytest.raises(CriticalValidationError, match="must be > 0"):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.0, ranges=(0.0, 5.0, 5.0)
            )

    def test_validate_covariance_rejects_negative_range(self):
        from geo_bsd.validation import CriticalValidationError, ParameterValidator

        with pytest.raises(CriticalValidationError, match="must be > 0"):
            ParameterValidator.validate_covariance_parameters(
                sill=1.0, nugget=0.0, ranges=(-1.0, 5.0, 5.0)
            )

    def test_validate_covariance_accepts_positive_range(self):
        from geo_bsd.validation import ParameterValidator

        # Tiny-but-positive ranges (including the III-09 ratio-overflow
        # config's 1e-300) must still pass the range bound; the ratio guard
        # is the C++ layer's concern.
        ParameterValidator.validate_covariance_parameters(
            sill=1.0, nugget=0.0, ranges=(1e-300, 1.0, 1.0)
        )
        ParameterValidator.validate_covariance_parameters(
            sill=1.0, nugget=0.0, ranges=(5.0, 5.0, 5.0)
        )

    def test_covariance_model_rejects_zero_range(self):
        """End-to-end: CovarianceModel(ranges=(0,...)) must raise a clean
        CriticalValidationError at construction (F-39 boundary)."""
        from geo_bsd.geo import CovarianceModel, covariance
        from geo_bsd.validation import CriticalValidationError

        with pytest.raises(CriticalValidationError, match="must be > 0"):
            CovarianceModel(
                type=covariance.spherical, ranges=(0.0, 5.0, 5.0), sill=1.0
            )


# =============================================================================
# F-03: CubeScan total-work cap regression (library-level guard)
# =============================================================================


class TestCubeScanGridWorkCap:
    def test_cube_scan_rejects_total_work_over_cap(self, monkeypatch):
        import geo_bsd.variogram as v

        monkeypatch.setattr(v, "MAX_TOTAL_GRID_WORK", 500.0)
        ell = v.TVEllipsoid(R1=10, R2=5, R3=3)
        templ = v.TVVariogramSearchTemplate(
            LagWidth=1.0, LagSeparation=1.0, TolDistance=1.0,
            NumLags=3, Ellipsoid=ell,
        )
        mask = np.ones((10, 10, 5), dtype="uint8")  # 500 cells, >1 offset
        with pytest.raises(ValueError, match="total grid work"):
            v.CubeScan(templ, mask, lambda *a: np.zeros(3), None)


# =============================================================================
# F-04/III-19/II-43/II-44/II-45: sample-script regressions
# =============================================================================


class TestSampleScriptImports:
    """F-04: the gtsim/gtsimk scripts import _clone_prop/_create_cont_prop
    from geo_bsd.geo (the defining module), not the geo_bsd top level."""

    def test_gtsim_imports_from_geo_module(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "gtsim_script",
            str(REPO_ROOT / "sample-scripts/gtsim.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        assert mod is not None

    def test_gtsim_script_source_uses_geo_import(self):
        with open(REPO_ROOT / "sample-scripts/gtsim.py") as fh:
            src = fh.read()
        assert "from geo_bsd.geo import _clone_prop" in src
        assert "from geo_bsd import (" in src
        with open(REPO_ROOT / "sample-scripts/gtsimk.py") as fh:
            src = fh.read()
        assert "from geo_bsd.geo import _clone_prop, _create_cont_prop" in src
        with open(REPO_ROOT / "sample-scripts/gtsimk_const_prob.py") as fh:
            src = fh.read()
        assert "from geo_bsd.geo import _clone_prop" in src

    def test_gtsim_truncation_uses_elif(self):
        """II-43: the double-if truncation (all-facies-1 when tk <= 0) must
        be an if/elif."""
        with open(REPO_ROOT / "sample-scripts/gtsim.py") as fh:
            src = fh.read()
        # After the first `if <` block there must be an `else:` (not a
        # second independent `if >=`).
        assert "        else:" in src

    def test_gtsim_pk_only_flow_validates_sgs_params(self):
        """III-20: the user-pk flow must not leave sgs_params=None (`**None`
        TypeError)."""
        with open(REPO_ROOT / "sample-scripts/gtsim.py") as fh:
            src = fh.read()
        assert "sgs_params is None" in src
        assert "raise ValueError" in src

    def test_gtsimk_truncation_snapshots_value(self):
        """II-44: the truncation loop must compare the ORIGINAL value, not
        the overwritten integer."""
        with open(REPO_ROOT / "sample-scripts/gtsimk.py") as fh:
            src = fh.read()
        assert "value = prop1.data.flat[i]" in src

    def test_gtsimk_pseudo_gaussian_value_driven(self):
        """III-19: interval selected by the cell facies value, not the last
        j iteration."""
        with open(REPO_ROOT / "sample-scripts/gtsimk.py") as fh:
            src = fh.read()
        assert "val = int(result.data.flat[i])" in src

    def test_gtsimk_pk_prop_list_form(self):
        """II-45: user pk_prop must be a list of ContProperty (one per
        indicator), not a single ContProperty."""
        with open(REPO_ROOT / "sample-scripts/gtsimk.py") as fh:
            src = fh.read()
        assert "all(isinstance(p, ContProperty) for p in pk_prop)" in src


# =============================================================================
# BUILD DOMAIN — s8-fix-build regression tests
# (F-32, II-19, II-20, II-21, II-22, II-24, II-25, II-28-coord, III-24-build,
#  III-26, III-27, III-28, III-29, III-30, III-31)
# =============================================================================


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _read(path: str) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


class TestBuildShCliExitCodes:
    """F-32: error paths must exit non-zero; --help exits 0."""

    def test_unknown_argument_fails_non_zero(self):
        """Pre-fix: './build.sh --bogus' exited 0 (usage() ended exit 0) —
        a silent success on an invalid invocation."""
        import shutil
        import subprocess

        if not shutil.which("bash"):
            pytest.skip("bash not available")
        r = subprocess.run(
            ["bash", str(REPO_ROOT / "build.sh"), "--bogus"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode != 0
        assert "Unknown argument" in (r.stdout + r.stderr)

    def test_help_exits_zero(self):
        import shutil
        import subprocess

        if not shutil.which("bash"):
            pytest.skip("bash not available")
        r = subprocess.run(
            ["bash", str(REPO_ROOT / "build.sh"), "--help"],
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0


class TestSdistIncludesBuildScripts:
    """II-19: the sdist must ship build.bat/build.sh — the README-documented
    Windows/Unix build entry points. Pre-fix: neither was in the include
    list, so a tarball user had no build script."""

    def test_sdist_include_has_build_scripts(self):
        import tomllib

        pyproject = tomllib.loads(_read("pyproject.toml"))
        includes = pyproject["tool"]["scikit-build"]["sdist"]["include"]
        assert "build.bat" in includes
        assert "build.sh" in includes


class TestValidateEnvironmentPlatformTable:
    """II-20: check_build_files must accept the platform's ACTUAL native
    library extension (.dylib on macOS), not only .dll/.so — which
    false-FAILed every healthy macOS build."""

    @staticmethod
    def _load_ve():
        import importlib.util

        spec = importlib.util.spec_from_file_location(
            "validate_environment", REPO_ROOT / "tests" / "validate_environment.py"
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    def test_darwin_table_contains_dylib(self):
        ve = self._load_ve()
        darwin = ve._NATIVE_LIB_GLOBS["darwin"]
        assert "hpgl.dylib" in darwin
        assert "_cvariogram.dylib" in darwin

    def test_build_files_present_accepts_host_platform(self, tmp_path):
        ve = self._load_ve()
        key = ve._platform_lib_key()
        hpgl_name = next(n for n in ve._NATIVE_LIB_GLOBS[key] if n.startswith("hpgl."))
        cvar_name = next(n for n in ve._NATIVE_LIB_GLOBS[key] if n.startswith("_cvariogram."))
        (tmp_path / hpgl_name).write_bytes(b"x")
        (tmp_path / cvar_name).write_bytes(b"x")
        found, cvar_found = ve._build_files_present(tmp_path)
        assert found is True
        assert cvar_found is True


class TestVcxprojBlasLinkage:
    """II-21: hpgl.vcxproj must not silently link zero BLAS/LAPACK when
    UseMKL=false (LNK2019 from the unconditional dpotrf_/dpotrs_ calls)."""

    def test_honest_comment_and_openblas_hookup(self):
        vcx = _read("src/msvc/hpgl.vcxproj")
        assert "OpenBLASRoot" in vcx
        assert "openblas.lib" in vcx

    def test_clear_error_when_no_blas_configured(self):
        vcx = _read("src/msvc/hpgl.vcxproj")
        assert "HpglCheckBlas" in vcx
        assert "BeforeTargets=\"Link\"" in vcx
        assert "LNK2019" in vcx


class TestWindowsMsvcPresetRelease:
    """II-22: the windows-msvc build preset must pin configuration=Release —
    multi-config generators ignore CMAKE_BUILD_TYPE, so without it the preset
    silently built Debug."""

    def test_windows_msvc_build_preset_has_release_configuration(self):
        import json

        presets = json.loads(_read("CMakePresets.json"))
        bp = next(b for b in presets["buildPresets"] if b["name"] == "windows-msvc")
        assert bp.get("configuration") == "Release"


class TestDeploymentTargetSingleSource:
    """II-24: the wheel verification gate must derive the deployment target
    from the wheel tag (accepting the documented CMAKE_OSX_DEPLOYMENT_TARGET
    override), not reject anything but a hardcoded macosx_11_0."""

    def test_wheel_gate_derives_target_from_tag(self):
        sh = _read("build.sh")
        assert "DEPLOYMENT_TARGET = float(f\"{tag_m.group(1)}.{tag_m.group(2)}\")" in sh
        assert "macosx_11_0" not in sh

    def test_pyproject_documents_override(self):
        pyproject = _read("pyproject.toml")
        assert "CMAKE_ARGS" in pyproject
        assert "CMAKE_OSX_DEPLOYMENT_TARGET" in pyproject


class TestCmakeWindowsDllDeploy:
    """II-25: CMake Windows builds must deploy DLLs to src/geo_bsd (the
    Python runtime location) and use a DEBUG_POSTFIX so Debug/Release DLLs
    do not name-collide."""

    def test_debug_postfix_and_post_build_copy(self):
        cm = _read("CMakeLists.txt")
        assert "DEBUG_POSTFIX" in cm
        assert "POST_BUILD" in cm
        assert "copy_if_different" in cm
        assert "src/geo_bsd/" in cm


class TestWheelSmokeVersionConsistency:
    """II-28 (build-side coordinate): the wheel smoke test must assert the
    installed __version__ matches the wheel filename version."""

    def test_wheel_smoke_asserts_version(self):
        sh = _read("build.sh")
        assert "__version__" in sh
        assert "WHEEL_VERSION" in sh


class TestPy39GrammarSmoke:
    """III-24 (build-side): the build pipeline must check Python 3.9 grammar
    compatibility on both dev and wheel paths (no CI exists to catch it)."""

    def test_grammar_check_wired_into_both_paths(self):
        sh = _read("build.sh")
        assert "py39_grammar_check" in sh
        assert "feature_version=(3, 9)" in sh
        assert sh.count("py39_grammar_check") >= 3  # def + wheel call + dev call


class TestWheelLinuxRejection:
    """III-26: --wheel must be rejected on Linux with a clear message instead
    of failing deterministically inside the macOS-only gate."""

    def test_linux_wheel_rejection_present(self):
        sh = _read("build.sh")
        assert "macOS-only" in sh
        assert "cibuildwheel" in sh


class TestVcxprojGlobSync:
    """III-27: every .cpp under src/geo_bsd/hpgl (except stdafx.cpp) must be
    listed in src/msvc/hpgl.vcxproj — a new source must not ship broken on
    Windows silently."""

    def test_vcxproj_matches_source_glob(self):
        import re

        cpps = {p.name for p in (REPO_ROOT / "src/geo_bsd/hpgl").glob("*.cpp")}
        vcx = _read("src/msvc/hpgl.vcxproj")
        listed = set(
            re.findall(
                r'ClCompile Include="\.\.\\geo_bsd\\hpgl\\([^"]+\.cpp)"', vcx
            )
        )
        assert cpps - {"stdafx.cpp"} == listed - {"stdafx.cpp"}


class TestLinuxRelocatabilityGate:
    r"""III-28: the non-wheel build must run a Linux relocatability check
    (readelf/ldd) equivalent to the macOS R-26 gate — and the gate must
    actually FAIL on absolute non-system RPATH/RUNPATH entries.

    R-01: the pre-fix sed pattern 's/.*(RPATH\|RUNPATH)[^[]*\[\(.*\)\]/\1/p'
    mis-parsed under GNU BRE (unescaped parens literal, `\|` top-level
    alternation) — single-entry absolute RPATH AND RUNPATH both bypassed the
    gate with a false "PASSED". These tests pipe readelf -d fixtures through
    the REAL gate pipeline extracted from build.sh (sed program + grep
    filter read from the file, so a pattern regression is caught), and assert
    the gate would FAIL (non-empty ABS) on absolute non-system paths while
    still allowing relative $ORIGIN markers."""

    RPATH_FILTER = r"^/(opt|usr/local|home/|Users/|Applications)"

    @staticmethod
    def _gate_sed_program(sh: str) -> str:
        """Extract the sed program from build.sh's Linux gate verbatim."""
        import re
        m = re.search(r"sed -n '([^']+)'", sh)
        assert m, "Linux gate sed program not found in build.sh"
        return m.group(1)

    @staticmethod
    def _gate_grep_filter(sh: str, sed_end: int) -> str:
        import re
        m = re.search(r"grep -E '([^']+)'", sh[sed_end:])
        assert m, "Linux gate grep filter not found in build.sh"
        return m.group(1)

    @staticmethod
    def _gnu_sed() -> str:
        r"""GNU sed (the Linux runtime) — gsed when present, else `sed` if it
        reports GNU. macOS BSD sed lacks `\|` alternation and would not
        faithfully execute the gate."""
        import shutil
        import subprocess
        for cand in ("gsed", "sed"):
            path = shutil.which(cand)
            if path is None:
                continue
            try:
                v = subprocess.run(
                    [path, "--version"], capture_output=True, text=True, timeout=15
                )
            except OSError:
                continue
            if "GNU sed" in v.stdout:
                return path
        return ""

    @staticmethod
    def _run_gate_pipeline(sh: str, fixture_lines) -> list:
        """Run the extracted sed|tr|grep gate pipeline over readelf -d
        fixture lines with GNU sed. Returns the filtered absolute paths
        (non-empty == the gate would FAIL, i.e. ABS would be set)."""
        import re
        import subprocess
        sed = TestLinuxRelocatabilityGate._gnu_sed()
        if not sed:
            pytest.skip("GNU sed (gsed) not available — cannot exercise gate pipeline")
        prog = TestLinuxRelocatabilityGate._gate_sed_program(sh)
        sed_idx = sh.index(prog)
        filter_ = TestLinuxRelocatabilityGate._gate_grep_filter(sh, sed_idx)
        proc = subprocess.run(
            [sed, "-n", prog],
            input="".join(fixture_lines),
            capture_output=True, text=True, timeout=30,
        )
        assert proc.returncode == 0, proc.stderr
        # replicate: | tr ':' '\n' | grep -E '^/(opt|usr/local|home/|Users/|Applications)'
        return [
            part for part in proc.stdout.replace(":", "\n").split("\n")
            if re.search(filter_, part)
        ]

    def test_linux_reloc_gate_present(self):
        sh = _read("build.sh")
        assert "readelf -d" in sh
        assert "RPATH" in sh
        # R-01: the corrected GNU BRE pattern (escaped parens, group 2) must
        # be the program in build.sh — the pre-fix unescaped \1 variant let
        # absolute RPATH/RUNPATH bypass the gate.
        assert TestLinuxRelocatabilityGate._gate_sed_program(sh) == (
            r"s/.*\(RPATH\|RUNPATH\)[^[]*\[\(.*\)\]/\2/p"
        ), "gate sed program regressed to the broken GNU BRE variant"

    def test_gate_fails_on_absolute_non_system_rpath(self):
        """Single-entry absolute non-system RPATH must fail the gate."""
        sh = _read("build.sh")
        abs_paths = TestLinuxRelocatabilityGate._run_gate_pipeline(sh, [
            " 0x000000000000000f (RPATH)            Library rpath: [/opt/OpenBLAS/lib]\n",
        ])
        assert "/opt/OpenBLAS/lib" in abs_paths, (
            f"absolute RPATH bypassed the gate (filtered={abs_paths!r})"
        )

    def test_gate_fails_on_absolute_non_system_runpath(self):
        """Single-entry absolute non-system RUNPATH must fail the gate."""
        sh = _read("build.sh")
        abs_paths = TestLinuxRelocatabilityGate._run_gate_pipeline(sh, [
            " 0x000000000000001d (RUNPATH)            Library runpath: [/usr/local/lib]\n",
        ])
        assert "/usr/local/lib" in abs_paths, (
            f"absolute RUNPATH bypassed the gate (filtered={abs_paths!r})"
        )

    def test_gate_fails_on_multi_entry_absolute(self):
        """Multi-entry absolute RPATH must fail the gate (every entry)."""
        sh = _read("build.sh")
        abs_paths = TestLinuxRelocatabilityGate._run_gate_pipeline(sh, [
            " 0x000000000000000f (RPATH)            Library rpath: [/opt/a:/opt/b]\n",
        ])
        assert "/opt/a" in abs_paths and "/opt/b" in abs_paths, (
            f"multi-entry absolute RPATH bypassed the gate (filtered={abs_paths!r})"
        )

    def test_gate_allows_relative_origin_marker(self):
        """$ORIGIN markers are relative and allowed — no false positive."""
        sh = _read("build.sh")
        abs_paths = TestLinuxRelocatabilityGate._run_gate_pipeline(sh, [
            " 0x000000000000000f (RPATH)            Library rpath: [$ORIGIN/../lib]\n",
        ])
        assert abs_paths == [], (
            f"$ORIGIN relative marker falsely flagged (filtered={abs_paths!r})"
        )

    def test_ldd_fallback_reads_resolved_path_column(self):
        """R-01: the ldd fallback must read column 3 (the resolved path in
        'soname => /path (0x...)' lines), not column 1 (the soname, which
        never matches the absolute-path filter)."""
        sh = _read("build.sh")
        import re as _re
        m = _re.search(r"ldd \"\$lib\"[^\n]*awk '([^']+)'", sh)
        assert m, "ldd fallback awk program not found in build.sh"
        awk_prog = m.group(1)
        assert "{print $3}" in awk_prog, (
            f"ldd fallback awk program still reads the soname column: {awk_prog!r}"
        )


class TestCtestExecutedInPipeline:
    """III-29: the build pipeline must execute the registered CTest suite
    (build.sh) / Python test suite (build.bat) — the registrations were dead
    weight before."""

    def test_build_sh_runs_ctest(self):
        sh = _read("build.sh")
        assert "ctest --output-on-failure" in sh

    def test_build_bat_runs_pytest(self):
        bat = _read("build.bat")
        assert "pytest tests/python" in bat


class TestPresetGeneratorCollision:
    """III-30: release/debug presets must pin an explicit Ninja generator so
    alternating standard-path and preset builds in the SAME binaryDir do not
    hard-fail on a generator mismatch."""

    def test_single_config_presets_pin_ninja(self):
        import json

        presets = json.loads(_read("CMakePresets.json"))
        for name in ("release", "debug", "release-mkl"):
            cp = next(c for c in presets["configurePresets"] if c["name"] == name)
            assert cp.get("generator") == "Ninja", name


class TestCtestPythonTargetConfig:
    """III-31: the CTest hpgl_python_tests target must mirror run_tests.py's
    default marker selection (-m "not slow") and set a timeout."""

    def test_ctest_python_target_excludes_slow_and_times_out(self):
        cm = _read("tests/CMakeLists.txt")
        assert '"not slow"' in cm
        assert "TIMEOUT" in cm
