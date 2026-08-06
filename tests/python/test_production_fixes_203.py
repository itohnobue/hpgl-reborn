"""Regression tests for finding B-08 (TEST-ADD T-27): the 38-name top-level
public API (``geo_bsd.__all__``) must not only IMPORT — every name must WORK
through the top-level import path. Pre-fix the suite imported submodules
(``from geo_bsd.geo import X``) almost exclusively, so a top-level re-export
regression (a name dropped from ``__all__`` or a broken top-level import
chain) was undetected for the majority of the 38 names.

The __all__ list is read dynamically at collection time so the parametrized
presence test stays correct if the public surface changes; the behavioral
tests then exercise representative top-level entry points with value
assertions (not presence-only).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    import geo_bsd
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd) not available")
class TestAllNamesPresent:
    """B-08: every name in __all__ must resolve at the package top level."""

    def test_all_names_in_all_resolve(self):
        names = list(geo_bsd.__all__)
        assert len(names) >= 38, (
            f"__all__ shrank to {len(names)} names — the documented public "
            f"surface is 38 (B-08)"
        )
        missing = [n for n in names if not hasattr(geo_bsd, n)]
        assert not missing, (
            f"top-level export missing for __all__ names: {missing}"
        )

    @pytest.mark.parametrize(
        "name",
        [
            "ordinary_kriging", "simple_kriging", "lvm_kriging",
            "indicator_kriging", "median_ik", "simple_cokriging_markI",
            "simple_cokriging_markII", "simple_kriging_weights",
            "sgs_simulation", "sis_simulation", "gtsim_2ind",
            "get_kriging_stats", "calc_cdf",
            "ContProperty", "IndProperty", "CovarianceModel", "SugarboxGrid",
            "covariance", "CdfData", "SGSConfig", "SISConfig", "GTSIMConfig",
            "variogram", "routines", "cvariogram", "validation",
            "load_cont_property", "load_ind_property", "read_inc_file_float",
            "read_inc_file_byte", "write_property", "write_gslib_property",
            "calc_mean", "set_thread_num", "get_thread_num",
            "set_output_handler", "set_progress_handler", "get_gslib_property",
        ],
    )
    def test_name_in_all_and_resolvable(self, name):
        """Every __all__ name must resolve via getattr (B-08). Note: the
        cvariogram module is None when the optional C++ extension is not
        built (documented fallback) — presence, not non-None, is the contract."""
        assert name in geo_bsd.__all__, f"{name} missing from __all__"
        assert hasattr(geo_bsd, name), f"geo_bsd.{name} missing at top level"

    def test_all_matches_star_import_surface(self):
        """`from geo_bsd import *` must expose exactly the __all__ names."""
        ns = {}
        exec("from geo_bsd import *", ns)  # noqa: S102 — deliberately star-import
        star_names = {k for k in ns if not k.startswith("__")}
        assert star_names == set(geo_bsd.__all__), (
            f"star-import surface {sorted(star_names)} != __all__ "
            f"{sorted(geo_bsd.__all__)}"
        )


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd) not available")
class TestTopLevelEntryPointsBehavioral:
    """B-08 behavioral leg: the top-level names WORK — each entry point
    executes a minimal valid call through ``from geo_bsd import X`` and the
    result carries a value-level assertion (not presence-only)."""

    def _grid_prop(self, x=4, y=4, z=2):
        grid = geo_bsd.SugarboxGrid(x=x, y=y, z=z)
        size = x * y * z
        rng = np.random.RandomState(7)
        data = rng.rand(size).astype("float32") * 100
        mask = np.ones(size, dtype="uint8")
        mask[::4] = 0  # ~25% uninformed
        prop = geo_bsd.ContProperty(data, mask)
        prop.fix_shape(grid)
        return grid, prop

    def _cov(self):
        return geo_bsd.CovarianceModel(
            type=geo_bsd.covariance.spherical,
            ranges=(2.0, 2.0, 1.0), sill=1.0, nugget=0.1,
        )

    def test_calc_mean_top_level(self):
        grid, prop = self._grid_prop()
        mean = geo_bsd.calc_mean(prop)
        assert np.isfinite(mean)
        assert 0.0 < mean < 100.0

    def test_set_get_thread_num_top_level(self):
        old = geo_bsd.get_thread_num()
        geo_bsd.set_thread_num(2)
        try:
            assert geo_bsd.get_thread_num() == 2
        finally:
            geo_bsd.set_thread_num(old)

    def test_cdf_data_and_calc_cdf_top_level(self):
        cdf = geo_bsd.CdfData(
            np.array([0.0, 50.0, 100.0], dtype="float32"),
            np.array([0.0, 0.5, 1.0], dtype="float32"),
        )
        assert cdf.values.size == 3
        grid, prop = self._grid_prop()
        out_cdf = geo_bsd.calc_cdf(prop)
        assert np.all(np.isfinite(out_cdf.values))

    def test_cont_property_construct_top_level(self):
        prop = geo_bsd.ContProperty(
            np.ones(4, dtype="float32"), np.ones(4, dtype="uint8")
        )
        assert prop.data.dtype == np.float32
        assert prop.mask.dtype == np.uint8

    def test_kriging_family_top_level(self):
        """ordinary_kriging / simple_kriging / simple_kriging_weights /
        lvm_kriging through the top-level imports produce finite output."""
        grid, prop = self._grid_prop()
        cov = self._cov()

        ok_out = geo_bsd.ordinary_kriging(
            prop=prop, grid=grid, radiuses=(2, 2, 1),
            max_neighbours=8, cov_model=cov,
        )
        assert isinstance(ok_out, geo_bsd.ContProperty)
        assert np.all(np.isfinite(ok_out.data.astype("float64")))

        sk_out = geo_bsd.simple_kriging(
            prop=prop, grid=grid, radiuses=(2, 2, 1),
            max_neighbours=8, cov_model=cov, mean=50.0,
        )
        assert np.all(np.isfinite(sk_out.data.astype("float64")))

        lvm_out = geo_bsd.lvm_kriging(
            prop=prop, grid=grid, radiuses=(2, 2, 1),
            max_neighbours=8, cov_model=cov,
            mean_data=np.full(grid.x * grid.y * grid.z, 50.0, dtype="float32"),
        )
        assert np.all(np.isfinite(lvm_out.data.astype("float64")))

        center = np.array([0.0, 0.0, 0.0], dtype="float32")
        nx = np.array([1.0, 0.0, 0.0], dtype="float32")
        ny = np.array([0.0, 1.0, 0.0], dtype="float32")
        nz = np.array([0.0, 0.0, 1.0], dtype="float32")
        w = geo_bsd.simple_kriging_weights(
            center, nx, ny, nz,
            ranges=(2.0, 2.0, 2.0), sill=1.0,
            cov_type=geo_bsd.covariance.spherical,
            nugget=0.0, angles=(0.0, 0.0, 0.0),
        )
        assert np.all(np.isfinite(np.asarray(w, dtype="float64")))

    def test_simulation_family_top_level(self):
        grid, prop = self._grid_prop()
        cov = self._cov()
        sgs_out = geo_bsd.sgs_simulation(
            prop=prop, grid=grid, cdf_data=None, radiuses=(2, 2, 1),
            max_neighbours=8, cov_model=cov, seed=42,
        )
        assert np.all(np.isfinite(sgs_out.data.astype("float64")))

    def test_sis_top_level(self):
        grid = geo_bsd.SugarboxGrid(x=6, y=6, z=2)
        size = grid.x * grid.y * grid.z
        rng = np.random.RandomState(7)
        data = rng.randint(0, 2, size, dtype="uint8")
        mask = np.ones(size, dtype="uint8")
        mask[::5] = 0
        prop = geo_bsd.IndProperty(data, mask, 2)
        prop.fix_shape(grid)
        cov = self._cov()
        sis_data = [
            {"cov_model": cov, "radiuses": (2, 2, 1), "max_neighbours": 8},
            {"cov_model": cov, "radiuses": (2, 2, 1), "max_neighbours": 8},
        ]
        out = geo_bsd.sis_simulation(
            prop=prop, grid=grid, data=sis_data, seed=42,
            marginal_probs=[0.5, 0.5],
        )
        assert out.indicator_count == 2
        assert np.all(np.isfinite(np.asarray(out.data, dtype="float32")))

    def test_get_kriging_stats_top_level(self):
        stats = geo_bsd.get_kriging_stats()
        assert stats is None or isinstance(stats, dict)

    def test_io_roundtrip_top_level(self, tmp_path):
        grid, prop = self._grid_prop(x=2, y=2, z=1)
        fname = str(tmp_path / "io.inc")
        geo_bsd.write_property(prop, fname, "col", -99.0, basedir=str(tmp_path))
        loaded = geo_bsd.read_inc_file_float(
            fname, -99.0, 4, basedir=str(tmp_path)
        )
        # The mask is the round-trip contract; informed values survive exactly.
        # read_inc_file_float returns a FLAT property (the writer serialized
        # the prop in its flat F-ordered layout), so compare against the
        # prop's flat arrays.
        flat_mask = prop.mask.ravel(order="F")
        flat_data = prop.data.ravel(order="F")
        np.testing.assert_array_equal(loaded.mask, flat_mask)
        informed = flat_mask != 0
        np.testing.assert_allclose(
            loaded.data[informed], flat_data[informed], rtol=1e-6
        )
        # The masked cell reads back as the undefined sentinel.
        np.testing.assert_array_equal(loaded.data[~informed], [-99.0])

    def test_load_cont_property_and_gslib_getter_top_level(self, tmp_path):
        grid, prop = self._grid_prop(x=2, y=2, z=1)
        fname = str(tmp_path / "ld.inc")
        geo_bsd.write_property(prop, fname, "col", -99.0, basedir=str(tmp_path))
        loaded = geo_bsd.load_cont_property(
            fname, -99.0, basedir=str(tmp_path)
        )
        assert np.all(np.isfinite(loaded.data.astype("float64")))
        prop_dict = {"col": loaded.data.ravel(order="F").copy()}
        # get_gslib_property returns a (data, mask) tuple.
        data, mask = geo_bsd.get_gslib_property(prop_dict, "col", -99.0)
        assert data.shape == loaded.data.ravel(order="F").shape
        assert mask.shape == data.shape

    def test_handler_setters_top_level(self):
        # Setting and clearing handlers through the top-level exports must not
        # raise; the conftest autouse fixture clears them on teardown.
        geo_bsd.set_output_handler(None, None)
        geo_bsd.set_progress_handler(None, None)

    def test_submodules_importable_top_level(self):
        assert geo_bsd.variogram is not None
        assert geo_bsd.routines is not None
        assert geo_bsd.validation is not None
        assert geo_bsd.variogram.TVEllipsoid is not None
        assert geo_bsd.routines.MovingAverage3D is not None
        assert geo_bsd.validation.ParameterValidator is not None
