"""
Regression tests for geo.py group 2 FIX findings (F-28, F-33, F-44, F-54, F-55).

Each test fails against the pre-fix code and passes against the fixed code.

Covers:
- F-28: geo.py I/O functions use a trusted base (DEFAULT_BASE_DIR or explicit
        basedir) instead of the self-referential dirname-of-file, so symlink
        containment actually rejects escapes
- F-33: simple_kriging / lvm_kriging surface partial/total solver failure
        from _last_kriging_stats instead of returning mean-filled output
        indistinguishably from success
- F-44 + I2-55: write paths validate indicator_values ([0,255] integral, no
        duplicates) and undefined_value ([0,255] for byte writers) before the
        FFI call
- F-54: slow parsers match the C++ fast reader's token/terminator semantics
        (mid-line '--' comment skips the rest of the line)
- F-55: load_cont_property / load_ind_property return the same shape from
        both the fast C++ path and the slow-parser fallback (3D for 3-tuples)
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        IndProperty,
        SugarboxGrid,
        _load_prop_cont_slow,
        _load_prop_ind_slow,
        covariance,
        load_cont_property,
        load_ind_property,
        lvm_kriging,
        simple_kriging,
        write_gslib_property,
        write_property,
    )
    from geo_bsd.validation import CriticalValidationError

    FIXES_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError):
    FIXES_AVAILABLE = False


def _write_text(path, content):
    path.write_text(content)


# =============================================================================
# F-28 — trusted basedir replaces self-referential basedir
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestTrustedBasedirGeo:
    """F-28: geo I/O functions must use a trusted base, not the file's own dir."""

    def test_write_outside_default_basedir_rejected(self, tmp_path):
        """Writing to a path outside DEFAULT_BASE_DIR must raise when no
        explicit basedir is given.

        Pre-fix: the self-referential basedir (dirname of the file) always
        passes containment, so the write succeeds. Post-fix: the default is
        DEFAULT_BASE_DIR (process cwd); tmp_path lies outside it, so the
        write is rejected.
        """
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)
        fpath = str(tmp_path / "escape.inc")
        with pytest.raises(CriticalValidationError, match="outside allowed base"):
            write_property(prop, fpath, "cap", -99.0)

    def test_explicit_basedir_allows_write(self, tmp_path):
        """Passing an explicit trusted base permits writes inside it."""
        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)
        fpath = str(tmp_path / "ok.inc")
        write_property(prop, fpath, "cap", -99.0, basedir=str(tmp_path))
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

        data = np.array([1.0, 2.0, 3.0], dtype="float32")
        mask = np.ones(3, dtype="uint8")
        prop = ContProperty(data, mask)
        fpath = str(link / "out.inc")
        with pytest.raises(CriticalValidationError, match="outside allowed base"):
            write_property(prop, fpath, "cap", -99.0, basedir=str(base))

    def test_read_outside_default_basedir_rejected(self, tmp_path):
        """Reading a file outside DEFAULT_BASE_DIR without explicit basedir
        must be rejected (pre-fix: self-referential basedir always passes)."""
        fpath = tmp_path / "data.inc"
        _write_text(fpath, "prop\n1.0 2.0 3.0\n/\n")
        with pytest.raises(CriticalValidationError, match="outside allowed base"):
            load_cont_property(str(fpath), -99.0)

    def test_read_with_explicit_basedir_succeeds(self, tmp_path):
        """Reading a file with an explicit basedir works."""
        fpath = tmp_path / "data.inc"
        _write_text(fpath, "prop\n1.0 2.0 3.0\n/\n")
        prop = load_cont_property(str(fpath), -99.0, basedir=str(tmp_path))
        assert prop.data.size == 3


# =============================================================================
# F-33 — SK/LVM partial kriging failure surfaced
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestKrigingFailureSurfacing:
    """F-33: simple_kriging/lvm_kriging surface solver failure via stats."""

    def test_simple_kriging_singularity_raises(self, monkeypatch, caplog):
        """A singular kriging system must raise RuntimeError.

        Pre-fix: the C++ stats reported points_singularity but geo.py never
        consumed them — the call returned mean-filled output silently.
        Post-fix: points_singularity > 0 raises RuntimeError.
        """
        from geo_bsd import geo

        # Fabricate a singular-system stats response. Use a real small call
        # with stats monkeypatched to report singularity so we exercise the
        # public wrapper's consumption path.
        real_stats = {
            "points_calculated": 0,
            "points_without_neighbours": 0,
            "points_singularity": 5,
            "mean": 1.0,
            "speed_nps": 0.0,
        }

        def fake_stats():
            return real_stats

        monkeypatch.setattr(geo, "get_kriging_stats", fake_stats)

        grid = SugarboxGrid(x=2, y=2, z=1)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        mask = np.array([1, 1, 0, 0], dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(RuntimeError, match="singular"):
            simple_kriging(prop, grid, (1, 1, 1), 4, cov_model)

    def test_simple_kriging_no_neighbours_warns(self, caplog):
        """Cells with no neighbours are mean-filled (documented contract) but
        the failure is surfaced as a warning, not silent.

        Pre-fix: no warning was emitted for points_without_neighbours > 0.
        Post-fix: a warning is logged with the failure stats.
        """
        import logging

        grid = SugarboxGrid(x=5, y=5, z=1)
        data = np.random.RandomState(0).rand(25).astype("float32") * 100
        mask = np.zeros(25, dtype="uint8")
        mask[0] = 1  # single informed cell → most cells have no neighbours
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(2.0, 2.0, 1.0), sill=1.0, nugget=0.1
        )

        with caplog.at_level(logging.WARNING, logger="geo_bsd.geo"):
            result = simple_kriging(prop, grid, (1, 1, 1), 4, cov_model)

        assert isinstance(result, ContProperty)
        assert any(
            "could not be kriged" in rec.message for rec in caplog.records
        ), f"expected no-neighbour warning, got: {[r.message for r in caplog.records]}"

    def test_lvm_kriging_singularity_raises(self, monkeypatch):
        """lvm_kriging must also consume stats and raise on singularity."""
        from geo_bsd import geo

        real_stats = {
            "points_calculated": 0,
            "points_without_neighbours": 0,
            "points_singularity": 3,
            "mean": 1.0,
            "speed_nps": 0.0,
        }

        def fake_stats():
            return real_stats

        monkeypatch.setattr(geo, "get_kriging_stats", fake_stats)

        grid = SugarboxGrid(x=2, y=2, z=1)
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype="float32")
        mask = np.array([1, 1, 0, 0], dtype="uint8")
        prop = ContProperty(data, mask)
        mean_data = np.ones(4, dtype="float32") * 2.0
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(1.0, 1.0, 1.0), sill=1.0, nugget=0.1
        )

        with pytest.raises(RuntimeError, match="singular"):
            lvm_kriging(prop, grid, mean_data, (1, 1, 1), 4, cov_model)


# =============================================================================
# F-44 + I2-55 — write-path [0,255] validation
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestIndicatorWriteValidation:
    """F-44 + I2-55: indicator_values and undefined_value validated on write."""

    def _ind_prop(self, n=3):
        data = np.zeros(n, dtype="uint8")
        mask = np.ones(n, dtype="uint8")
        return IndProperty(data, mask, 2)

    def test_indicator_value_out_of_range_inc_raises(self, tmp_path):
        """write_property with indicator_values=[300] must raise ValueError.

        Pre-fix: numpy 2.x raised a confusing OverflowError (or ctypes
        silently wrapped 300->44 on the GSLIB path). Post-fix: clear
        ValueError before the FFI call.
        """
        prop = self._ind_prop()
        fpath = str(tmp_path / "bad.inc")
        with pytest.raises(ValueError, match="[0, 255]"):
            write_property(prop, fpath, "test", 255, [300], basedir=str(tmp_path))
        assert not os.path.exists(fpath), "no file should be created on rejection"

    def test_indicator_value_out_of_range_gslib_raises(self, tmp_path):
        """write_gslib_property with indicator_values=[300] must raise ValueError."""
        prop = self._ind_prop()
        fpath = str(tmp_path / "bad.dat")
        with pytest.raises(ValueError, match="[0, 255]"):
            write_gslib_property(prop, fpath, "test", 255, [300], basedir=str(tmp_path))
        assert not os.path.exists(fpath)

    def test_fractional_indicator_value_raises(self, tmp_path):
        """Fractional indicator values (1.5) must raise, not silently truncate."""
        prop = self._ind_prop()
        fpath = str(tmp_path / "frac.inc")
        with pytest.raises(ValueError, match="[0, 255]"):
            write_property(prop, fpath, "test", 255, [1.5], basedir=str(tmp_path))

    def test_duplicate_indicator_values_raise(self, tmp_path):
        """Duplicate indicator values must raise (port of slow-parser seen-set).

        Pre-fix: the write path accepted duplicates while the slow parser
        rejected them — asymmetric contract. Post-fix: duplicates raise.
        """
        prop = self._ind_prop()
        fpath = str(tmp_path / "dup.inc")
        with pytest.raises(ValueError, match="duplicate"):
            write_property(prop, fpath, "test", 255, [10, 10], basedir=str(tmp_path))

    def test_undefined_value_out_of_range_byte_writer_raises(self, tmp_path):
        """I2-55: undefined_value outside [0,255] must raise for byte writers."""
        prop = self._ind_prop()
        fpath = str(tmp_path / "uv.inc")
        with pytest.raises(ValueError, match="undefined_value.*[0, 255]"):
            write_property(prop, fpath, "test", 300, [10], basedir=str(tmp_path))
        with pytest.raises(ValueError, match="undefined_value.*[0, 255]"):
            write_gslib_property(prop, str(tmp_path / "uv.dat"), "test", 300, [10],
                                 basedir=str(tmp_path))

    def test_valid_indicator_values_still_write(self, tmp_path):
        """In-range unique indicator values still write successfully."""
        prop = self._ind_prop()
        fpath = str(tmp_path / "ok.inc")
        write_property(prop, fpath, "test", 255, [10, 20], basedir=str(tmp_path))
        assert os.path.exists(fpath)


# =============================================================================
# F-54 — slow parser token/terminator semantics match the C++ fast reader
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestSlowParserTokenSemantics:
    """F-54: slow parsers match the C++ fast reader on token semantics."""

    def test_mid_line_comment_skips_rest_of_line(self, tmp_path):
        """A mid-line '--' token is a comment: the rest of the line is skipped.

        Pre-fix: the slow parser skipped only the '--' token and continued
        parsing the remainder of the line (3.0 4.0 were consumed), while the
        C++ fast reader skipped the rest of the line — divergent streams.
        Post-fix: both consume [1.0, 2.0] only.
        """
        fpath = tmp_path / "comment.inc"
        _write_text(fpath, "prop\n1.0 2.0 -- 3.0 4.0\n/\n")
        prop = _load_prop_cont_slow(str(fpath), -99.0, basedir=str(tmp_path))
        assert len(prop.data) == 2
        assert list(prop.data) == [1.0, 2.0]

    def test_ind_mid_line_comment_skips_rest_of_line(self, tmp_path):
        """The IND slow parser applies the same '--' comment rule."""
        fpath = tmp_path / "comment_ind.inc"
        _write_text(fpath, "prop\n10 20 -- 30 40\n/\n")
        prop = _load_prop_ind_slow(str(fpath), 255, [10, 20, 30, 40],
                                   basedir=str(tmp_path))
        assert len(prop.data) == 2
        assert list(prop.data) == [0, 1]

    def test_line_start_comment_and_slash_still_work(self, tmp_path):
        """Line-start '--' comments and the '/' terminator keep working."""
        fpath = tmp_path / "comments.inc"
        _write_text(
            fpath,
            "-- header comment\nprop\n1.0 2.0\n-- mid comment\n3.0\n/\n",
        )
        prop = _load_prop_cont_slow(str(fpath), -99.0, basedir=str(tmp_path))
        assert list(prop.data) == [1.0, 2.0, 3.0]

    def test_slow_and_fast_agree_on_comment_file(self, tmp_path):
        """A file with a mid-line comment loads identically via both paths.

        Pre-fix: fast C++ threw 'Unexpected end of file' on this file while
        the slow parser loaded it — divergent per-path behavior. Post-fix:
        both load [1.0, 2.0] (the mid-line comment line contributes nothing).
        """
        fpath = tmp_path / "agree.inc"
        _write_text(fpath, "prop\n1.0 2.0 -- 3.0 4.0\n/\n")
        slow = _load_prop_cont_slow(str(fpath), -99.0, basedir=str(tmp_path))
        fast = load_cont_property(str(fpath), -99.0, 2, basedir=str(tmp_path))
        np.testing.assert_array_equal(slow.data, fast.data)
        np.testing.assert_array_equal(slow.mask, fast.mask)


# =============================================================================
# F-55 — consistent return shape across parser paths
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestLoadShapeNormalization:
    """F-55: fast path and fallback return the same shape (3D for 3-tuples)."""

    def test_cont_3tuple_size_returns_3d(self, tmp_path):
        """load_cont_property with a 3-tuple size returns 3D arrays.

        Pre-fix: the fast C++ path returned 1D while the slow-parser fallback
        returned 3D — same public call, different shapes per path. Post-fix:
        both return 3D Fortran-order arrays.
        """
        data = np.arange(8, dtype="float32")
        mask = np.ones(8, dtype="uint8")
        prop = ContProperty(data, mask)
        fpath = str(tmp_path / "cube.inc")
        write_property(prop, fpath, "t", -99.0, basedir=str(tmp_path))

        loaded = load_cont_property(fpath, -99.0, (2, 2, 2), basedir=str(tmp_path))
        assert loaded.data.shape == (2, 2, 2)
        assert loaded.mask.shape == (2, 2, 2)

    def test_ind_3tuple_size_returns_3d(self, tmp_path):
        """load_ind_property with a 3-tuple size returns 3D arrays."""
        data = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="uint8")
        mask = np.ones(8, dtype="uint8")
        prop = IndProperty(data, mask, 2)
        fpath = str(tmp_path / "ind.inc")
        write_property(prop, fpath, "t", 255, [10, 20], basedir=str(tmp_path))

        loaded = load_ind_property(fpath, 255, [10, 20], (2, 2, 2), basedir=str(tmp_path))
        assert loaded.data.shape == (2, 2, 2)
        assert loaded.mask.shape == (2, 2, 2)

    def test_scalar_size_stays_1d(self, tmp_path):
        """A scalar size keeps the 1D contract on both paths."""
        data = np.arange(6, dtype="float32")
        mask = np.ones(6, dtype="uint8")
        prop = ContProperty(data, mask)
        fpath = str(tmp_path / "flat.inc")
        write_property(prop, fpath, "t", -99.0, basedir=str(tmp_path))

        loaded = load_cont_property(fpath, -99.0, 6, basedir=str(tmp_path))
        assert loaded.data.ndim == 1
        assert loaded.data.size == 6

    def test_fallback_and_fast_same_shape(self, tmp_path):
        """Force the fallback path (malformed file the C++ reader rejects)
        and verify it returns the SAME 3D shape as a successful fast path.

        Pre-fix: fast success returned 1D, fallback returned 3D. Post-fix:
        the public load_cont_property normalizes both to 3D for a 3-tuple.
        """
        fpath = tmp_path / "fallback.inc"
        # File with 8 values and a mid-line comment the fast reader rejects.
        _write_text(fpath, "prop\n0 1 2 3 4 5 6 7 8 -- 9\n/\n")

        loaded = load_cont_property(str(fpath), -99.0, (3, 3, 1), basedir=str(tmp_path))
        assert loaded.data.shape == (3, 3, 1)
        assert loaded.mask.shape == (3, 3, 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
