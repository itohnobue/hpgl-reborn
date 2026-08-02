"""Regression tests for s6-py-io FIX stage findings (Python file-I/O security).

Covers:
- F-N17  routines.py GSLIB writers reject crafted keys/captions (\n, --, control
         chars, leading/trailing whitespace) via the shared validate_property_name
- F-N16-py  geo.py write_property / write_gslib_property reject crafted prop_name
         BEFORE the FFI call (shared contract with the C++ writers)
- F-N18  Python GSLIB writers write atomically via temp + os.replace — no
         leftover temp files, target intact on rejection
- F-M18-py  get_gslib_property / LoadGslibFile implement the ±1.0e21 GSLIB
         missing-value trimming window (mask=0 / NaN)
- F-M16  MovingAverage3D validates radiuses and allocates the mask AFTER the
         volume cap; GetCubicalMask rejects 0/negative radiuses
- F-N21  PathValidator.DEFAULT_BASE_DIR resolved lazily (respects os.chdir)

Each test fails against the pre-fix code and passes against the fixed code.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import ContProperty, get_gslib_property, write_gslib_property, write_property
    from geo_bsd.routines import (
        GetCubicalMask,
        LoadGslibFile,
        MovingAverage3D,
        SaveGSLIBCubes,
        SaveGSLIBPointSet,
    )
    from geo_bsd.validation import (
        PathValidator,
        validate_property_name,
    )

    FIXES_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError, RuntimeError, OSError):
    FIXES_AVAILABLE = False


# =============================================================================
# validate_property_name contract (shared with C++ property_writer.cpp:34-52)
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestValidatePropertyName:
    """F-N16/F-N17: shared property-name validation contract."""

    @pytest.mark.parametrize(
        "bad",
        [
            "",            # empty
            "A\nB",        # newline injects phantom header line
            "A\rB",        # CR
            "A\tB",        # TAB is a control char
            "\x01",        # C0 control char
            "\x7f",        # DEL
            "--x",         # comment marker skipped by readers
            "/x",          # INC end-of-data marker
            " x",          # leading whitespace
            "x ",          # trailing whitespace
        ],
    )
    def test_rejects_invalid_names(self, bad):
        with pytest.raises(ValueError):
            validate_property_name(bad)

    @pytest.mark.parametrize(
        "good",
        [
            "prop",
            "my prop",            # internal spaces allowed
            "prop_1",
            "проба",              # non-ASCII UTF-8 allowed
            "x" * 200,            # long names allowed
        ],
    )
    def test_accepts_valid_names(self, good):
        assert validate_property_name(good) == good


# =============================================================================
# F-N17 — routines.py GSLIB writers reject crafted keys/captions
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestGslibWritersRejectCraftedNames:
    """F-N17: dict keys and Caption validated before GSLIB header writes."""

    @pytest.mark.parametrize("bad_key", ["A\nB", "--prop", " prop", "prop ", ""])
    def test_pointset_crafted_key_rejected(self, tmp_path, bad_key):
        fpath = str(tmp_path / "ps.inc")
        ps = {bad_key: np.array([1.0, 2.0], dtype="float32")}
        with pytest.raises(ValueError, match="property name"):
            SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        assert not os.path.exists(fpath), "no file should be created on rejection"

    @pytest.mark.parametrize("bad_key", ["A\nB", "--prop", " prop", ""])
    def test_cubes_crafted_key_rejected(self, tmp_path, bad_key):
        fpath = str(tmp_path / "cubes.inc")
        cubes = {bad_key: np.ones((2, 2, 2), dtype="float32")}
        with pytest.raises(ValueError, match="property name"):
            SaveGSLIBCubes(cubes, fpath, "cap", basedir=str(tmp_path))
        assert not os.path.exists(fpath)

    @pytest.mark.parametrize("bad_caption", ["cap\ninjected\n3", "--cap", " cap", ""])
    def test_pointset_crafted_caption_rejected(self, tmp_path, bad_caption):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.0, 2.0], dtype="float32")}
        with pytest.raises(ValueError, match="property name"):
            SaveGSLIBPointSet(ps, fpath, bad_caption, basedir=str(tmp_path))
        assert not os.path.exists(fpath)

    @pytest.mark.parametrize("bad_caption", ["cap\ninjected\n3", "--cap", " cap"])
    def test_cubes_crafted_caption_rejected(self, tmp_path, bad_caption):
        fpath = str(tmp_path / "cubes.inc")
        cubes = {"P": np.ones((2, 2, 2), dtype="float32")}
        with pytest.raises(ValueError, match="property name"):
            SaveGSLIBCubes(cubes, fpath, bad_caption, basedir=str(tmp_path))
        assert not os.path.exists(fpath)

    def test_valid_names_still_write(self, tmp_path):
        fpath = str(tmp_path / "ok.inc")
        ps = {"P": np.array([1.0, 2.0], dtype="float32")}
        SaveGSLIBPointSet(ps, fpath, "Test Point Set", basedir=str(tmp_path))
        assert os.path.exists(fpath)


# =============================================================================
# F-N18 — atomic temp+rename for Python GSLIB writers
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestGslibWritersAtomicWrite:
    """F-N18: GSLIB writers write to a temp file then os.replace."""

    def test_no_temp_files_left_after_success(self, tmp_path):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.0, 2.0, 3.0], dtype="float32")}
        SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        assert os.path.exists(fpath)
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftovers == [], f"temp files left behind: {leftovers}"

    def test_no_temp_files_left_after_rejection(self, tmp_path):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.0, np.nan], dtype="float32")}
        with pytest.raises(ValueError, match="non-finite"):
            SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        assert not os.path.exists(fpath)
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftovers == [], f"temp files left behind: {leftovers}"

    def test_cubes_no_temp_files_after_success(self, tmp_path):
        fpath = str(tmp_path / "cubes.inc")
        cubes = {"P": np.ones((2, 2, 2), dtype="float32")}
        SaveGSLIBCubes(cubes, fpath, "cap", basedir=str(tmp_path))
        assert os.path.exists(fpath)
        leftovers = [p.name for p in tmp_path.iterdir() if ".tmp." in p.name]
        assert leftovers == [], f"temp files left behind: {leftovers}"

    def test_target_content_complete(self, tmp_path):
        fpath = str(tmp_path / "ps.inc")
        ps = {"P": np.array([1.5, 2.5, 3.5], dtype="float32")}
        SaveGSLIBPointSet(ps, fpath, "cap", basedir=str(tmp_path))
        content = open(fpath).read()
        assert "cap" in content
        assert "P" in content
        assert "1.5" in content


# =============================================================================
# F-N16-py — geo.py write wrappers reject crafted prop_name before FFI
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestWriteWrappersRejectCraftedNames:
    """F-N16-py: write_property / write_gslib_property validate prop_name."""

    def _cont_prop(self, n=3):
        data = np.arange(n, dtype="float32")
        mask = np.ones(n, dtype="uint8")
        return ContProperty(data, mask)

    @pytest.mark.parametrize("bad_name", ["A\nB", "--x", "/x", " x", "x ", ""])
    def test_write_property_rejects(self, tmp_path, bad_name):
        fpath = str(tmp_path / "bad.inc")
        with pytest.raises(ValueError, match="property name"):
            write_property(self._cont_prop(), fpath, bad_name, -99.0, basedir=str(tmp_path))
        assert not os.path.exists(fpath), "no file should be created on rejection"

    @pytest.mark.parametrize("bad_name", ["A\nB", "--x", "/x", " x", ""])
    def test_write_gslib_property_rejects(self, tmp_path, bad_name):
        fpath = str(tmp_path / "bad.dat")
        with pytest.raises(ValueError, match="property name"):
            write_gslib_property(
                self._cont_prop(), fpath, bad_name, -99.0, basedir=str(tmp_path)
            )
        assert not os.path.exists(fpath)

    def test_valid_name_still_writes(self, tmp_path):
        fpath = str(tmp_path / "ok.inc")
        write_property(self._cont_prop(), fpath, "my prop", -99.0, basedir=str(tmp_path))
        assert os.path.exists(fpath)


# =============================================================================
# F-M18-py — ±1.0e21 GSLIB missing-value trimming
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestGslibSentinelTrimming:
    """F-M18-py: values outside ±1.0e21 treated as missing (mask=0 / NaN)."""

    def test_get_gslib_property_masks_out_of_window(self):
        data = np.array([1.0, 1.0e30, -1.0e22, 5.0], dtype="float64")
        prop_dict = {"prop": data}
        _, mask = get_gslib_property(prop_dict, "prop", -99.0)
        assert mask[0] == 1
        assert mask[1] == 0  # 1e30 > 1.0e21 → missing
        assert mask[2] == 0  # -1e22 < -1.0e21 → missing
        assert mask[3] == 1

    def test_get_gslib_property_exact_window_boundary_kept(self):
        # Strict inequality per the GSLIB convention: exactly ±1.0e21 is data.
        data = np.array([1.0e21, -1.0e21], dtype="float64")
        prop_dict = {"prop": data}
        _, mask = get_gslib_property(prop_dict, "prop", -99.0)
        assert mask[0] == 1
        assert mask[1] == 1

    def test_get_gslib_property_undefined_still_masked(self):
        data = np.array([1.0, -99.0, 1.0e30], dtype="float64")
        prop_dict = {"prop": data}
        _, mask = get_gslib_property(prop_dict, "prop", -99.0)
        assert mask[1] == 0  # exact undefined match
        assert mask[2] == 0  # window trim

    def test_load_gslib_file_trims_out_of_window(self, tmp_path):
        fpath = tmp_path / "sent.gslib"
        # caption / 1 prop / values: 1.0, 1e30 (sentinel), 2.0, 3.0
        fpath.write_text("cap\n1\nP\n1.0\n1.0e30\n2.0\n3.0\n")
        result = LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))
        vals = result["P"].ravel(order="F")
        assert vals[0] == 1.0
        assert np.isnan(vals[1])  # 1e30 trimmed to NaN
        assert vals[2] == 2.0
        assert vals[3] == 3.0

    def test_load_gslib_file_normal_values_unchanged(self, tmp_path):
        fpath = tmp_path / "ok.gslib"
        fpath.write_text("cap\n1\nP\n1.0\n2.0\n3.0\n4.0\n")
        result = LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))
        np.testing.assert_array_equal(result["P"].ravel(order="F"), [1.0, 2.0, 3.0, 4.0])

    def test_load_gslib_file_still_rejects_genuine_nan(self, tmp_path):
        fpath = tmp_path / "nan.gslib"
        fpath.write_text("cap\n1\nP\n1.0\nnan\n3.0\n4.0\n")
        with pytest.raises(ValueError, match="non-finite"):
            LoadGslibFile(str(fpath), property_size=(2, 2, 1), basedir=str(tmp_path))


# =============================================================================
# F-M16 — MovingAverage3D allocation order + radius validation
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestMovingAverageRadiusValidation:
    """F-M16: mask allocated AFTER volume cap; radiuses validated."""

    def test_radius_exceeding_grid_rejected(self):
        cube = np.ones((3, 3, 3), dtype="float32")
        mask = np.ones((3, 3, 3), dtype="uint8")
        # radius 5 > grid dim 3 → must be rejected BEFORE any mask allocation
        with pytest.raises(ValueError, match="exceeds the grid"):
            MovingAverage3D((cube, mask), (5, 5, 5), -999.0, GetCubicalMask)

    def test_zero_radius_rejected(self):
        cube = np.ones((3, 3, 3), dtype="float32")
        mask = np.ones((3, 3, 3), dtype="uint8")
        with pytest.raises(ValueError, match="positive"):
            MovingAverage3D((cube, mask), (0, 0, 0), -999.0, GetCubicalMask)

    def test_get_cubical_mask_rejects_zero_radius(self):
        with pytest.raises(ValueError, match="positive"):
            GetCubicalMask((0, 2, 2))

    def test_get_cubical_mask_rejects_negative_radius(self):
        with pytest.raises(ValueError, match="positive"):
            GetCubicalMask((-1, 2, 2))

    def test_valid_radius_still_works(self):
        cube = np.ones((4, 4, 4), dtype="float32")
        mask = np.ones((4, 4, 4), dtype="uint8")
        result = MovingAverage3D((cube, mask), (1, 1, 1), -999.0, GetCubicalMask)
        assert result.shape == cube.shape

    def test_volume_cap_still_enforced(self, monkeypatch):
        from geo_bsd.validation import ValidationConstants

        monkeypatch.setattr(ValidationConstants, "MAX_MOVING_AVERAGE_VOLUME", 8)
        cube = np.ones((3, 3, 3), dtype="float32")  # volume 27 > 8
        mask = np.ones((3, 3, 3), dtype="uint8")
        with pytest.raises(ValueError, match="exceeds the maximum"):
            MovingAverage3D((cube, mask), (1, 1, 1), -999.0, GetCubicalMask)


# =============================================================================
# F-N21 — lazy DEFAULT_BASE_DIR
# =============================================================================


@pytest.mark.skipif(not FIXES_AVAILABLE, reason="modules not available")
class TestLazyDefaultBaseDir:
    """F-N21: DEFAULT_BASE_DIR resolved lazily, not at import time."""

    def test_reflects_chdir(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        resolved = PathValidator.DEFAULT_BASE_DIR
        assert os.path.realpath(resolved) == os.path.realpath(str(tmp_path))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
