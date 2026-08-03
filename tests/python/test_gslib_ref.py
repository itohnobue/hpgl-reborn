"""GSLIB reference-fact table tests (got-20260802092630 — recurring class).

The GSLIB contract facts (sentinel window, data-vs-normal-score space, ndmin
original-data-only, OK->SK downgrade, ordrel) drifted across runs because
they were duplicated as inline constants in geo.py / routines.py /
read_inc_file.cpp. The table in ``geo_bsd.gslib_ref`` is the single source
of truth; these tests pin the semantics so a path cannot silently rely on a
different interpretation.

The sentinel-window tests target the EXACT boundary: strict inequality at
±1.0e21 (an exact ±1.0e21 value is a real data point; |v| > 1.0e21 is a
missing sentinel). This is the exact boundary value that a regression test
must straddle (got-20260802092618) — a test on 9e20 vs 1.1e21 only checks
the coarse behavior.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from geo_bsd import gslib_ref  # noqa: E402


class TestGslibReferenceFacts:
    """The reference-fact table exists and documents every contract facet."""

    def test_sentinel_window_constant(self):
        assert gslib_ref.GSLIB_SENTINEL_WINDOW == 1.0e21

    def test_data_and_normal_score_space_constants(self):
        assert gslib_ref.GSLIB_DATA_SPACE == "data"
        assert gslib_ref.GSLIB_NORMAL_SCORE_SPACE == "normal-score"
        assert gslib_ref.GSLIB_DATA_SPACE != gslib_ref.GSLIB_NORMAL_SCORE_SPACE

    def test_ndmin_original_data_only(self):
        assert gslib_ref.GSLIB_NDMIN_ORIGINAL_DATA_ONLY is True

    def test_ok_sk_downgrade_drops_secondary_equation(self):
        assert gslib_ref.GSLIB_OK_SK_DOWNGRADE_DROPS_SECONDARY_EQUATION is True

    def test_ordrel_space(self):
        assert gslib_ref.GSLIB_ORDREL_SPACE == "posterior-probability"


class TestGslibSentinelWindow:
    """The sentinel window is STRICT inequality at ±1.0e21 (GSLIB doc:
    'less than -1.0e21 or greater than 1.0e21')."""

    def test_exact_positive_edge_is_not_sentinel(self):
        # 1.0e21 exactly: strict > 1.0e21 is false → real data.
        assert gslib_ref.is_gslib_sentinel(1.0e21) is False
        assert not gslib_ref.is_gslib_missing_sentinel(np.array([1.0e21]))[0]

    def test_exact_negative_edge_is_not_sentinel(self):
        assert gslib_ref.is_gslib_sentinel(-1.0e21) is False
        assert not gslib_ref.is_gslib_missing_sentinel(np.array([-1.0e21]))[0]

    def test_just_above_edge_is_sentinel(self):
        # 1.0e21 + 1 ulp: strictly greater → sentinel.
        above = np.nextafter(1.0e21, np.inf)
        assert gslib_ref.is_gslib_sentinel(above) is True
        assert gslib_ref.is_gslib_missing_sentinel(np.array([above]))[0]

    def test_just_below_edge_is_real(self):
        below = np.nextafter(1.0e21, -np.inf)
        assert gslib_ref.is_gslib_sentinel(below) is False
        assert not gslib_ref.is_gslib_missing_sentinel(np.array([below]))[0]

    def test_typical_values_not_sentinel(self):
        assert gslib_ref.is_gslib_sentinel(0.0) is False
        assert gslib_ref.is_gslib_sentinel(-999.0) is False  # INC sentinel ≠ GSLIB sentinel
        assert gslib_ref.is_gslib_sentinel(1e10) is False

    def test_vectorized(self):
        vals = np.array([0.0, 1.0e21, 2.0e21, -2.0e21, -1.0e21, 1.5])
        result = gslib_ref.is_gslib_missing_sentinel(vals)
        assert result.tolist() == [False, False, True, True, False, False]
