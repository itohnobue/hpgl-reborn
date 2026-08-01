"""Regression tests for s6r FIX CONVERGENCE pass — Python-side fixes (PR-01..PR-08).

Covers:
- PR-01  cvariogram error guard: success-after-failure must NOT false-raise
         (cvar_clear_last_error wired into the guard; clear-on-consume
         mirroring ffi_adapter's F-07 pattern)
- PR-02  ffi_adapter _hpgl_call_lock is reentrant (RLock) so a handler
         calling set_output_handler from inside a callback does not
         self-deadlock (empirically reproduced TIMEOUT-DEADLOCK pre-fix)
- PR-05  geo.py kriging entry points wire MIN_KRIGING_RADIUS so a
         zero-radius search is rejected at the Python level (F-34 partial —
         the constant was dead code in validation.py)
- PR-08  hpgl_wrap freshness check detects a stale _cvariogram missing
         cvar_clear_last_error (symbol list was too small to catch it)

Each test fails against the pre-fix code and passes against the fixed code.
"""

import _thread
import logging
import sys
import threading
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.validation import CriticalValidationError
    VALIDATION_AVAILABLE = True
except Exception:
    VALIDATION_AVAILABLE = False

try:
    import geo_bsd.cvariogram as cvmod
    CVAR_AVAILABLE = True
except Exception:
    CVAR_AVAILABLE = False

try:
    import geo_bsd.geo as geo_mod
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


# ===========================================================================
# PR-01: cvariogram error guard — success-after-failure must not false-raise
# ===========================================================================


@pytest.mark.skipif(not CVAR_AVAILABLE, reason="cvariogram C library not available")
class TestPR01CvarSuccessAfterFailure:
    """PR-01: the C++ error is process-global and never cleared on success.
    The guard must consume it (cvar_clear_last_error) on the raise path so a
    later SUCCESSFUL call starts from a clean state instead of false-raising
    on the stale error forever."""

    def test_success_after_failure_does_not_false_raise(self, monkeypatch):
        cvmod._cvar_error_local = type(cvmod._cvar_error_local)()
        state = {"err": None}

        class _FakeCvar:
            @staticmethod
            def cvar_get_last_error():
                return state["err"]

            @staticmethod
            def cvar_clear_last_error():
                state["err"] = b""

        monkeypatch.setattr(cvmod, "cvar", _FakeCvar())

        # CALL1: the C++ computation fails (error set by the C side).
        cvmod._snapshot_cvar_error()
        state["err"] = b"no valid pairs found in search template"
        with pytest.raises(RuntimeError, match="no valid pairs"):
            cvmod._check_cvar_error("CalcVariograms")

        # The guard must have CONSUMED the C++ error on the raise path.
        # Pre-fix the error stays set (process-global, never cleared) →
        # this assertion fails.
        assert state["err"] == b""

        # CALL2: a subsequent SUCCESSFUL call (no new C++ error) must NOT
        # raise — pre-fix it false-raises on the stale error forever.
        cvmod._snapshot_cvar_error()
        cvmod._check_cvar_error("CalcVariograms")  # must NOT raise

    def test_clear_cvar_error_noop_when_symbol_absent(self, monkeypatch):
        """PR-01: _clear_cvar_error must no-op when the loaded library lacks
        cvar_clear_last_error (stale binary, detected by the PR-08 freshness
        check) instead of crashing — the snapshot/flag logic then still
        suppresses stale errors (pre-fix behavior)."""

        class _StaleCvar:
            def cvar_get_last_error(self):
                return None  # no clear symbol

        monkeypatch.setattr(cvmod, "cvar", _StaleCvar())
        cvmod._clear_cvar_error()  # must not raise

    def test_consecutive_identical_failures_still_both_raise(self, monkeypatch):
        """PR-01 must not regress F-37: two consecutive identical failures
        both raise (the clear-on-consume must not suppress a genuinely new
        identical error in a new call window)."""
        cvmod._cvar_error_local = type(cvmod._cvar_error_local)()
        state = {"err": None}

        class _FakeCvar:
            @staticmethod
            def cvar_get_last_error():
                return state["err"]

            @staticmethod
            def cvar_clear_last_error():
                state["err"] = b""

        monkeypatch.setattr(cvmod, "cvar", _FakeCvar())

        cvmod._snapshot_cvar_error()
        state["err"] = b"singular matrix"
        with pytest.raises(RuntimeError, match="singular matrix"):
            cvmod._check_cvar_error("op1")

        # CALL2 fails with the IDENTICAL message → must raise again.
        cvmod._snapshot_cvar_error()
        state["err"] = b"singular matrix"
        with pytest.raises(RuntimeError, match="singular matrix"):
            cvmod._check_cvar_error("op2")


# ===========================================================================
# PR-02: _hpgl_call_lock reentrancy — no handler self-deadlock
# ===========================================================================


class TestPR02ReentrantCallLock:
    """PR-02: C++ invokes user output/progress handlers on the calling thread
    DURING kriging/simulation FFI calls that hold _hpgl_call_lock. A handler
    calling set_output_handler (or re-entering kriging) self-deadlocks on a
    plain Lock; RLock allows the same thread to re-acquire."""

    def test_hpgl_call_lock_is_reentrant_rlock(self):
        from geo_bsd import ffi_adapter

        # threading.RLock is a factory function (not a type); the underlying
        # C object is _thread.RLock.
        assert isinstance(ffi_adapter._hpgl_call_lock, _thread.RLock)

    def test_reentrant_set_output_handler_from_handler_does_not_deadlock(self):
        """Simulates the exact deadlock scenario: a kriging FFI call holds
        _hpgl_call_lock while C++ invokes the user output handler on the same
        thread, and the handler calls set_output_handler (re-enters the
        lock). RLock completes; a plain Lock hangs forever (pre-fix)."""
        from geo_bsd import ffi_adapter
        from geo_bsd.geo import set_output_handler

        finished = []

        def handler_thread():
            # Kriging FFI call acquires the lock...
            ffi_adapter._hpgl_call_lock.acquire()
            try:
                # ...C++ fires the output handler on this thread, and the
                # handler re-enters set_output_handler (same-thread acquire).
                set_output_handler(lambda msg, p: 0, None)
            finally:
                ffi_adapter._hpgl_call_lock.release()
            finished.append(True)

        t = threading.Thread(target=handler_thread, daemon=True)
        t.start()
        t.join(timeout=5)
        assert not t.is_alive(), "DEADLOCK: reentrant set_output_handler blocked"
        assert finished == [True]

    def test_cross_thread_exclusion_still_enforced(self):
        """RLock must still serialize ACROSS threads: a second thread cannot
        acquire while the first holds the lock."""
        from geo_bsd import ffi_adapter

        lock = ffi_adapter._hpgl_call_lock
        lock.acquire()
        try:
            second = []

            def other():
                second.append(lock.acquire(blocking=False))

            t = threading.Thread(target=other)
            t.start()
            t.join(timeout=5)
            assert second == [False], "cross-thread exclusion broken by RLock"
        finally:
            lock.release()


# ===========================================================================
# PR-05: kriging entry points reject zero-radius via MIN_KRIGING_RADIUS
# ===========================================================================


@pytest.mark.skipif(not HPGL_AVAILABLE, reason="HPGL (geo_bsd.geo) not available")
@pytest.mark.skipif(not VALIDATION_AVAILABLE, reason="validation not available")
class TestPR05MinKrigingRadius:
    """PR-05: MIN_KRIGING_RADIUS was dead code (validation.py only). Wire it
    into every geo.py kriging entry point so a zero-radius search (which
    silently mean-fills every node in C++) is rejected at the Python level.
    SGS zero-radius CDF-draw behavior is untouched (sgs.py does not pass the
    kriging minimum)."""

    @staticmethod
    def _grid():
        return geo_mod.SugarboxGrid(x=5, y=5, z=5)

    @staticmethod
    def _cov():
        return geo_mod.CovarianceModel(
            type=geo_mod.covariance.spherical,
            ranges=(3.0, 3.0, 3.0),
            sill=1.0,
            nugget=0.1,
        )

    @staticmethod
    def _cont_prop():
        rng = np.random.RandomState(42)
        return geo_mod.ContProperty(
            rng.rand(125).astype("float32") * 100, np.ones(125, dtype="uint8")
        )

    @staticmethod
    def _ind_prop(count=2):
        return geo_mod.IndProperty(
            np.zeros(125, dtype="uint8"), np.ones(125, dtype="uint8"), count
        )

    def test_ordinary_kriging_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.ordinary_kriging(
                self._cont_prop(), self._grid(), (0, 0, 0), 8, self._cov()
            )

    def test_simple_kriging_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.simple_kriging(
                self._cont_prop(), self._grid(), (0, 0, 0), 8, self._cov()
            )

    def test_lvm_kriging_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.lvm_kriging(
                self._cont_prop(),
                self._grid(),
                np.zeros(125, dtype="float32"),
                (0, 0, 0),
                8,
                self._cov(),
            )

    def test_median_ik_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.median_ik(
                self._ind_prop(2), self._grid(), (0.5, 0.5), (0, 0, 0), 8, self._cov()
            )

    def test_indicator_kriging_three_categories_rejects_zero_radius(self):
        """The ≥3-category path calls hpgl_indicator_kriging directly (the
        2-category path redirects to median_ik); its per-indicator radius
        validation must reject radius 0 too."""
        prop = self._ind_prop(3)
        data = [
            {"cov_model": self._cov(), "radiuses": (0, 0, 0), "max_neighbours": 8}
            for _ in range(3)
        ]
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.indicator_kriging(prop, self._grid(), data, (0.3, 0.3, 0.4))

    def test_simple_cokriging_markI_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.simple_cokriging_markI(
                self._cont_prop(),
                self._grid(),
                (0, 0, 0),
                8,
                self._cov(),
                self._cont_prop(),
                1.0,
                1.0,
                1.0,
                0.5,
            )

    def test_simple_cokriging_markII_rejects_zero_radius(self):
        with pytest.raises(CriticalValidationError, match="less than minimum"):
            geo_mod.simple_cokriging_markII(
                self._grid(), None, None, 0.5, (0, 0, 0), 8
            )

    def test_sgs_zero_radius_cdf_draw_behavior_untouched(self):
        """The kriging minimum must NOT leak into SGS: sgs.py validates with
        the default MIN_RADIUS=0.0 so the documented zero-radius CDF-draw
        behavior keeps working."""
        from geo_bsd.validation import validate_kriging_params

        # Direct validation path sgs.py uses (no min_radius kwarg) accepts 0.
        valid = validate_kriging_params(self._grid(), (0, 0, 0), 8, self._cov())
        assert valid == (0, 0, 0)


# ===========================================================================
# PR-08: freshness check detects stale _cvariogram missing cvar_clear_last_error
# ===========================================================================


class TestPR08CvariogramFreshnessSymbol:
    """PR-08: _EXPECTED_LIBRARY_SYMBOLS["_cvariogram"] previously checked only
    cvar_get_last_error + cvar_stack_layers — both present in the stale
    deployed lib, so the freshness check passed silently. Adding
    cvar_clear_last_error makes the stale binary detectable at load."""

    def test_expected_symbols_include_clear_last_error(self):
        import geo_bsd.hpgl_wrap as hw

        assert "cvar_clear_last_error" in hw._EXPECTED_LIBRARY_SYMBOLS["_cvariogram"]

    def test_freshness_warns_when_cvar_clear_missing(self, caplog):
        import geo_bsd.hpgl_wrap as hw

        stale_lib = mock.Mock(spec=["cvar_get_last_error", "cvar_stack_layers"])
        with caplog.at_level(logging.WARNING, logger="geo_bsd.hpgl_wrap"):
            hw._verify_library_freshness("_cvariogram", stale_lib)
        assert "does not export expected symbols" in caplog.text
        assert "cvar_clear_last_error" in caplog.text

    def test_freshness_silent_when_all_symbols_present(self, caplog):
        import geo_bsd.hpgl_wrap as hw

        fresh_lib = mock.MagicMock()
        fresh_lib.cvar_get_last_error = mock.Mock()
        fresh_lib.cvar_stack_layers = mock.Mock()
        fresh_lib.cvar_clear_last_error = mock.Mock()
        with caplog.at_level(logging.WARNING, logger="geo_bsd.hpgl_wrap"):
            hw._verify_library_freshness("_cvariogram", fresh_lib)
        assert "does not export expected symbols" not in caplog.text
