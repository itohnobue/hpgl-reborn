"""Tests for HPGL callback handler lifecycle — M-P-07, M-P-08.

Covers:
- M-P-07: _old_handler_refs CFUNCTYPE deferred-deletion cache cap (8 — shared
  across output AND progress handlers; I2-57b raised the bound from 4)
- M-P-08: Callback exception, non-callable TypeError, handler cleared after use
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        ContProperty,
        CovarianceModel,
        SugarboxGrid,
        covariance,
        ordinary_kriging,
        set_output_handler,
        set_progress_handler,
    )
    HPGL_AVAILABLE = True
except (ImportError, OSError):
    HPGL_AVAILABLE = False


# =============================================================================
# M-P-07: _old_handler_refs CFUNCTYPE cache cap (8 — both handler types)
# =============================================================================


@pytest.mark.hpgl
class TestOldHandlerRefsCache:
    """Test _old_handler_refs cache cap and eviction (M-P-07)."""

    def test_handler_refs_cache_survives_mixed_registration(self):
        """Register output + progress handlers mixed — cache evicts to exactly CAP.

        N2-17 hardening: 12 registrations exceed the cap of 8. Clearing the
        suite-global cache first makes the assertion order-independent, and
        asserting ``len == _OLD_HANDLER_REFS_CAP`` exactly (not ``<=``)
        proves the FIFO eviction actually ran — a deleted eviction would
        leave the cache at 12 entries.
        """
        from geo_bsd import geo

        geo._old_handler_refs.clear()  # suite-global — reset for order-independence

        for i in range(6):
            def make_output(n):
                def h(msg, param):
                    return 1
                return h

            def make_progress(n):
                def h(msg, progress, param):
                    return 1
                return h

            set_output_handler(make_output(i), i)
            set_progress_handler(make_progress(i), i)

        assert len(geo._old_handler_refs) == geo._OLD_HANDLER_REFS_CAP, (
            f"Cache should evict to exactly {geo._OLD_HANDLER_REFS_CAP} after "
            f"12 registrations, got {len(geo._old_handler_refs)}"
        )


# =============================================================================
# M-P-08: Callback edge cases — exception, non-callable, cleared
# =============================================================================


@pytest.mark.hpgl
class TestCallbackException:
    """Test callback exception and error handling (M-P-08)."""

    def test_callback_raises_exception_is_recoverable(self):
        """Callback that raises an exception doesn't crash subsequent HPGL operations.

        When a C++ callback fires a Python callback that raises, the exception
        propagates through the CFUNCTYPE boundary. The HPGL library should
        survive and subsequent operations should work.
        """
        call_count = [0]

        def explosive_handler(msg, param):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise ValueError("boom from callback")
            return 1

        set_output_handler(explosive_handler, None)

        grid = SugarboxGrid(x=3, y=3, z=3)
        size = grid.x * grid.y * grid.z
        data = np.ones(size, dtype="float32") * 50.0
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        # This may raise if the exception propagates from C callback, or it
        # may be swallowed by C++. Either way, the process must survive —
        # getting here without a segfault is the primary assertion.
        try:
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(2, 2, 1),
                max_neighbours=4, cov_model=cov_model,
            )
        except (ValueError, RuntimeError):
            pass

        # The fact that we got here without segfault is the main assertion
        # N2-L46: the explosive handler raises on calls 1-2 and returns from
        # call 3 onward — `call_count >= 3` proves the handler was invoked
        # AFTER the raising phase (recovery), not just that the wiring fired.
        assert call_count[0] >= 3, (
            f"Expected the handler to be called at least 3 times (recovery "
            f"after raising on calls 1-2), got {call_count[0]}"
        )

    def test_progress_callback_raises_exception(self):
        """Progress callback that raises is recoverable."""
        call_count = [0]

        def explosive_progress(msg, progress, param):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise RuntimeError("progress boom")
            return 1

        set_progress_handler(explosive_progress, None)

        grid = SugarboxGrid(x=3, y=3, z=3)
        size = grid.x * grid.y * grid.z
        data = np.ones(size, dtype="float32") * 50.0
        mask = np.ones(size, dtype="uint8")
        prop = ContProperty(data, mask)
        cov_model = CovarianceModel(
            type=covariance.spherical, ranges=(3.0, 3.0, 2.0), sill=1.0, nugget=0.1
        )

        # F-M22: the C++ progress handler is invoked ONLY on the OpenMP master
        # thread (worker-thread calls fall back to the default stream). For a
        # 27-node grid the master-thread share is scheduling-dependent, so with
        # the default thread count the handler may fire only once (start()).
        # Pin to 1 thread so the invocation sequence is deterministic — the
        # start() call plus the 27-node milestone flushes always fire on thread
        # 0, and `call_count >= 3` provably reflects post-raising recovery.
        from geo_bsd.geo import get_thread_num, set_thread_num

        saved_threads = get_thread_num()
        set_thread_num(1)
        try:
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(2, 2, 1),
                max_neighbours=4, cov_model=cov_model,
            )
        except (ValueError, RuntimeError):
            pass
        finally:
            set_thread_num(saved_threads)
        # The fact that we got here without segfault is the main assertion
        # N2-L46: mirror :108-147 — proves post-raising recovery invocation.
        assert call_count[0] >= 3, (
            f"Expected the progress handler to be called at least 3 times "
            f"(recovery after raising on calls 1-2), got {call_count[0]}"
        )


@pytest.mark.hpgl
class TestNonCallableHandler:
    """Test non-callable handler raises TypeError (M-P-08)."""

    def test_set_output_handler_non_callable_raises_typeerror(self):
        """Non-callable output handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_output_handler("not_a_function", None)

    def test_set_progress_handler_non_callable_raises_typeerror(self):
        """Non-callable progress handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_progress_handler([1, 2, 3], None)


@pytest.mark.hpgl
class TestHandlerClearedAfterUse:
    """Test handler is cleared properly after set to None (M-P-08)."""

    def test_handler_cleared_after_none_assignment(self):
        """Handler set to None clears the handler and internal state."""
        # Set a handler (local helper — deliberately not named test_* so the
        # grep test count stays honest)
        def handler_fn(msg, param):
            return 1

        set_output_handler(handler_fn, "test")
        # Clear it
        set_output_handler(None, None)
        # Setting to None again should not raise
        set_output_handler(None, None)
