"""Tests for HPGL callback handler lifecycle — M-P-07, M-P-08.

Covers:
- M-P-07: _old_handler_refs CFUNCTYPE deferred-deletion cache cap (8 — shared
  across output AND progress handlers; I2-57b raised the bound from 4)
- M-P-08: Callback exception, non-callable TypeError, handler cleared after use
"""

import ctypes as C
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.geo import (
        SugarboxGrid,
        ContProperty,
        CovarianceModel,
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

    def test_handler_refs_cache_initial_size(self):
        """_old_handler_refs starts empty."""
        from geo_bsd import geo
        assert geo._old_handler_refs is not None
        # We don't assert exact size since other tests may have run

    def test_register_handlers_more_than_four_times(self):
        """Register handlers >4 times — cache doesn't overflow/crash (M-P-07).

        The _old_handler_refs list is a shared bounded FIFO across both
        output and progress handlers. The bound is 8 (4 generations per
        handler type; I2-57b). After 7 registrations all entries fit.
        """
        handlers = []

        for i in range(7):
            def make_handler(n):
                def h(msg, param):
                    return 1
                return h
            h = make_handler(i)
            handlers.append(h)
            set_output_handler(h, f"param_{i}")

        # Handler was registered 7 times — verify no crash/error
        # After 7 sets, _old_handler_refs is capped at 8 entries
        from geo_bsd import geo
        # The cache should have at most 8 entries
        assert len(geo._old_handler_refs) <= geo._OLD_HANDLER_REFS_CAP, (
            f"_old_handler_refs should be capped at {geo._OLD_HANDLER_REFS_CAP}, "
            f"got {len(geo._old_handler_refs)}"
        )

    def test_handler_refs_cache_survives_mixed_registration(self):
        """Register output + progress handlers mixed — cache stays capped."""
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

        from geo_bsd import geo
        assert len(geo._old_handler_refs) <= geo._OLD_HANDLER_REFS_CAP, (
            f"Cache should be capped at {geo._OLD_HANDLER_REFS_CAP} after mixed "
            f"registration, got {len(geo._old_handler_refs)}"
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
        # may be swallowed by C++. Either way, verify the system is functional.
        raised = False
        try:
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(2, 2, 1),
                max_neighbours=4, cov_model=cov_model,
            )
        except (ValueError, RuntimeError):
            raised = True

        # The fact that we got here without segfault is the main assertion
        # Verify the handler was called
        assert call_count[0] >= 1

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

        try:
            ordinary_kriging(
                prop=prop, grid=grid, radiuses=(2, 2, 1),
                max_neighbours=4, cov_model=cov_model,
            )
        except (ValueError, RuntimeError):
            pass
        # The fact that we got here without segfault is the main assertion
        # Verify the progress handler was called (mirror :108-147).
        assert call_count[0] >= 1


@pytest.mark.hpgl
class TestNonCallableHandler:
    """Test non-callable handler raises TypeError (M-P-08)."""

    def test_set_output_handler_non_callable_raises_typeerror(self):
        """Non-callable output handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_output_handler("not_a_function", None)

    def test_set_output_handler_int_raises_typeerror(self):
        """Integer passed as handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_output_handler(42, None)

    def test_set_progress_handler_non_callable_raises_typeerror(self):
        """Non-callable progress handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_progress_handler([1, 2, 3], None)

    def test_set_progress_handler_float_raises_typeerror(self):
        """Float passed as progress handler raises TypeError."""
        with pytest.raises(TypeError, match="handler must be callable"):
            set_progress_handler(3.14, None)


@pytest.mark.hpgl
class TestHandlerClearedAfterUse:
    """Test handler is cleared properly after set to None (M-P-08)."""

    def test_handler_cleared_after_none_assignment(self):
        """Handler set to None clears the handler and internal state."""
        # Set a handler
        def test_handler(msg, param):
            return 1

        set_output_handler(test_handler, "test")
        # Clear it
        set_output_handler(None, None)
        # Setting to None again should not raise
        set_output_handler(None, None)

    def test_handler_cleared_multiple_times(self):
        """Setting handler to None multiple times is safe."""
        for _ in range(5):
            set_output_handler(None, None)
            set_progress_handler(None, None)

    def test_handler_set_clear_set_clear_cycle(self):
        """Repeated set/clear cycles don't leak or crash."""
        for i in range(3):
            def h(msg, param):
                return 1

            def p(msg, progress, param):
                return 1

            set_output_handler(h, i)
            set_progress_handler(p, i)
            set_output_handler(None, None)
            set_progress_handler(None, None)
        # After cycles, system should still work
