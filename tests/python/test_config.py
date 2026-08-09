"""
Unit tests for HPGL config dataclasses (config.py).

Covers SGSConfig, SISConfig, GTSIMConfig — valid construction, type errors,
value errors, and NaN/Inf rejection (I2-F02: zero dedicated test coverage).
"""

from __future__ import annotations

import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

try:
    from geo_bsd.config import GTSIMConfig, SGSConfig, SISConfig

    CONFIG_AVAILABLE = True
except (ImportError, SyntaxError, IndentationError):
    CONFIG_AVAILABLE = False


# =============================================================================
# SGSConfig Tests
# =============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="config module not available")
class TestSGSConfig:
    """Test SGSConfig validation at construction time."""

    def test_default_construction(self):
        """Default parameters construct successfully."""
        cfg = SGSConfig()
        assert cfg.kriging_type == "sk"
        assert cfg.seed == 0
        assert cfg.min_neighbours == 0
        assert cfg.max_neighbours == 12
        assert cfg.radiuses == (5, 5, 3)
        assert cfg.use_harddata is True

    def test_custom_params(self):
        """Custom parameters construct with correct values."""
        cfg = SGSConfig(
            kriging_type="ok", seed=42, min_neighbours=2,
            max_neighbours=50, radiuses=(10, 20, 30), use_harddata=False,
        )
        assert cfg.kriging_type == "ok"
        assert cfg.seed == 42
        assert cfg.max_neighbours == 50
        assert cfg.radiuses == (10, 20, 30)

    def test_frozen_prevents_mutation(self):
        """Frozen dataclass prevents attribute reassignment."""
        cfg = SGSConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.seed = 123  # type: ignore[misc]

    # ---- kriging_type ----

    def test_invalid_kriging_type_raises(self):
        """Invalid kriging_type raises ValueError."""
        with pytest.raises(ValueError, match="kriging_type"):
            SGSConfig(kriging_type="invalid")

    # ---- seed ----

    def test_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed"):
            SGSConfig(seed=-1)

    def test_bool_seed_raises(self):
        """Bool seed raises TypeError."""
        with pytest.raises(TypeError, match="seed"):
            SGSConfig(seed=True)  # type: ignore[arg-type]

    def test_non_int_seed_raises(self):
        """Non-integer seed raises TypeError."""
        with pytest.raises(TypeError, match="seed"):
            SGSConfig(seed=3.14)  # type: ignore[arg-type]

    def test_numpy_integer_seed_accepted(self):
        """NumPy integer seed is accepted."""
        cfg = SGSConfig(seed=np.int32(5))
        assert cfg.seed == 5

    # ---- min_neighbours ----

    def test_negative_min_neighbours_raises(self):
        """Negative min_neighbours raises ValueError."""
        with pytest.raises(ValueError, match="min_neighbours"):
            SGSConfig(min_neighbours=-1)

    def test_bool_min_neighbours_raises(self):
        """Bool min_neighbours raises TypeError."""
        with pytest.raises(TypeError, match="min_neighbours"):
            SGSConfig(min_neighbours=True)  # type: ignore[arg-type]

    # ---- max_neighbours ----

    def test_zero_max_neighbours_accepted(self):
        """Zero max_neighbours accepted (R-05/E-M9 unconditional mode)."""
        cfg = SGSConfig(max_neighbours=0)
        assert cfg.max_neighbours == 0

    def test_negative_max_neighbours_raises(self):
        """Negative max_neighbours raises ValueError."""
        with pytest.raises(ValueError, match="max_neighbours"):
            SGSConfig(max_neighbours=-5)

    def test_max_neighbours_over_hard_limit_raises(self):
        """M-20: max_neighbours above the C++ hard limit (10000) is rejected.

        The discriminating boundary is 10001 — a limit regression to 100000
        passes every pre-existing test (which only cover 100001-rejected and
        10000-accepted), so this pins the exact cap+1 rejection.
        """
        with pytest.raises(ValueError, match="exceeds the maximum"):
            SGSConfig(max_neighbours=10001)

    def test_max_neighbours_at_hard_limit_accepted(self):
        """M-20: max_neighbours == the C++ hard limit (10000) is accepted."""
        cfg = SGSConfig(max_neighbours=10000)
        assert cfg.max_neighbours == 10000

    # ---- radiuses ----

    def test_radiuses_tuple_accepted(self):
        """Tuple radiuses accepted — uses default."""
        cfg = SGSConfig(radiuses=(7, 8, 9))
        assert cfg.radiuses == (7, 8, 9)

    def test_radiuses_non_tuple_raises(self):
        """Non-tuple radiuses raises TypeError."""
        with pytest.raises(TypeError, match="radiuses"):
            SGSConfig(radiuses=[1, 2, 3])  # type: ignore[arg-type]

    def test_radiuses_wrong_length_raises(self):
        """Wrong-length radiuses tuple raises ValueError."""
        with pytest.raises(ValueError, match="radiuses"):
            SGSConfig(radiuses=(1, 2))

    @pytest.mark.parametrize("bad_value,desc", [
        (float("nan"), "NaN"),
        (float("inf"), "Inf"),
        (-float("inf"), "-Inf"),
    ])
    def test_radiuses_nan_inf_raises(self, bad_value, desc):
        """NaN/Inf in radiuses raises ValueError (F-02)."""
        with pytest.raises(ValueError, match="radiuses"):
            SGSConfig(radiuses=(5, bad_value, 3))

    def test_negative_radius_raises(self):
        """Negative radius raises ValueError."""
        with pytest.raises(ValueError, match="radiuses"):
            SGSConfig(radiuses=(5, -1, 3))

    def test_zero_radius_accepted(self):
        """Zero radius is accepted (F-11 — aligned with ParameterValidator)."""
        cfg = SGSConfig(radiuses=(0, 0, 0))
        assert cfg.radiuses == (0, 0, 0)

    # ---- use_harddata ----

    def test_non_bool_use_harddata_raises(self):
        """Non-bool use_harddata raises TypeError."""
        with pytest.raises(TypeError, match="use_harddata"):
            SGSConfig(use_harddata="yes")  # type: ignore[arg-type]


# =============================================================================
# SISConfig Tests
# =============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="config module not available")
class TestSISConfig:
    """Test SISConfig validation at construction time."""

    def test_default_construction(self):
        """Default parameters construct successfully."""
        cfg = SISConfig()
        assert cfg.seed == 0
        assert cfg.max_neighbours == 12
        assert cfg.radiuses == (5, 5, 3)
        assert cfg.use_harddata is True
        assert cfg.use_correlogram is True
        assert cfg.marginal_probs is None

    def test_custom_params(self):
        """Custom parameters construct correctly."""
        cfg = SISConfig(
            seed=99, max_neighbours=30, radiuses=(15, 20, 5),
            use_harddata=False, use_correlogram=False,
            marginal_probs=(0.3, 0.7),
        )
        assert cfg.seed == 99
        assert cfg.marginal_probs == (0.3, 0.7)

    def test_frozen_prevents_mutation(self):
        """Frozen dataclass prevents attribute reassignment."""
        cfg = SISConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.seed = 99  # type: ignore[misc]

    # ---- seed ----

    def test_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed"):
            SISConfig(seed=-1)

    def test_bool_seed_raises(self):
        """Bool seed raises TypeError."""
        with pytest.raises(TypeError, match="seed"):
            SISConfig(seed=True)  # type: ignore[arg-type]

    # ---- max_neighbours ----

    def test_zero_max_neighbours_accepted(self):
        """Zero max_neighbours accepted (R-05/E-M9 unconditional mode)."""
        cfg = SISConfig(max_neighbours=0)
        assert cfg.max_neighbours == 0

    def test_max_neighbours_over_hard_limit_raises(self):
        """M-20: max_neighbours above the C++ hard limit (10000) is rejected."""
        with pytest.raises(ValueError, match="exceeds the maximum"):
            SISConfig(max_neighbours=10001)

    def test_max_neighbours_at_hard_limit_accepted(self):
        """M-20: max_neighbours == the C++ hard limit (10000) is accepted."""
        cfg = SISConfig(max_neighbours=10000)
        assert cfg.max_neighbours == 10000

    # ---- radiuses NaN/Inf (F-02) ----

    @pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
    def test_radiuses_nan_inf_raises(self, bad_value):
        """NaN/Inf in radiuses raises ValueError (F-02)."""
        with pytest.raises(ValueError, match="radiuses"):
            SISConfig(radiuses=(5, bad_value, 3))

    def test_zero_radius_accepted(self):
        """Zero radius accepted (F-11 — aligned with ParameterValidator)."""
        cfg = SISConfig(radiuses=(0, 0, 0))
        assert cfg.radiuses == (0, 0, 0)

    # ---- use_harddata ----

    def test_non_bool_use_harddata_raises(self):
        """Non-bool use_harddata raises TypeError."""
        with pytest.raises(TypeError, match="use_harddata"):
            SISConfig(use_harddata=1)  # type: ignore[arg-type]

    # ---- use_correlogram ----

    def test_non_bool_use_correlogram_raises(self):
        """Non-bool use_correlogram raises TypeError."""
        with pytest.raises(TypeError, match="use_correlogram"):
            SISConfig(use_correlogram="yes")  # type: ignore[arg-type]

    # ---- marginal_probs ----

    def test_marginal_probs_tuple_accepted(self):
        """Tuple marginal_probs accepted."""
        cfg = SISConfig(marginal_probs=(0.2, 0.3, 0.5))
        assert cfg.marginal_probs == (0.2, 0.3, 0.5)

    def test_marginal_probs_non_tuple_raises(self):
        """Non-tuple marginal_probs raises TypeError."""
        with pytest.raises(TypeError, match="marginal_probs"):
            SISConfig(marginal_probs=[0.2, 0.8])  # type: ignore[arg-type]


# =============================================================================
# GTSIMConfig Tests
# =============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="config module not available")
class TestGTSIMConfig:
    """Test GTSIMConfig validation at construction time."""

    def test_default_construction(self):
        """Default parameters construct successfully."""
        cfg = GTSIMConfig()
        assert cfg.tk_mean == 0.0
        assert cfg.tk_std_dev == 1.0
        assert cfg.seed == 3439275

    def test_custom_params(self):
        """Custom parameters construct correctly."""
        cfg = GTSIMConfig(tk_mean=5.0, tk_std_dev=2.0, seed=42)
        assert cfg.tk_mean == 5.0
        assert cfg.tk_std_dev == 2.0
        assert cfg.seed == 42

    def test_frozen_prevents_mutation(self):
        """Frozen dataclass prevents attribute reassignment."""
        cfg = GTSIMConfig()
        with pytest.raises(FrozenInstanceError):
            cfg.tk_mean = 1.0  # type: ignore[misc]

    # ---- tk_mean ----

    def test_tk_mean_non_number_raises(self):
        """Non-number tk_mean raises TypeError."""
        with pytest.raises(TypeError, match="tk_mean"):
            GTSIMConfig(tk_mean="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value,desc", [
        (float("nan"), "NaN"),
        (float("inf"), "Inf"),
        (-float("inf"), "-Inf"),
    ])
    def test_tk_mean_nan_inf_raises(self, bad_value, desc):
        """NaN/Inf tk_mean raises ValueError."""
        with pytest.raises(ValueError, match="tk_mean"):
            GTSIMConfig(tk_mean=bad_value)

    # ---- tk_std_dev ----

    def test_tk_std_dev_non_number_raises(self):
        """Non-number tk_std_dev raises TypeError."""
        with pytest.raises(TypeError, match="tk_std_dev"):
            GTSIMConfig(tk_std_dev="bad")  # type: ignore[arg-type]

    @pytest.mark.parametrize("bad_value,desc", [
        (float("nan"), "NaN"),
        (float("inf"), "Inf"),
        (-float("inf"), "-Inf"),
    ])
    def test_tk_std_dev_nan_inf_raises(self, bad_value, desc):
        """NaN/Inf tk_std_dev raises ValueError."""
        with pytest.raises(ValueError, match="tk_std_dev"):
            GTSIMConfig(tk_std_dev=bad_value)

    def test_tk_std_dev_negative_raises(self):
        """Negative tk_std_dev raises ValueError."""
        with pytest.raises(ValueError, match="tk_std_dev"):
            GTSIMConfig(tk_std_dev=-1.0)

    def test_tk_std_dev_zero_raises(self):
        """Zero tk_std_dev raises ValueError."""
        with pytest.raises(ValueError, match="tk_std_dev"):
            GTSIMConfig(tk_std_dev=0.0)

    # ---- seed ----

    def test_negative_seed_raises(self):
        """Negative seed raises ValueError."""
        with pytest.raises(ValueError, match="seed"):
            GTSIMConfig(seed=-1)

    def test_bool_seed_raises(self):
        """Bool seed raises TypeError."""
        with pytest.raises(TypeError, match="seed"):
            GTSIMConfig(seed=True)  # type: ignore[arg-type]

    def test_numpy_integer_seed_accepted(self):
        """NumPy integer seed is accepted."""
        cfg = GTSIMConfig(seed=np.int64(10))
        assert cfg.seed == 10


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
