# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""Frozen configuration dataclasses for HPGL simulation parameters.

Provides immutable, validated parameter objects that can be passed
to simulation entry points (``sgs_simulation``, ``sis_simulation``,
``gtsim_2ind``) as an alternative to specifying parameters individually.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy

# ============================================================================
# SGS Configuration
# ============================================================================


@dataclass(frozen=True)
class SGSConfig:
    """Frozen configuration for Sequential Gaussian Simulation (SGS).

    All fields are validated at construction time.  Once created the
    instance cannot be mutated, ensuring simulation parameters are
    consistent throughout a workflow.

    Parameters
    ----------
    kriging_type : str
        Kriging method: ``"sk"`` for Simple Kriging or ``"ok"`` for
        Ordinary Kriging.  Default: ``"sk"``.
    seed : int
        Seed for the random number generator.  Must be non-negative.
        Default: ``0``.
    min_neighbours : int
        Minimum number of neighbours required for kriging.
        Default: ``0``.
    max_neighbours : int
        Maximum number of neighbour points to use for kriging.
        Default: ``12``.
    radiuses : tuple
        Search radiuses in (X, Y, Z) directions.  Each value must be
        a positive number.  Default: ``(5, 5, 3)``.
    use_harddata : bool
        If ``True``, use source data values for simulation.
        Default: ``True``.
    """

    kriging_type: str = "sk"
    seed: int = 0
    min_neighbours: int = 0
    max_neighbours: int = 12
    radiuses: tuple = (5, 5, 3)
    use_harddata: bool = True

    def __post_init__(self) -> None:
        """Validate all fields after dataclass construction.

        Uses ``object.__setattr__`` because the dataclass is frozen
        and normal attribute assignment is blocked.
        """
        # -- kriging_type --
        if self.kriging_type not in ("sk", "ok"):
            raise ValueError(
                f"SGSConfig: kriging_type must be 'sk' or 'ok', "
                f"got {self.kriging_type!r}"
            )

        # -- seed --
        if not isinstance(self.seed, (int, numpy.integer)) or isinstance(self.seed, bool):
            raise TypeError(
                f"SGSConfig: seed must be an int, got {type(self.seed).__name__}"
            )
        if self.seed < 0:
            raise ValueError(
                f"SGSConfig: seed must be non-negative, got {self.seed}"
            )

        # -- min_neighbours --
        if not isinstance(self.min_neighbours, (int, numpy.integer)) or isinstance(self.min_neighbours, bool):
            raise TypeError(
                f"SGSConfig: min_neighbours must be an int, "
                f"got {type(self.min_neighbours).__name__}"
            )
        if self.min_neighbours < 0:
            raise ValueError(
                f"SGSConfig: min_neighbours must be non-negative, "
                f"got {self.min_neighbours}"
            )

        # -- max_neighbours --
        if not isinstance(self.max_neighbours, (int, numpy.integer)) or isinstance(self.max_neighbours, bool):
            raise TypeError(
                f"SGSConfig: max_neighbours must be an int, "
                f"got {type(self.max_neighbours).__name__}"
            )
        if self.max_neighbours < 1:
            raise ValueError(
                f"SGSConfig: max_neighbours must be positive, "
                f"got {self.max_neighbours}"
            )

        # -- radiuses --
        if not isinstance(self.radiuses, tuple):
            raise TypeError(
                f"SGSConfig: radiuses must be a tuple, "
                f"got {type(self.radiuses).__name__}"
            )
        if len(self.radiuses) != 3:
            raise ValueError(
                f"SGSConfig: radiuses must have 3 values, "
                f"got {len(self.radiuses)}"
            )
        import math

        for i, r in enumerate(self.radiuses):
            if not isinstance(r, (int, float)):
                raise TypeError(
                    f"SGSConfig: radiuses[{i}] must be a number, "
                    f"got {type(r).__name__}"
                )
            if math.isnan(r) or math.isinf(r):
                raise ValueError(
                    f"SGSConfig: radiuses[{i}] must be finite, got {r}"
                )
            if r < 0:
                raise ValueError(
                    f"SGSConfig: radiuses[{i}] must be non-negative, got {r}"
                )
            if not float(r).is_integer():
                raise ValueError(
                    f"SGSConfig: radiuses[{i}] = {r} is not an integer "
                    f"(radius must be a whole number of grid cells)"
                )

        # -- use_harddata --
        if not isinstance(self.use_harddata, bool):
            raise TypeError(
                f"SGSConfig: use_harddata must be a bool, "
                f"got {type(self.use_harddata).__name__}"
            )


# ============================================================================
# SIS Configuration
# ============================================================================


@dataclass(frozen=True)
class SISConfig:
    """Frozen configuration for Sequential Indicator Simulation (SIS).

    All fields are validated at construction time.  Once created the
    instance cannot be mutated.

    Parameters
    ----------
    seed : int
        Seed for the random number generator.  Must be non-negative.
        Default: ``0``.
    max_neighbours : int
        Maximum number of neighbour points to use for kriging.
        Default: ``12``.
    radiuses : tuple
        Search radiuses in (X, Y, Z) directions.  Each value must be
        a positive number.  Default: ``(5, 5, 3)``.
    use_harddata : bool
        If ``True``, use source data values for simulation.
        Default: ``True``.
    use_correlogram : bool
        If ``True``, use correlogram-based simulation.  Only applicable
        in LVM mode.  Default: ``True``.
    marginal_probs : tuple or None
        Marginal probabilities for each indicator category.  When
        ``None``, must be provided directly to ``sis_simulation``.
        Default: ``None``.
    """

    seed: int = 0
    max_neighbours: int = 12
    radiuses: tuple = (5, 5, 3)
    use_harddata: bool = True
    use_correlogram: bool = True
    marginal_probs: tuple | None = None

    def __post_init__(self) -> None:
        """Validate all fields after dataclass construction."""
        # -- seed --
        if not isinstance(self.seed, (int, numpy.integer)) or isinstance(self.seed, bool):
            raise TypeError(
                f"SISConfig: seed must be an int, got {type(self.seed).__name__}"
            )
        if self.seed < 0:
            raise ValueError(
                f"SISConfig: seed must be non-negative, got {self.seed}"
            )

        # -- max_neighbours --
        if not isinstance(self.max_neighbours, (int, numpy.integer)) or isinstance(self.max_neighbours, bool):
            raise TypeError(
                f"SISConfig: max_neighbours must be an int, "
                f"got {type(self.max_neighbours).__name__}"
            )
        if self.max_neighbours < 1:
            raise ValueError(
                f"SISConfig: max_neighbours must be positive, "
                f"got {self.max_neighbours}"
            )

        # -- radiuses --
        if not isinstance(self.radiuses, tuple):
            raise TypeError(
                f"SISConfig: radiuses must be a tuple, "
                f"got {type(self.radiuses).__name__}"
            )
        if len(self.radiuses) != 3:
            raise ValueError(
                f"SISConfig: radiuses must have 3 values, "
                f"got {len(self.radiuses)}"
            )
        import math

        for i, r in enumerate(self.radiuses):
            if not isinstance(r, (int, float)):
                raise TypeError(
                    f"SISConfig: radiuses[{i}] must be a number, "
                    f"got {type(r).__name__}"
                )
            if math.isnan(r) or math.isinf(r):
                raise ValueError(
                    f"SISConfig: radiuses[{i}] must be finite, got {r}"
                )
            if r < 0:
                raise ValueError(
                    f"SISConfig: radiuses[{i}] must be non-negative, got {r}"
                )
            if not float(r).is_integer():
                raise ValueError(
                    f"SISConfig: radiuses[{i}] = {r} is not an integer "
                    f"(radius must be a whole number of grid cells)"
                )

        # -- use_harddata --
        if not isinstance(self.use_harddata, bool):
            raise TypeError(
                f"SISConfig: use_harddata must be a bool, "
                f"got {type(self.use_harddata).__name__}"
            )

        # -- use_correlogram --
        if not isinstance(self.use_correlogram, bool):
            raise TypeError(
                f"SISConfig: use_correlogram must be a bool, "
                f"got {type(self.use_correlogram).__name__}"
            )

        # -- marginal_probs --
        if self.marginal_probs is not None:
            if not isinstance(self.marginal_probs, tuple):
                raise TypeError(
                    f"SISConfig: marginal_probs must be a tuple or None, "
                    f"got {type(self.marginal_probs).__name__}"
                )


# ============================================================================
# GTSIM Configuration
# ============================================================================


@dataclass(frozen=True)
class GTSIMConfig:
    """Frozen configuration for Gaussian Truncated Simulation (GTSIM).

    Parameters are validated at construction time.  Once created the
    instance cannot be mutated.

    Parameters
    ----------
    tk_mean : float
        Mean of the Gaussian distribution used for threshold
        calculation.  Default: ``0.0``.
    tk_std_dev : float
        Standard deviation of the Gaussian distribution used for
        threshold calculation.  Must be positive.
        Default: ``1.0``.
    seed : int
        Seed for the random number generator.  Must be non-negative.
        Default: ``3439275``.
    """

    tk_mean: float = 0.0
    tk_std_dev: float = 1.0
    seed: int = 3439275

    def __post_init__(self) -> None:
        """Validate all fields after dataclass construction."""
        # -- tk_mean --
        if not isinstance(self.tk_mean, (int, float)):
            raise TypeError(
                f"GTSIMConfig: tk_mean must be a number, "
                f"got {type(self.tk_mean).__name__}"
            )
        import math

        if math.isnan(self.tk_mean) or math.isinf(self.tk_mean):
            raise ValueError(
                f"GTSIMConfig: tk_mean must be finite, got {self.tk_mean}"
            )

        # -- tk_std_dev --
        if not isinstance(self.tk_std_dev, (int, float)):
            raise TypeError(
                f"GTSIMConfig: tk_std_dev must be a number, "
                f"got {type(self.tk_std_dev).__name__}"
            )
        if math.isnan(self.tk_std_dev) or math.isinf(self.tk_std_dev):
            raise ValueError(
                f"GTSIMConfig: tk_std_dev must be finite, got {self.tk_std_dev}"
            )
        if self.tk_std_dev <= 0:
            raise ValueError(
                f"GTSIMConfig: tk_std_dev must be positive, got {self.tk_std_dev}"
            )

        # -- seed --
        if not isinstance(self.seed, (int, numpy.integer)) or isinstance(self.seed, bool):
            raise TypeError(
                f"GTSIMConfig: seed must be an int, got {type(self.seed).__name__}"
            )
        if self.seed < 0:
            raise ValueError(
                f"GTSIMConfig: seed must be non-negative, got {self.seed}"
            )
