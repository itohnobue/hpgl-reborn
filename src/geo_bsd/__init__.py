# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2009, HPGL Team
"""
HPGL Reborn — High Performance Geostatistics Library (Python bindings).

This package provides a Python interface to the HPGL C++ geostatistics
engine.  All public names are listed below and accessible as
``geo_bsd.<name>``.

**Kriging**: ordinary_kriging, simple_kriging, lvm_kriging,
indicator_kriging, median_ik, simple_cokriging_markI,
simple_cokriging_markII, simple_kriging_weights

**Simulation**: sgs_simulation, sis_simulation

**CDF**: calc_cdf

**Data classes**: ContProperty, IndProperty, CovarianceModel,
SugarboxGrid, CdfData, covariance

**I/O**: load_cont_property, load_ind_property, read_inc_file_float,
read_inc_file_byte, write_property, write_gslib_property,
get_gslib_property

**Utilities**: calc_mean, set_thread_num, get_thread_num,
set_output_handler, set_progress_handler

**Submodules** (imported for side-effects; use their public API):
variogram, cvariogram, routines, validation

**Diagnostics**: get_kriging_stats (retrieve C++ kriging stats after a call)
"""

# Import validation module for user convenience
import logging

from . import routines, validation, variogram

try:
    from . import cvariogram
except (ImportError, OSError) as e:
    logging.getLogger(__name__).warning(
        "cvariogram C++ extension not available; variogram C++ functions "
        "will not work. Build with HPGL_BUILD_VARIOGRAM=ON to enable. "
        f"Error: {e}"
    )
    cvariogram = None  # type: ignore[assignment]
from .cdf import *
from .ffi_adapter import get_kriging_stats
from .geo import *
from .sgs import sgs_simulation
from .sis import sis_simulation

try:
    from importlib.metadata import PackageNotFoundError  # noqa: E402
    from importlib.metadata import version as _get_version
except ImportError:  # pragma: no cover - pre-3.8 fallback
    __version__ = "2.0.2"
else:
    # M-23: the hardcoded fallback must apply ONLY when the installed
    # package metadata cannot be found — never unconditionally. Pre-fix,
    # the fallback assignment was dedented out of the except block
    # (a23a079), so `__version__` was always overwritten with the hardcoded
    # value and a version bump in pyproject.toml never surfaced. Keep the
    # fallback value consistent with pyproject.toml's `version`.
    try:
        __version__ = _get_version("hpgl")
    except PackageNotFoundError:
        logging.warning(
            "hpgl package not found via importlib.metadata; falling back to default version 2.0.2"
        )
        __version__ = "2.0.2"

__all__ = [
    # Kriging algorithms
    "ordinary_kriging",
    "simple_kriging",
    "lvm_kriging",
    "indicator_kriging",
    "median_ik",
    "simple_cokriging_markI",
    "simple_cokriging_markII",
    "simple_kriging_weights",
    # Simulation algorithms
    "sgs_simulation",
    "sis_simulation",
    # Diagnostics
    "get_kriging_stats",
    # CDF
    "calc_cdf",
    # Data classes
    "ContProperty",
    "IndProperty",
    "CovarianceModel",
    "SugarboxGrid",
    "covariance",
    "CdfData",
    # Variogram
    "variogram",
    # Routines
    "routines",
    # C-variogram
    "cvariogram",
    # Validation
    "validation",
    # IO
    "load_cont_property",
    "load_ind_property",
    "read_inc_file_float",
    "read_inc_file_byte",
    "write_property",
    "write_gslib_property",
    "calc_mean",
    "set_thread_num",
    "get_thread_num",
    "set_output_handler",
    "set_progress_handler",
    "get_gslib_property",
]
