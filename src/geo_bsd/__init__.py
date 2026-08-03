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

**Simulation**: sgs_simulation, sis_simulation, gtsim_2ind

**CDF**: calc_cdf

**Data classes**: ContProperty, IndProperty, CovarianceModel,
SugarboxGrid, CdfData, covariance, SGSConfig, SISConfig, GTSIMConfig

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
import re
from pathlib import Path

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

# II-30: the frozen-config dataclasses are documented public API (config.py)
# and must be reachable at the package top level.
from .config import GTSIMConfig, SGSConfig, SISConfig
from .ffi_adapter import get_kriging_stats
from .geo import *

# II-29: gtsim_2ind is a first-class simulation entry point; export it at
# the top level like its siblings sgs_simulation / sis_simulation.
from .gtsim import gtsim_2ind
from .sgs import sgs_simulation
from .sis import sis_simulation


def _source_version():
    """Return the package version from the source pyproject.toml, or None.

    The package is versioned by pyproject.toml's ``[project] version``. A
    stale *installed* dist-info can disagree with the source tree (e.g. an
    old hpgl-1.6.0.dist-info shadowing the venv), and importlib.metadata
    would then report the wrong version for the code actually imported.
    When the source tree is reachable, its version is authoritative (II-28);
    installed-metadata resolution applies only when it is not (e.g. an
    installed wheel without the source tree).

    Python 3.9/3.10 fallback: ``tomllib`` is stdlib from 3.11; without
    ``tomli`` installed the version is read from the ``[project]`` table
    with a small regex walk (mirrors how the build parses it).
    """
    pyproject = Path(__file__).resolve().parent.parent.parent / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            tomllib = None
    if tomllib is not None:
        try:
            with pyproject.open("rb") as fh:
                data = tomllib.load(fh)
        except (OSError, ValueError, TypeError):
            data = {}
        version = data.get("project", {}).get("version")
        if isinstance(version, str) and version:
            return version
    try:
        lines = pyproject.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    in_project = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if in_project:
            m = re.match(r'version\s*=\s*"([^"]+)"', stripped)
            if m:
                return m.group(1)
    return None


_source_version_value = _source_version()

try:
    from importlib.metadata import PackageNotFoundError  # noqa: E402
    from importlib.metadata import version as _get_version
except ImportError:  # pragma: no cover - pre-3.8 fallback
    __version__ = _source_version_value or "2.0.3"
else:
    if _source_version_value is not None:
        # II-28: the source tree is authoritative when present — a stale
        # installed dist-info must not make __version__ disagree with the
        # code actually imported (pre-fix: __init__.py:66-72 read
        # importlib.metadata unconditionally, reporting 1.6.0 against
        # 2.0.3 source/wheel). PRIOR_FIX_ATTEMPT bde1e24 narrowed the
        # except clause only and could not fix stale metadata.
        __version__ = _source_version_value
    else:
        try:
            __version__ = _get_version("hpgl")
        except PackageNotFoundError:
            logging.warning(
                "hpgl package not found via importlib.metadata; falling back to default version 2.0.3"
            )
            __version__ = "2.0.3"

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
    "gtsim_2ind",
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
    "SGSConfig",
    "SISConfig",
    "GTSIMConfig",
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
