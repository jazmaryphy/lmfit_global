"""
Public API for lmfit_global utilities.

This module exposes the *stable, user-facing* symbols.
Internal helpers remain in their respective submodules.
"""
# ------------------------------
# Core dependency (LMFIT)
# ------------------------------
from ._deps import lmfit

# ------------------------------
# Core user-facing classes
# ------------------------------
from .fitdata import FitData
from .modelspec import ModelSpec
from .plotting import FitPlotter

# ------------------------------
# Logging / config
# ------------------------------
from ._config import LoggerLike, get_default_logger

# ------------------------------
# Parameter utilities (public)
# ------------------------------
from .parameters import normalize_parameter_specs

# ------------------------------
# Reporting utilities (public)
# ------------------------------
from .reporting import (
    lmfit_report,
    pretty_print_params,
    r_squared_safe,
)

# ------------------------------
# IO utilities (public)
# ------------------------------
from .io_utils import (
    parse_xrange,
    export_fit_to_dict,
    export_fit_to_json,
    export_fit_to_numpy,
)

__all__ = [
    "lmfit",
    "FitData",
    "ModelSpec",
    "FitPlotter",
    "LoggerLike",
    "get_default_logger",
    "normalize_parameter_specs",
    "lmfit_report",
    "pretty_print_params",
    "r_squared_safe",
    "parse_xrange",
    "export_fit_to_dict",
    "export_fit_to_json",
    "export_fit_to_numpy",
]