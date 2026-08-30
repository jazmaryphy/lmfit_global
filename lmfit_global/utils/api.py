# %%
"""
Public API for lmfit_global utilities.

This module exposes the *stable, user-facing* symbols.
Internal helpers remain in their respective submodules.
"""

# Core dependency (LMFIT)
from lmfit_global.utils._deps import lmfit

# Core user-facing classes
from lmfit_global.utils.fitdata import FitData
from lmfit_global.utils.modelspec import ModelSpec
from lmfit_global.utils.plotting import FitPlotter

# Logging / config
from lmfit_global.utils._config import LoggerLike, get_default_logger

# Parameter utilities (public)
from lmfit_global.utils.parameters import finalize_parameter_specs, normalize_parameter_specs

# Reporting utilities (public)
from lmfit_global.utils.reporting import (
    lmfit_report,
    pretty_print_params,
    r_squared_safe,
)

# IO utilities (public)
from lmfit_global.utils.io_utils import (
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
    "finalize_parameter_specs",
    "normalize_parameter_specs",
    "lmfit_report",
    "pretty_print_params",
    "r_squared_safe",
    "parse_xrange",
    "export_fit_to_dict",
    "export_fit_to_json",
    "export_fit_to_numpy",
]