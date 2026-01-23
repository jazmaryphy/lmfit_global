"""
A collection of tools to read and process lmfit-global.

NOTE
----
- This file have multiple functions "messy-kitchen"
- The *stable public API* is defined in ``utils.api``.
- This module re-exports symbols for backward compatibility.
"""

__all__ = [
    # ============================
    # CORE-dependency
    # ============================
    "lmfit",

    # ============================
    # Top-level public classes
    # ============================
    "FitData",
    "ModelSpec",
    "LoggerLike",
    "FitPlotter",

    # ============================
    # Parameter utilities
    # ============================
    "finalize_parameter_specs",
    "normalize_parameter_specs",
    "_UNSET",
    "_ALLOWED_NUMERIC",
    "_LMFIT_INIT_PARAMETER_DEFAULTS",

    # ============================
    # Reporting utilities
    # ============================
    "wrap_expr",
    "build_expr",
    "pretty_expr",
    "lmfit_report",
    "r_squared_safe",
    "pretty_print_params",

    # ============================
    # IO utilities
    # ============================
    "parse_xrange",
    "export_ascii",
    "grid_and_eval",
    "export_fit_to_dict",
    "export_fit_to_json",
    "export_fit_to_numpy",
    "build_ascii_columns",
    "export_data_to_dataframe",
    "export_params_to_dataframe",

    # ============================
    # Config / logging
    # ============================
    "get_default_logger",

    # ============================
    # Submodules (optional)
    # ============================
    "builders",
    "io_utils",
    "plotting",
    "reporting",
    "parameters",
    "lineshapes",
    "api",
]

# ------------------------------
# Core dependency (LMFIT)
# ------------------------------
from ._deps import lmfit

# ------------------------------
# Re-export submodules
# ------------------------------
from . import (
    builders,
    io_utils,
    plotting,
    reporting,
    parameters,
    lineshapes,
    api,
)

# ------------------------------
# Core classes
# ------------------------------
from .fitdata import FitData
from .modelspec import ModelSpec
from ._config import LoggerLike, get_default_logger

# ------------------------------
# Parameter utilities
# ------------------------------
from .parameters import (
    _UNSET,
    _ALLOWED_NUMERIC,
    finalize_parameter_specs,
    normalize_parameter_specs,
    _LMFIT_INIT_PARAMETER_DEFAULTS,
)

# ------------------------------
# Reporting utilities
# ------------------------------
from .reporting import (
    wrap_expr,
    build_expr,
    pretty_expr,
    lmfit_report,
    r_squared_safe,
    pretty_print_params,
)

# ------------------------------
# IO utilities
# ------------------------------
from .io_utils import (
    parse_xrange,
    export_ascii,
    grid_and_eval,
    export_fit_to_dict,
    export_fit_to_json,
    export_fit_to_numpy,
    build_ascii_columns,
    export_data_to_dataframe,
    export_params_to_dataframe,
)

# ------------------------------
# Plotting utilities
# ------------------------------
from .plotting import FitPlotter


### DEPRECTATED WARNINGS
# from ._deprecation import deprecated
# from .reporting import pretty_expr as _pretty_expr

# def pretty_expr(*args, **kwargs):
#     deprecated("pretty_expr", "utils.api.pretty_expr")
#     return _pretty_expr(*args, **kwargs)


# from .parameters import normalize_parameter_specs as _norm

# def normalize_parameter_specs(*args, **kwargs):
#     deprecated(
#         "normalize_parameter_specs",
#         "utils.api.normalize_parameter_specs",
#     )
#     return _norm(*args, **kwargs)