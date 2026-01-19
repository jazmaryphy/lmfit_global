"""A collection of tools to read and process lmfit-global."""

__all__ = [
    # top-level classes
    "FitData",
    "ModelSpec",
    "LoggerLike",
    "FitPlotter",

    # parameter utilities
    "normalize_parameter_specs",
    "_UNSET",
    "_ALLOWED_NUMERIC",
    "_LMFIT_INIT_PARAMETER_DEFAULTS",

    # reporting utilities
    "wrap_expr",
    "build_expr",
    "pretty_expr",
    "lmfit_report",
    "r_squared_safe",
    "pretty_print_params",

    # io utilities
    "parse_xrange",
    "export_ascii",
    "grid_and_eval",
    "build_ascii_columns",

    # config
    "get_default_logger",

    # submodules (optional but often useful)
    "builders",
    "io_utils",
    "plotting",
    "reporting",
    "parameters",
    "lineshapes",
]

# --- Re-export submodules ---
from . import (
    builders,
    io_utils,
    plotting,
    reporting,
    parameters,
    lineshapes,
)

# --- Re-export top-level classes ---
from .fitdata import FitData
from .modelspec import ModelSpec
from ._config import LoggerLike, get_default_logger

# --- Re-export parameter utilities ---
from .parameters import (
    _UNSET,
    _ALLOWED_NUMERIC,
    normalize_parameter_specs,
    _LMFIT_INIT_PARAMETER_DEFAULTS,
)

# --- Re-export reporting utilities ---
from .reporting import (
    wrap_expr,
    build_expr,
    pretty_expr,
    lmfit_report,
    r_squared_safe,
    pretty_print_params,
)

# --- Re-export IO utilities ---
from .io_utils import (
    parse_xrange,
    export_ascii,
    grid_and_eval,
    build_ascii_columns,
)

# --- Re-export plotting utilities ---
from .plotting import FitPlotter