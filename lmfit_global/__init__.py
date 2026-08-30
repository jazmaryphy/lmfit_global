"""A collection of codes & tools to read and process fitting."""

__version__ = "0.1.0"
__author__ = "Muhammad Maikudi Isah"

# Import main classes and functions to expose them at top-level
from lmfit_global import utils
from lmfit_global.core import LmfitGlobal
from lmfit_global.simplefit import simplefit


__all__ = [
    "utils",
    "simplefit",
    "LmfitGlobal",
]