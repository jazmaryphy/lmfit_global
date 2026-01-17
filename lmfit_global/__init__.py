"""A collection of codes & tools to read and process fitting"""

__version__ = "0.1"

__all__ = [
    "utils",
    "simplefit", 
    "LmfitGlobal", 
]

from . import utils
from .simplefit import simplefit
from .lmfit_global import LmfitGlobal