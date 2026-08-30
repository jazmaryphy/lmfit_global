import pytest


def test_lmfit_import():
    from lmfit_global.utils import lmfit
    assert hasattr(lmfit, "Parameters")


def test_import_lmfit_global():
    from lmfit_global import LmfitGlobal


def test_import_utils():
    from lmfit_global.utils import (
        lineshapes, 
        parameters, 
        plotting, 
        reporting, 
        modelspec,
        builders, 
        io_utils,
        fitdata,
    )
    

def test_utils_helper():
    from lmfit_global.utils.io_utils import parse_xrange
    from lmfit_global.utils.builders import GlobalFitBuilder
    assert callable(parse_xrange)