def test_import_lmfit_global():
    from lmfit_global import LmfitGlobal

def test_import_utils():
    from lmfit_global.util import parameters, plotting, reporting, utils

def test_utils_helper():
    from lmfit_global.util.utils import parse_xrange
    assert callable(parse_xrange)
