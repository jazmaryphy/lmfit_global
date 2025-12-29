# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./"))
# sys.path.insert(0, ROOT)
# # print(sys.path)

import pytest

def test_import_lmfit_global():
    from lmfit_global import LmfitGlobal

def test_import_utils():
    from lmfit_global.util import lineshapes, parameters, plotting, reporting, utils, io

def test_utils_helper():
    from lmfit_global.util.utils import parse_xrange, GlobalFitBuilder
    assert callable(parse_xrange)
