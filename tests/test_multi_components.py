# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./"))
# sys.path.insert(0, ROOT)
# # print(sys.path)

import pytest
import numpy as np
from lmfit_global import LmfitGlobal
from lmfit_global.util import lineshapes, parameters
from lmfit_global.util.utils import GlobalFitBuilder




# def testStepmodel_erf():
#     check_fit(lineshapes.step, dict(amplitude=1, center=3, sigma=1.5), func_kws={'form':'erf'})


# def testStepmodel_linear():
#     check_fit(lineshapes.step, dict(amplitude=1, center=3, sigma=1.5), func_kws={'form':'linear'})