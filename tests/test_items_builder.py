# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./"))
# sys.path.insert(0, ROOT)
# # print(sys.path)

import pytest
import numpy as np
from lmfit_global import LmfitGlobal
from lmfit_global.utils import lineshapes, parameters
from lmfit_global.utils.builders import GlobalFitBuilder

def _isclose(name, expected_value, fit_value, atol, rtol):
    """isclose with error message"""
    assert np.isclose(expected_value, fit_value, atol=atol, rtol=rtol), \
           f"bad value for {name}: expected {expected_value}, got {fit_value}."

def get_data(func, **params):
    if not callable(func):
        raise TypeError("func must be callable")
    # np.random.seed(2021)
    try:
        x = np.linspace(-5, 10, 501)
        y = func(x, **params)
    except TypeError as exc:
        raise TypeError(f"Parameter mismatch for {func.__name__}: {exc}")

    return x, y

def get_init_params(params):
    pardict = parameters.normalize_parameter_specs(params)

    init_params = {}
    for name, spec in pardict.items():
        final = {}
        for key, default in parameters._LMFIT_INIT_PARAMETER_DEFAULTS.items():
            val = spec.get(key, parameters._UNSET)
            final[key] = default if val is parameters._UNSET else val
        # print(final)
        init_params[name] = final

    return init_params

def build_items(func, params, xrange=None, func_kws=None):
    """
    """
    func_kws = func_kws or {}
    # x = np.linspace(-5, 10, 501)
    x, y = get_data(func, **params)

    builder = (
        GlobalFitBuilder()
        .set_data(x, y, xrange=xrange)                      # x and all y datasets
        .add_model(func, get_init_params(params), func_kws=func_kws)       
        # .connect("+")                             # how to combine multiple functions
    )
    # Above: you can test different form of step functions

    return builder.build()


def check_fit(func, test_params, noise_scale=1.e-3, atol=0.1, rtol=0.05, func_kws=None):
    """Checks that a model fits noisy data well

    Args:
        func (callable):  model function to use
        test_params: dict of 'true values'
        noise_scale (float, optional): The standard deviation of noise that is added to the test data.
        atol (float, optional): Absolute tolerance for considering fit parameters close to the
            parameters test data was generated with.
        rtol (float, optional): Relative tolerance for considering fit parameters close to the
            parameters test data was generated with.
        func_kws (dict, optional): Additional keyword arguments to pass to model function. 
            Defaults to None.

    Returns:
        fit result

    Raises:
        AssertionError: Any fit parameter that is not close to the parameter used to
            generate the test data raises this error.
    """
    items = build_items(
        func=func, 
        params=test_params, 
        xrange=None,             # use default data range
        func_kws=func_kws
    )
    xy = items['data']['xy']      # extract x, y
    x, y = xy[:, 0], xy[:, 1]
    y += np.random.normal(scale=noise_scale, size=len(y))        # add noise
    items['data']['xy'] = np.column_stack([x, y])
    lg = LmfitGlobal(items)
    lg.fit()
    fit_values = lg.best_values
    for name, test_val in test_params.items():
        # print(name, test_val, fit_values[name+'_0'])
        _isclose(name, test_val, fit_values[name+'_0'], atol, rtol)   # underscore is dataset index
    return lg.result



def testLinear():
    check_fit(lineshapes.linear, dict(intercept=2, slope=2))


def testGaussian():
    check_fit(lineshapes.gaussian, dict(amplitude=8, center=4, sigma=1))


def testSine():
    check_fit(lineshapes.sine, dict(amplitude=1, frequency=5, shift=0))


def testExpsine():
    check_fit(lineshapes.expcosine, dict(amplitude=1, frequency=5, shift=0, decay=0.5))