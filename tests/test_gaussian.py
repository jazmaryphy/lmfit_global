# import os, sys
# ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "./"))
# sys.path.insert(0, ROOT)
# # print(sys.path)

import pytest
import numpy as np
from lmfit_global import LmfitGlobal
from lmfit_global.util import lineshapes, parameters

def _isclose(name, expected_value, fit_value, atol, rtol):
    """isclose with error message"""
    assert np.isclose(expected_value, fit_value, atol=atol, rtol=rtol), \
           f"bad value for {name}: expected {expected_value}, got {fit_value}."

def get_data(func, **params):
    if not callable(func):
        raise TypeError("func must be callable")
    # np.random.seed(2021)
    try:
        x = np.linspace(0, 10, 201)
        y = func(x, **params)
    except TypeError as exc:
        raise TypeError(f"Parameter mismatch for {func.__name__}: {exc}")

    return x, y

def build_items(func, params, xrange=None, func_kws=None):
    """build LmfitGlobal(items=items) items"""
    x, y = get_data(func, **params)
    init_params = parameters.normalize_parameter_specs(params)
    data_dict = {
        'xy': np.column_stack([x, y]),  
        'xrange': xrange  
        }
    func_lst = [
        {
            'func_name': func,
            'init_params' : init_params,
            'func_kws': func_kws or {}
 
        },
    ]
    function_dict = {
        'theory': func_lst,
        'theory_connectors': None,
    }
    # items 
    items = {
        'data': data_dict,             
        'functions': function_dict,    
    }
    return items

def check_fit(func, test_params, noise_scale=1.e-3, atol=0.1, rtol=0.05):
    """Checks that a model fits noisy data well

    Args:
        func (callable):  model function to use
        test_params: dict of 'true values'
        noise_scale (float, optional): The standard deviation of noise that is added to the test data.
        atol (float, optional): Absolute tolerance for considering fit parameters close to the
            parameters test data was generated with.
        rtol (float, optional): Relative tolerance for considering fit parameters close to the
            parameters test data was generated with.

        Returns:
        fit result

        Raises:
        AssertionError: Any fit parameter that is not close to the parameter used to
            generate the test data raises this error.
    """
    items = build_items(
        func=func, 
        params=test_params, 
        xrange=None, 
        func_kws=None
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


def testGaussian():
    check_fit(lineshapes.gaussian, dict(amplitude=8, center=4, sigma=1))