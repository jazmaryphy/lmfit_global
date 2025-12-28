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

def build_items(func, xrange=None, init_params=None, func_kws=None):
    # init_params = {} if init_params is None else init_params
    # func_kws = {} if func_kws is None else func_kws
    # params = {name: cfg.get("value", 0) for name, cfg in init_params.items()}
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

def check_fit(x, y, noise_scale=1.e-3, atol=0.1, rtol=0.05):

    y += np.random.normal(scale=noise_scale, size=len(y))
    return 

params = {
    "amplitude": 2.0,
    "center": 0.5,
    "sigma": 0.8
}
init_params = parameters.normalize_parameter_specs(params)
# print(init_params)

init_params = {
    'amplitude': {'value': 2.0, 'vary': True, 'min': -np.inf, 'max': +np.inf},
    'center':    {'value': 0.5, 'min': 0, 'max': 10},
    'sigma':     {'value': 0.8},
}


params = {name: cfg.get("value", 0) for name, cfg in init_params.items()}


x, y = get_data(lineshapes.gaussian, **params)