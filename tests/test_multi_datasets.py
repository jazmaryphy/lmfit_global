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

def build_items(func, params_list, xrange=None, func_kws=None):
    """
    """
    x = np.linspace(-5, 10, 501)
    y_list = []

    for params in params_list:
        y = func(x, **params)
        y_list.append(y)

    xy = np.column_stack([x] + y_list)

    data_dict = {
        "xy": xy,
        "xrange": xrange,
    }

    func_lst = [{
        "func_name": func,
        "init_params": get_init_params(params_list[0]),
        "func_kws": func_kws or {},
    }]

    function_dict = {
        "theory": func_lst,
        "theory_connectors": None,
    }

    return {
        "data": data_dict,
        "functions": function_dict,
    }


def check_fit(
    func,
    test_params_list,
    noise_scale=1e-3,
    atol=0.1,
    rtol=0.05,
    func_kws=None,
):
    """
    """
    items = build_items(
        func=func,
        params_list=test_params_list,
        func_kws=func_kws,
    )

    xy = items["data"]["xy"]
    x = xy[:, 0]
    y = xy[:, 1:]

    # Add noise independently per dataset
    y += np.random.normal(scale=noise_scale, size=y.shape)
    items["data"]["xy"] = np.column_stack([x, *y.T])

    lg = LmfitGlobal(items)
    lg.fit()

    fit_values = lg.best_values

    for j, params in enumerate(test_params_list):
        for name, expected in params.items():
            fitted = fit_values[f"{name}_{j}"]
            _isclose(f"{name}_{j}", expected, fitted, atol, rtol)

    return lg.result


def testLinear():
    def random_params(N):
        return [
            {
                "intercept": np.random.uniform(0.6, 10.1),
                "slope":     np.random.uniform(-0.2, 1.0),
            }
            for _ in range(N)
        ]
    check_fit(lineshapes.linear, random_params(N=5))


def testGaussian():
    def random_params(N):
        return [
            {
                "amplitude": np.random.uniform(0.6, 10.1),
                "center":    np.random.uniform(-0.2, 1.0),
                "sigma":     np.random.uniform(0.25, 0.28),
            }
            for _ in range(N)
        ]
    check_fit(lineshapes.gaussian, random_params(N=5), noise_scale=0.1)



def testParabolic():
    def random_params(N):
        return [
            {
                "a": np.random.uniform(-1.0, 1.0),
                "b": np.random.uniform(-5.0, 5.0),
                "c": np.random.uniform(0.0, 10.0),
            }
            for _ in range(N)
        ]
    check_fit(lineshapes.parabolic, random_params(N=5), noise_scale=0.1)


def testStaticGssKT():
    def random_params(N):
        return [
            {
                "sigma":     np.random.uniform(0.1, 2.0),
            }
            for _ in range(N)
        ]
    check_fit(lineshapes.gaussian, random_params(N=5), noise_scale=0.1)