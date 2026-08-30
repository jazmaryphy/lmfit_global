import pytest
import numpy as np
from lmfit_global import LmfitGlobal
from lmfit_global.utils import lineshapes


@pytest.fixture
def x():
    return np.linspace(-5, 5, 200)


@pytest.fixture
def gaussian():
    def _gauss(x, amp=1.0, cen=0.0, wid=1.0):
        return amp * np.exp(-(x - cen)**2 / (2 * wid**2))
    return _gauss


@pytest.fixture
def step():
    def _step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
        arg = (x - center) / sigma
        if form == 'linear':
            out = np.minimum(1, np.maximum(0, arg + 0.5))
        elif form == 'erf':
            from scipy.special import erf
            out = 0.5 * (1 + erf(arg))
        else:
            raise ValueError("invalid form")
        return amplitude * out
    return _step


def lg_simple():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x, amplitude=3, center=1, sigma=0.8)

    items = {
        "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "amplitude": {"value": 2.0},
                        "center": {"value": 0.0},
                        "sigma": {"value": 1.0},
                    },
                }
            ]
        },
    }

    return LmfitGlobal(items, log_level="info")


def test_lmfit_import():
    from lmfit_global.utils._deps import lmfit 
    assert hasattr(lmfit, "Parameters")


def test_parameters_explicitly_normalized():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x, amplitude=3, center=0, sigma=1)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "amplitude": {"value": 2.5, "min": 0},
                        "center": 0.0,
                        "sigma": {"value": 1.2, "min": 0},
                    },
                }
            ]
        },
    }

    lg = LmfitGlobal(items, log_level="warning")
    lg.set_data(x, y)

    params = lg.model_specs[0].init_params

    assert set(params) == {"amplitude", "center", "sigma"}
    assert params["amplitude"]["value"] == 2.5
    assert params["amplitude"]["min"] == 0
    assert params["amplitude"]["max"] == +np.inf
    assert params["center"]["min"] == -np.inf
    assert params["sigma"]["vary"] is True



def test_multicomponent_requires_connectors():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x) + lineshapes.step(x)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {},
                },
                {
                    "func_name": lineshapes.step,
                    "init_params": {},
                    "func_kws": {"form": "erf"},
                },
            ],
        },
    }


    with pytest.raises(ValueError, match="theory_connectors"):
        LmfitGlobal(items, log_level="INFO")



def test_parameters_autogenerate_normalized():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x, amplitude=3, center=0, sigma=1)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {},
                }
            ]
        },
    }

    lg = LmfitGlobal(items, log_level="INFO")
    lg.set_data(x, y)

    params = lg.model_specs[0].init_params

    assert set(params) == {"amplitude", "center", "sigma"}
    for p in params.values():
        assert "value" in p
        assert "vary" in p



def test_parameters_fixedarguments_func_kws():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x) + lineshapes.step(x, form="erf")

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "amplitude": dict(value=1.0, min=0),
                        "center": dict(value=0.0),
                        "sigma": dict(value=1.0),                    
                    },
                },
                {
                    "func_name": lineshapes.step,
                    "init_params": {
                        "amplitude": dict(value=0.1),
                        "center": dict(value=0.3),   
                        "sigma": dict(value=0.4, min=0),                 
                    },
                    "func_kws": {"form": "erf"},
                },
            ],
            "theory_connectors": ["+"],
        },
    }


    lg = LmfitGlobal(items, log_level="INFO")
    lg.set_data(x, y)

    spec = lg.model_specs[1]

    assert "form" not in spec.init_params
    assert spec.func_kws["form"] == "erf"



def test_parameters_invalid_initial_parameter():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x, amplitude=3, center=0, sigma=1)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "invalid_param": 1.0,
                    },
                }
            ]
        },
    }

    with pytest.raises(ValueError, match="unexpected `init_params`"):
        LmfitGlobal(items, log_level="WARNING")



def test_parameters_invalid_initial_parameter_funckws():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.step(x)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": step,
                    "init_params": {
                        "form": "erf",  #  wrong
                    },
                }
            ]
        },
    }

    with pytest.raises(ValueError):
        LmfitGlobal(items, log_level="CRITICAL")



def test_update_par():
    lg = lg_simple()

    pardict = {"amplitude_0": dict(value=5.0, min=0)}
    lg.update_par(pardict)

    p = lg.init_params["amplitude_0"]

    assert p.value == 5.0
    assert p.min == 0


def test_update_params():
    lg = lg_simple()

    # use update_par() or with
    lg.update_params({
        "amplitude_0": {"value": 4.0},
        "center_0": {"vary": False},
    })

    assert lg.init_params["amplitude_0"].value == 4.0
    assert lg.init_params["center_0"].vary is False


def test_add_params():
    lg = lg_simple()

    # use add_par() or with
    lg.add_params({
        "background": {"value": 0.1, "vary": False}
    })

    assert "background" in lg.init_params
    assert lg.init_params["background"].value == 0.1
    assert lg.init_params["background"].vary is False


def test_add_params_existing_raises():
    lg = lg_simple()

    lg.add_params({"amplitude_0": {"value": 10}})


def test_remove_params():
    lg = lg_simple()

    lg.remove_params(["center_0"])

    assert "center_0" not in lg.init_params


def test_remove_missing_param_raises():
    lg = lg_simple()

    lg.remove_params(["not_a_param"], force=True)


def test_set_value():
    lg = lg_simple()
    lg.set_value({"amplitude_0": 9.0})
    assert lg.init_params["amplitude_0"].value == 9.0


def test_set_min():
    lg = lg_simple()
    lg.set_min({"amplitude_0": 1.0})
    assert lg.init_params["amplitude_0"].min == 1.0


def test_set_max():
    lg = lg_simple()
    lg.set_max({"amplitude_0": 10.0})
    assert lg.init_params["amplitude_0"].max == 10.0


def test_set_vary():
    lg = lg_simple()
    lg.set_vary({"amplitude_0": False})
    assert lg.init_params["amplitude_0"].vary is False


def test_set_global_params():
    lg = lg_simple()
    parlist = ["center_0", "sigma_0"]
    # or with set_global()
    lg.set_global_params(parlist, overwrite_expr=True)

    assert lg.init_params["sigma_0"].vary is False


def test_set_global_params_multicomp():
    x = np.linspace(-5, 5, 200)
    y = lineshapes.gaussian(x, 5.0, 0, 2.0) + lineshapes.gaussian(x, 1.0, 0, 2.0)

    items = {
        # "data": {"xy": np.column_stack([x, y])},
        "functions": {
            "theory": [
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "amplitude": dict(value=5.0, min=0),
                        "center": dict(value=0.0),
                        "sigma": dict(value=2.0),                    
                    },
                },
                {
                    "func_name": lineshapes.gaussian,
                    "init_params": {
                        "amplitude": dict(value=1.0),
                        "center": dict(value=0.0),   
                        "sigma": dict(value=2.0),                 
                    },
                },
            ],
            "theory_connectors": ["+"],
        },
    }


    lg = LmfitGlobal(items)
    lg.set_data(x, y)

    parlist = ["c0_center_0", "c1_center_0"]
    # or with set_global()
    lg.set_global_params(parlist, overwrite_expr=True)

    parlist = ["c0_sigma_0", "c1_sigma_0"]
    # or with set_global()
    lg.set_global_params(parlist, overwrite_expr=True)

    assert lg.init_params["c1_center_0"].vary is False
    assert lg.init_params["c1_sigma_0"].vary is False

    assert lg.init_params["c0_center_0"].value ==  lg.init_params["c1_center_0"].value
    assert lg.init_params["c0_sigma_0"].value ==  lg.init_params["c1_sigma_0"].value