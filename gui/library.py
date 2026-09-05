# %%
"""
library.py

Automatically inspects `lineshapes.py` to construct a complete FUNCTION_LIBRARY
compatible with lmfit_global while keeping the main Streamlit GUI clean.
"""
from __future__ import annotations

import inspect
import numpy as np
import lmfit_global.utils.lineshapes as ls

# Connector options for multi-component model building
CONNECTORS = ["+", "-", "*", "/"]

# %%
# Custom Metadata Overrides (for bounds, defaults, display names)
# Functions in lineshapes.py that take non-standard required arguments 
# (like `dipolar` array or `**coeffs`) or need specific min/max constraints.
PARAM_OVERRIDES = {
    "gaussian": {
        "amplitude": {"value": 1.0, "min": 0.0},
        "sigma": {"value": 1.0, "min": 1e-6},
    },
    "lorentzian": {
        "amplitude": {"value": 1.0, "min": 0.0},
        "sigma": {"value": 1.0, "min": 1e-6},
    },
    "exponential": {
        "decay": {"value": 1.0, "min": 1e-6},
    },
    "simplExpo": {
        "lam": {"value": 1.0, "min": 0.0},
    },
    "generExpo": {
        "lam": {"value": 1.0, "min": 0.0},
        "beta": {"value": 1.0, "min": 0.01, "max": 2.0},
    },
    "simpleGss": {
        "sigma": {"value": 1.0, "min": 1e-6},
    },
    "statGssKT": {
        "sigma": {"value": 1.0, "min": 1e-6},
    },
    "strKT": {
        "sigma": {"value": 1.0, "min": 1e-6},
        "beta": {"value": 1.0, "min": 0.01, "max": 2.0},
    },
    "dynGLKT_F_LF": {
        "sigma": {"value": 1.0, "min": 1e-6},
        "gamma": {"value": 1.0, "min": 1e-6},
    },
    "internFld": {
        "alpha": {"value": 1.0, "min": 0.0, "max": 1.0},
        "lam_T": {"value": 1.0, "min": 0.0},
        "lam_L": {"value": 1.0, "min": 0.0},
    },
    "step": {
        "sigma": {"value": 1.0},
        "form": {"value": "linear", "vary": False}, # Non-numeric, fixed
    },
    # "napro_sum_gaussian": {
    #     "dipolar": {"value": [1.0], "vary": False}, # Required parameter override
    # },
    "polynom": {
        "a0": {"value": 1.0},
        "a1": {"value": 0.0},
        "a2": {"value": 0.0},
    }
}

# Functions or helpers in lineshapes.py to exclude from the UI
EXCLUDED_FUNCTIONS = {
    "not_zero", 
    "napro_sum_gaussian",
}


# ----------------------------------------------------------------------
# 2. Dynamic Inspection & Library Builder
# ----------------------------------------------------------------------
def build_function_library():
    """
    Inspects all callable functions inside `lineshapes.py` and returns 
    a dictionary structured for the Streamlit GUI.
    """
    library = {}

    for name, obj in inspect.getmembers(ls, inspect.isfunction):
        # Skip excluded utility functions or imports
        if name in EXCLUDED_FUNCTIONS or name.startswith("_"):
            continue

        sig = inspect.signature(obj)
        params_dict = {}

        # Parse signature parameters
        param_list = list(sig.parameters.values())
        
        # Skip the independent variable (1st parameter: 'x' or 't')
        eval_params = param_list[1:] if len(param_list) > 0 else []

        for p in eval_params:
            # Handle **kwargs (like **coeffs in polynom)
            if p.kind == inspect.Parameter.VAR_KEYWORD:
                continue

            # Get standard default value if provided, else set 1.0
            default_val = p.default if p.default is not inspect.Parameter.empty else 1.0
            
            # Non-numeric default fallback
            if not isinstance(default_val, (int, float)):
                params_dict[p.name] = {"value": default_val, "vary": False}
            else:
                params_dict[p.name] = {"value": float(default_val)}

        # Apply specific overrides if defined
        if name in PARAM_OVERRIDES:
            for pname, overrides in PARAM_OVERRIDES[name].items():
                if pname not in params_dict:
                    params_dict[pname] = {}
                params_dict[pname].update(overrides)

        # Pretty display label (e.g. dynGLKT_F_LF -> Dyn GLKT F LF)
        display_label = name.replace("_", " ").title()

        library[display_label] = {
            "func": obj,
            "func_name": name,
            "params": params_dict,
            "doc": obj.__doc__ or ""
        }

    return library


# Initialize the registry library for direct import
FUNCTION_LIBRARY = build_function_library()