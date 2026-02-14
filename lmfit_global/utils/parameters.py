# %%
import sys
import inspect
import numbers
import numpy as np
from collections import namedtuple
from typing import Union, Iterable, Callable, Dict, Any

# %%
# ----------------------------------------
# CORE package (LMFIT), MUST be installed
# ----------------------------------------
from ._deps import lmfit

# %%
_UNSET = object()

_ALLOWED_LMFIT_PARAMETER_HINT_KEYS = ('value', 'vary', 'min', 'max', 'expr')

_LMFIT_INIT_PARAMETER_DEFAULTS = {
            'value': -np.inf, 'vary': True,
            'min': -np.inf, 'max': +np.inf,
            'expr': None, 
            # 'brute_step': None  # NO LMFIT DEFAULT PARAMETER HINTS, TO BE HANDLED IN LMFIT GLOBAL
        }

_ALLOWED_NUMERIC = (int, float)
PY314_PLUS = sys.version_info >= (3, 14)

# %%
if PY314_PLUS:
    import annotationlib

    def wrapped_inspect_signature(obj):
        """Return inspect.Signature with legacy-compatible annotations."""
        return inspect.signature(
            obj,
            annotation_format=annotationlib.Format.FORWARDREF,
        )
else:
    wrapped_inspect_signature = inspect.signature


# ---------------------------------------------------------
# getfullargspec replacement
# ---------------------------------------------------------

FullArgSpec = namedtuple(
    "FullArgSpec",
    [
        "args",
        "varargs",
        "varkw",
        "defaults",
        "kwonlyargs",
        "kwonlydefaults",
        "annotations",
    ],
)


def getfullargspec_no_self(func) -> FullArgSpec:
    """
    Modern replacement for inspect.getfullargspec using inspect.signature.

    Removes 'self' for bound methods for consistency across Python versions.
    """

    sig = wrapped_inspect_signature(func)
    params = list(sig.parameters.values())

    # Positional args
    args = [
        p.name
        for p in params
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    # Remove self if bound method
    if inspect.ismethod(func) and args:
        args = args[1:]

    # *args
    varargs = next(
        (p.name for p in params if p.kind == inspect.Parameter.VAR_POSITIONAL),
        None,
    )

    # **kwargs
    varkw = next(
        (p.name for p in params if p.kind == inspect.Parameter.VAR_KEYWORD),
        None,
    )

    # Defaults (correct alignment with args)
    pos_params = [
        p for p in params
        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]

    default_vals = [p.default for p in pos_params if p.default is not p.empty]
    defaults = tuple(default_vals) if default_vals else None

    # Keyword-only args
    kwonlyargs = [
        p.name for p in params
        if p.kind == inspect.Parameter.KEYWORD_ONLY
    ]

    kwonlydefaults = {
        p.name: p.default
        for p in params
        if p.kind == inspect.Parameter.KEYWORD_ONLY
        and p.default is not p.empty
    } or None

    annotations = {
        p.name: p.annotation
        for p in params
        if p.annotation is not p.empty
    }

    return FullArgSpec(
        args,
        varargs,
        varkw,
        defaults,
        kwonlyargs,
        kwonlydefaults,
        annotations,
    )

# %%
def normalize_parameter_specs(
    *parlist: Union[str, lmfit.Parameter, lmfit.Parameters, Iterable, Dict]
) -> Dict[str, Dict[str, Any]]:
    """
    Normalize heterogeneous parameter specifications into a canonical dictionary.


    This utility accepts a flexible mix of parameter specifications
    (strings, lmfit.Parameter objects, lmfit.Parameters containers,
    lists/tuples, and dictionaries) and converts them into a unified form:

        Dict[str, Dict[str, Any]]

    where each key is a parameter name and each value is a dictionary
    of lmfit-compatible parameter attributes.

    Supported input forms
    ---------------------
    1. String (parameter name):
        - Creates a default parameter specification.

    2. lmfit.Parameter:
        - Extracts value, bounds, vary flag, expression, and brute_step.

    3. lmfit.Parameters:
        - Iterates over contained Parameter objects.

    4. Dictionary:
        - Full specification:
            {"sigma_0": {"value": 1.0, "min": 0}}
        - Shorthand value-only:
            {"sigma_1": 2.0}

    5. List or tuple:
        - May contain any mix of the above.
        - Nested lists/tuples are flattened.
        - Empty containers are not allowed.

    Notes
    -----
    - Later definitions override earlier ones (last-wins behavior).
    - Missing fields are filled with `_UNSET` placeholders.
    - Defaults are applied only when creating parameters.
    - Unsupported types raise TypeError.
    - Empty lists/tuples raise ValueError.
    - Invalid lmfit parameter hint keys raise KeyError.

    Returns:
        Dict[str, Dict[str, Any]]
            Canonical parameter specification dictionary.

    Examples
    --------
    Basic string input
    >>> normalize_parameter_specs("sigma_0")
    {'sigma_0': {'value': -inf, 'vary': True, 'min': -inf,
                 'max': inf, 'expr': None}}

    Shorthand dictionary (value-only)
    >>> normalize_parameter_specs({"sigma_1": 2.0})
    {'sigma_1': {'value': 2.0, 'vary': True, 'min': -inf,
                 'max': inf, 'expr': None}}

    Full dictionary specification
    >>> normalize_parameter_specs({"sigma_2": {"value": 1.0, "min": 0}})
    {'sigma_2': {'value': 1.0, 'vary': True, 'min': 0,
                 'max': inf, 'expr': None}}

    lmfit.Parameter input
    >>> p = lmfit.Parameter("sigma_3", value=3.0, vary=False)
    >>> normalize_parameter_specs(p)
    {'sigma_3': {'value': 3.0, 'vary': False, 'min': None,
                 'max': None, 'expr': None}}

    lmfit.Parameters input
    >>> params = lmfit.Parameters()
    >>> params.add("a", value=1.0, vary=True, min=0, max=10)
    >>> params.add("b", value=2.0, vary=False)
    >>> normalize_parameter_specs(params)
    {'a': {'value': 1.0, 'vary': True, 'min': 0, 'max': 10, 'expr': None}, 
     'b': {'value': 2.0, 'vary': False, 'min': -inf, 'max': inf, 'expr': None}}
    
    tuple input
    >>> par = ('sig', 10, True, None, None, None, None)
    >>> normalize_parameter_specs(par)
    {'sig': {'value': 10, 'vary': True, 'min': -inf, 'max': inf, 'expr': None}}


    Mixed and nested inputs
    >>> params = lmfit.Parameters()
    >>> params.add("a", value=1.0, vary=True, min=0, max=10)
    >>> params.add("b", value=2.0, vary=False)
    >>> normalize_parameter_specs(
    ...     "x",
    ...     {"y": 2.0},
    ...     [lmfit.Parameter("z", value=3.0), "c"],
    ...     params,
    ...     ('f', 10, True, None, None, None, None),
    ... )
    {
      'x': {...},
      'y': {...},
      'z': {...},
      'c': {...},
      'a': {...},
      'b': {...},
      'f': {...}
    }

    Override behavior (last wins)
    >>> normalize_parameter_specs("x", {"x": {"value": 5}})
    {'x': {'value': 5, 'vary': True, 'min': -inf,
           'max': inf, 'expr': None}}
    """

    out: Dict[str, Dict[str, Any]] = {}

    def _empty_spec():
        return {
            'value': _UNSET,
            'vary': _UNSET,
            'min': _UNSET,
            'max': _UNSET,
            'expr': _UNSET,
            # 'brute_step': _UNSET, # NO LMFIT DEFAULT PARAMETER HINTS
        }

    def _add_param(par):
        if isinstance(par, str):
            out.setdefault(par, _empty_spec())

        elif isinstance(par, lmfit.Parameter):
            out[par.name] = {
                'value': par.value,
                'vary': par.vary,
                'min': par.min,
                'max': par.max,
                'expr': par.expr,
                # 'brute_step': par.brute_step, # NO LMFIT DEFAULT PARAMETER HINTS
            }

        elif isinstance(par, lmfit.Parameters):
            for p in par.values():
                _add_param(p)

        elif isinstance(par, list):
            if not par:
                raise ValueError('Empty parameter list [] is not allowed.') 
            # if all(isinstance(p, str) for p in par):
            #     for p in par:
            #         _add_param(p)  
            # else:
            #     for p in par:
            #         _add_param(p)
            # if not all(isinstance(p, str) for p in par):
            #     raise TypeError(
            #         "List parameters must contain only strings (parameter names). "
            #         "For mixed or structured specs, use dict, tuple, or lmfit.Parameters."
            #     )
            for p in par:
                _add_param(p)

        elif isinstance(par, dict):
            for name, spec in par.items():
                base = _empty_spec()
                
                ### --- OLD --- ###
                # if isinstance(spec, dict):
                #     base.update(spec)
                # else:
                #     base['value'] = spec
                # out[name] = base
                ### --- OLD --- ###
    
                if isinstance(spec, dict):
                    # --- STRICT VALIDATION OF PARAMETER HINT KEYS ---
                    invalid_keys = (
                        set(spec) - set(_ALLOWED_LMFIT_PARAMETER_HINT_KEYS)
                    )
                    if invalid_keys:
                        raise KeyError(
                            f"Invalid lmfit parameter hint(s) for '{name}': "
                            f"{sorted(invalid_keys)}. "
                            f"Allowed keys are {_ALLOWED_LMFIT_PARAMETER_HINT_KEYS}."
                        )

                    base.update(spec)

                else:
                    # Shorthand value-only specification
                    base['value'] = spec

                out[name] = base

        elif isinstance(par, tuple):
            params = lmfit.Parameters()
            params.add_many(par)
            for p in params.values():
                _add_param(p)

        else:
            # raise TypeError(f"Unsupported parameter spec: {type(par)}")
            raise TypeError(f"Unsupported parameter spec: {par!r} of type {type(par)}")

    for item in parlist:
        _add_param(item)

    return out

# %%
def finalize_parameter_specs(pardict: dict) -> dict:
    """
    Convert canonical parameter specifications into lmfit-ready
    initialization dictionaries.

    Replaces all `_UNSET` placeholders using
    `_LMFIT_INIT_PARAMETER_DEFAULTS`.

    Args:
        pardict (dict):
            Output of normalize_parameter_specs().

    Returns:
        dict:
            Fully expanded parameter initialization dictionary
            suitable for lmfit.
    """
    out = {}

    for name, spec in pardict.items():
        final = {}
        for key, default in _LMFIT_INIT_PARAMETER_DEFAULTS.items():
            val = spec.get(key, _UNSET)
            final[key] = default if val is _UNSET else val
        out[name] = final

    return out

# %%
# def split_function_arguments(func) -> tuple[list[str], dict[str, Any]]:
#     """
#     Split function parameters into fit parameters and fixed keyword arguments.

#     Args:
#         func (callable): Model function.

#     Returns:
#         fit_params (list[str]):
#             Names of parameters suitable for lmfit fitting.
#         fixed_kws (dict[str, Any]):
#             Keyword arguments with fixed (non-fittable) defaults.
#     """
#     sig = inspect.signature(func)
#     params = list(sig.parameters.values())[1:]  # skip independent var

#     fit_params = []
#     fixed_kws = {}

#     for p in params:
#         default = p.default

#         if default is inspect._empty:
#             # No default → assume fit parameter
#             fit_params.append(p.name)

#         elif isinstance(default, _ALLOWED_NUMERIC):
#             fit_params.append(p.name)

#         else:
#             # str, None, bool, enums → fixed keyword
#             fixed_kws[p.name] = default

#     return fit_params, fixed_kws


def split_function_arguments(
    func: Callable,
) -> tuple[list[str], dict[str, Any]]:
    """
    Split function parameters into fit parameters and fixed keywords.

    Args:
        func (callable): Model function.

    Returns:
        fit_params (list[str]):
            Names of parameters suitable for lmfit fitting.
        fixed_kws (dict[str, Any]):
            Keyword arguments with fixed (non-fittable) defaults.
    """

    spec = getfullargspec_no_self(func)

    # Skip independent variable
    args = spec.args[1:]

    fit_params: list[str] = []
    fixed_kws: dict[str, Any] = {}

    defaults = spec.defaults or ()
    n_defaults = len(defaults)

    # Map args → defaults
    default_map = {
        name: defaults[i - (len(args) - n_defaults)]
        for i, name in enumerate(args)
        if i >= len(args) - n_defaults
    }

    for name in args:

        if name not in default_map:
            fit_params.append(name)
            continue

        default = default_map[name]

        if isinstance(default, numbers.Real):
            fit_params.append(name)
        else:
            fixed_kws[name] = default

    return fit_params, fixed_kws