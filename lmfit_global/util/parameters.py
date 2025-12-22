# %%
from typing import Union, Iterable, Dict, Any

# %%
# --- The package lmfit is a MUST
try:
    import lmfit
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("lmfit is required. Install with `pip install lmfit`") from exc

_UNSET = object()

# %%
# def normalize_parameter_specs(
#     *parlist: Union[str, lmfit.Parameter, lmfit.Parameters, Iterable, Dict]
# ) -> Dict[str, Dict[str, Any]]:
#     """
#     Normalize heterogeneous parameter specifications into a canonical dictionary.

#     This utility accepts a flexible mix of parameter specifications
#     (strings, lmfit.Parameter objects, lmfit.Parameters containers,
#     lists/tuples, and dictionaries) and converts them into a unified form:

#         Dict[str, Dict[str, Any]]

#     where each key is a parameter name and each value is a dictionary
#     of lmfit-compatible parameter attributes.

#     Supported input forms
#     ---------------------
#     1. String (parameter name):
#         - Creates a default parameter specification.

#     2. lmfit.Parameter:
#         - Extracts value, bounds, vary flag, expression, and brute_step.

#     3. lmfit.Parameters:
#         - Iterates over contained Parameter objects.

#     4. Dictionary:
#         - Full specification:
#             {"sigma_0": {"value": 1.0, "min": 0}}
#         - Shorthand value-only:
#             {"sigma_1": 2.0}

#     5. List or tuple:
#         - May contain any mix of the above.
#         - Nested lists/tuples are flattened.
#         - Empty containers are not allowed.

#     Notes
#     -----
#     - Later definitions override earlier ones (last-wins behavior).
#     - Missing fields are filled with sensible defaults.
#     - Unsupported types raise TypeError.
#     - Empty lists/tuples raise ValueError.

#     Returns
#     -------
#     Dict[str, Dict[str, Any]]
#         Canonical parameter specification dictionary.

#     Examples
#     --------
#     Basic string input
#     >>> normalize_parameter_specs("sigma_0")
#     {'sigma_0': {'value': -inf, 'vary': True, 'min': -inf,
#                  'max': inf, 'expr': None, 'brute_step': None}}

#     Shorthand dictionary (value-only)
#     >>> normalize_parameter_specs({"sigma_1": 2.0})
#     {'sigma_1': {'value': 2.0, 'vary': True, 'min': -inf,
#                  'max': inf, 'expr': None, 'brute_step': None}}

#     Full dictionary specification
#     >>> normalize_parameter_specs({"sigma_2": {"value": 1.0, "min": 0}})
#     {'sigma_2': {'value': 1.0, 'vary': True, 'min': 0,
#                  'max': inf, 'expr': None, 'brute_step': None}}

#     lmfit.Parameter input
#     >>> p = lmfit.Parameter("sigma_3", value=3.0, vary=False)
#     >>> normalize_parameter_specs(p)
#     {'sigma_3': {'value': 3.0, 'vary': False, 'min': None,
#                  'max': None, 'expr': None, 'brute_step': None}}

#     lmfit.Parameters input
#     >>> params = lmfit.Parameters()
#     >>> params.add("a", value=1.0, vary=True, min=0, max=10)
#     >>> params.add("b", value=2.0, vary=False)
#     >>> normalize_parameter_specs(params)
#     {'a': {'value': 1.0, 'vary': True, 'min': 0, 'max': 10, 'expr': None, 'brute_step': None}, 
#      'b': {'value': 2.0, 'vary': False, 'min': -inf, 'max': inf, 'expr': None, 'brute_step': None}}
    
#     tuple input
#     >>> par = ('sig', 10, True, None, None, None, None)
#     >>> normalize_parameter_specs(par)
#     {'sig': {'value': 10, 'vary': True, 'min': -inf, 'max': inf, 'expr': None, 'brute_step': None}}


#     Mixed and nested inputs
#     >>> params = lmfit.Parameters()
#     >>> params.add("a", value=1.0, vary=True, min=0, max=10)
#     >>> params.add("b", value=2.0, vary=False)
#     >>> normalize_parameter_specs(
#     ...     "x",
#     ...     {"y": 2.0},
#     ...     [lmfit.Parameter("z", value=3.0), "c"],
#     ...     params,
#     ...     ('f', 10, True, None, None, None, None),
#     ... )
#     {
#       'x': {...},
#       'y': {...},
#       'z': {...},
#       'c': {...},
#       'a': {...},
#       'b': {...},
#       'f': {...}
#     }

#     Override behavior (last wins)
#     >>> normalize_parameter_specs("x", {"x": {"value": 5}})
#     {'x': {'value': 5, 'vary': True, 'min': -inf,
#            'max': inf, 'expr': None, 'brute_step': None}}
#     """

#     out: Dict[str, Dict[str, Any]] = {}

#     def _add_param(par):
#         """
#         """
#         if isinstance(par, str):
#             out.setdefault(
#                 par, 
#                 {
#                     'value': -np.inf, 'vary': True,
#                     'min': -np.inf, 'max': +np.inf,
#                     'expr': None, 'brute_step': None
#                 }
#                     )

#         elif isinstance(par, lmfit.Parameter):
#             out[par.name] = {
#                 'value': par.value, 'vary': par.vary,
#                 'min': par.min, 'max': par.max,
#                 'expr': par.expr, 'brute_step': par.brute_step,
#             }

#         elif isinstance(par, lmfit.Parameters):
#             for p in par.values():
#                 _add_param(p)

#         elif isinstance(par, list):
#             if not par:
#                 raise ValueError('Empty parameter list [] is not allowed.') 
#             # if all(isinstance(p, str) for p in par):
#             #     for p in par:
#             #         _add_param(p)  
#             # else:
#             #     for p in par:
#             #         _add_param(p)
#             # if not all(isinstance(p, str) for p in par):
#             #     raise TypeError(
#             #         "List parameters must contain only strings (parameter names). "
#             #         "For mixed or structured specs, use dict, tuple, or lmfit.Parameters."
#             #     )
#             for p in par:
#                 _add_param(p)

#         elif isinstance(par, tuple):
#             # par = lmfit.Parameters().add_many(par)
#             params = lmfit.Parameters()
#             params.add_many(par)
#             for p in params.values():
#                 _add_param(p)
    
#         elif isinstance(par, dict):
#             for name, spec in par.items():
#                 if isinstance(spec, dict):
#                     out[name] = {
#                         'value': spec.get('value', -np.inf), 'vary': spec.get('vary', True),
#                         'min': spec.get('min', -np.inf), 'max': spec.get('max', +np.inf),
#                         'expr': spec.get('expr', None), 'brute_step': spec.get('brute_step', None),
#                     }
#                 else:
#                     out[name] = {
#                         'value': spec, 'vary': True,
#                         'min': -np.inf, 'max': +np.inf,
#                         'expr': None, 'brute_step': None,
#                     }

#         else:
#             raise TypeError(f"Unsupported param spec {par!r} of type {type(par)}")
           
#     #
#     for item in parlist:
#         _add_param(item)

#     return out

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
    - Missing fields are filled with sensible defaults as `_UNSET`
    - Defaults are applied only when creating parameters
    - Unsupported types raise TypeError.
    - Empty lists/tuples raise ValueError.

    Returns:
        Dict[str, Dict[str, Any]]
            Canonical parameter specification dictionary.

    Examples
    --------
    Basic string input
    >>> normalize_parameter_specs("sigma_0")
    {'sigma_0': {'value': -inf, 'vary': True, 'min': -inf,
                 'max': inf, 'expr': None, 'brute_step': None}}

    Shorthand dictionary (value-only)
    >>> normalize_parameter_specs({"sigma_1": 2.0})
    {'sigma_1': {'value': 2.0, 'vary': True, 'min': -inf,
                 'max': inf, 'expr': None, 'brute_step': None}}

    Full dictionary specification
    >>> normalize_parameter_specs({"sigma_2": {"value": 1.0, "min": 0}})
    {'sigma_2': {'value': 1.0, 'vary': True, 'min': 0,
                 'max': inf, 'expr': None, 'brute_step': None}}

    lmfit.Parameter input
    >>> p = lmfit.Parameter("sigma_3", value=3.0, vary=False)
    >>> normalize_parameter_specs(p)
    {'sigma_3': {'value': 3.0, 'vary': False, 'min': None,
                 'max': None, 'expr': None, 'brute_step': None}}

    lmfit.Parameters input
    >>> params = lmfit.Parameters()
    >>> params.add("a", value=1.0, vary=True, min=0, max=10)
    >>> params.add("b", value=2.0, vary=False)
    >>> normalize_parameter_specs(params)
    {'a': {'value': 1.0, 'vary': True, 'min': 0, 'max': 10, 'expr': None, 'brute_step': None}, 
     'b': {'value': 2.0, 'vary': False, 'min': -inf, 'max': inf, 'expr': None, 'brute_step': None}}
    
    tuple input
    >>> par = ('sig', 10, True, None, None, None, None)
    >>> normalize_parameter_specs(par)
    {'sig': {'value': 10, 'vary': True, 'min': -inf, 'max': inf, 'expr': None, 'brute_step': None}}


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
           'max': inf, 'expr': None, 'brute_step': None}}
    """

    out: Dict[str, Dict[str, Any]] = {}

    def _empty_spec():
        return {
            'value': _UNSET,
            'vary': _UNSET,
            'min': _UNSET,
            'max': _UNSET,
            'expr': _UNSET,
            'brute_step': _UNSET,
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
                'brute_step': par.brute_step,
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
                if isinstance(spec, dict):
                    base.update(spec)
                else:
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



