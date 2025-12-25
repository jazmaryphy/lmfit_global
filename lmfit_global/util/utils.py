# %%
import inspect
import numpy as np
from typing import Iterable, Callable, Sequence

# %%
def normalize_xrange(xrange_):
    """
    Normalize xrange input into (xmin, xmax).

    Supported formats:
      - None
      - (xmin, xmax)
      - [xmin, xmax]
      - {"min": xmin, "max": xmax}
      - {"xmin": xmin, "xmax": xmax}
    """
    if xrange_ is None:
        return None, None

    if isinstance(xrange_, dict):
        xmin = xrange_.get("min", xrange_.get("xmin"))
        xmax = xrange_.get("max", xrange_.get("xmax"))
        return xmin, xmax

    if isinstance(xrange_, (tuple, list)) and len(xrange_) == 2:
        return xrange_[0], xrange_[1]

    raise ValueError(
        "`xrange` must be None, (xmin, xmax), or dict "
        "{min/xmin, max/xmax}"
    )


def validate_xrange(xmin, xmax):
    """
    Validate xmin/xmax values.

    Returns:
        (xmin, xmax) : tuple[float | None, float | None]
    """
    for name, val in (("xmin", xmin), ("xmax", xmax)):
        if val is not None:
            try:
                val = float(val)
            except Exception:
                raise ValueError(f"`{name}` must be float or None")

        if name == "xmin":
            xmin = val
        else:
            xmax = val

    if xmin is not None and xmax is not None and xmin >= xmax:
        raise ValueError("`xmin` must be < `xmax`")

    return xmin, xmax


def parse_xrange(xrange_, *, xdata=None, clip=True, logger=None):
    """
    Parse, validate, and optionally clip xrange.

    Args:
        xrange_ (None | tuple | list | dict):
        xdata (ndarray, optional):
            Used for clipping if clip=True
        clip (bool): 
            Clip xrange to data limits
        logger (logging.Logger, optional)

    Returns:
        (xmin, xmax)
    """
    xmin, xmax = normalize_xrange(xrange_)
    xmin, xmax = validate_xrange(xmin, xmax)

    if xdata is None:
        return xmin, xmax

    dmin, dmax = np.min(xdata), np.max(xdata)

    if xmin is None:
        xmin = dmin
    if xmax is None:
        xmax = dmax

    if clip:
        if xmin < dmin:
            if logger:
                logger.warning(f"xmin={xmin} < data min {dmin}, clipping")
            xmin = dmin

        if xmax > dmax:
            if logger:
                logger.warning(f"xmax={xmax} > data max {dmax}, clipping")
            xmax = dmax

    return xmin, xmax

# %%
# -----------------------------------------------------------
# 1. Build the data dictionary
# -----------------------------------------------------------
def make_data_dict(xy, xrange=None):
    xy = np.asarray(xy)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError("`xy` must be a 2D array with columns [x, y1, y2, ...].")
    if xrange is not None:
        if not (isinstance(xrange, (tuple, list)) and len(xrange) == 2):
            raise ValueError("`xrange` must be tuple/list: (xmin, xmax).")
    return {"xy": xy, "xrange": xrange}


# -----------------------------------------------------------
# 2. Build the function dictionary
# -----------------------------------------------------------
def make_function_dict(func_list, connectors=None):
    if not isinstance(func_list, list) or len(func_list) == 0:
        raise ValueError("func_list must be a non-empty list of model specifications.")

    for entry in func_list:
        if "func_name" not in entry:
            raise ValueError("Each function spec must include 'func_name'")
        if not callable(entry["func_name"]):
            raise TypeError("'func_name' must be callable (a Python function).")
        if "init_params" not in entry or not isinstance(entry["init_params"], dict):
            raise ValueError("'init_params' must be a dict.")
        if "func_kws" not in entry:
            entry["func_kws"] = {}
        _check_function_signature(entry["func_name"], entry["init_params"])

    if connectors is None:
        connectors = [] if len(func_list) == 1 else None

    if connectors is None:
        raise ValueError(f"connectors must be provided for {len(func_list)} models.")

    if len(connectors) != len(func_list) - 1:
        raise ValueError("Number of connectors must be (n_functions - 1).")

    allowed = {"+", "-", "*", "/"}
    for op in connectors:
        if op not in allowed:
            raise ValueError(f"Unsupported operator '{op}'. Allowed: {allowed}.")

    return {"theory": func_list, "theory_connectors": connectors}


# -----------------------------------------------------------
# 3. Signature validation helper
# -----------------------------------------------------------
def _check_function_signature(func, init_params):
    sig = inspect.signature(func)
    argnames = list(sig.parameters.keys())[1:]  # skip x
    provided = list(init_params.keys())
    unknown = [p for p in provided if p not in argnames]
    missing = [p for p in argnames if p not in provided]
    if unknown:
        print(f"[WARNING] {func.__name__}: unknown init parameters {unknown}. Function args={argnames}")
    if missing:
        print(f"[INFO] {func.__name__}: missing init parameters {missing}.")


# -----------------------------------------------------------
# 4. Make items
# -----------------------------------------------------------
def make_items(data_dict, function_dict):
    return {"data": data_dict, "functions": function_dict}



def make_items_from_xy(x, y_list, func_list, connectors=None, xrange=None):
    """
    Build full items dict from simple inputs.

    Args:
        x (array-like): 1D x array.
        y_list (list of arrays): each y is same length as x.
        func_list: list of function specs (see examples)
        connectors: operators connecting functions
        xrange: optional tuple (xmin, xmax)

    Returns:
        items dict
    """
    x = np.asarray(x)
    y_list = [np.asarray(y) for y in y_list]
    for y in y_list:
        if len(y) != len(x):
            raise ValueError("All y arrays must have same length as x.")

    xy = np.column_stack([x] + y_list)
    data_dict = make_data_dict(xy, xrange)
    function_dict = make_function_dict(func_list, connectors)
    return make_items(data_dict, function_dict)


def make_items_from_dict(x, y_dict, func_list, connectors=None, xrange=None):
    """
    Build items when y datasets are provided in a dictionary.

    Example:
        y_dict = {
            "sample1": y1,
            "sample2": y2,
            "sample3": y3,
        }
    """
    x = np.asarray(x)
    names = list(y_dict.keys())
    y_list = [np.asarray(y_dict[k]) for k in names]

    return make_items_from_xy(x, y_list, func_list, connectors, xrange)

# %%
class GlobalFitBuilder:
    """
    Fluent interface for building `LmfitGlobal` input items.

    This builder simplifies the creation of the `items` dictionary
    required by `LmfitGlobal`, supporting:
    - multi-dataset data
    - multi-component models
    - connector validation
    - optional x-range filtering
    """

    def __init__(self):
        self.reset()

    # --------------------- LIFECYCLE ---------------------
    def reset(self):
        """Reset builder state for reuse."""
        self._x: np.ndarray | None = None
        self._datasets: list[np.ndarray] | None = None
        self._models: list[dict] = []
        self._connectors: list[str] = []
        self._xrange: tuple[float | None, float | None] | None = None
        return self

    # --------------------- DATA --------------------------
    def set_data(self, x, y, *, xrange=None):
        """
        Set x and y data.

        Args:
            x (array_like): 1D array of x values, shape (N,)
            y (array_like | list[array_like]):
                - list of arrays [y0, y1, ...], each shape (N,)
                - OR ndarray shape (N, ny)
            xrange (tuple[float | None, float | None], optional):
                Optional (xmin, xmax) range filter.
        """
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("x must be a 1D array")

        # --- Case 1: y is list/tuple of 1D arrays ---
        if isinstance(y, (list, tuple)):
            datasets = [np.asarray(yi) for yi in y]
            if any(yi.ndim != 1 for yi in datasets):
                raise ValueError("Each y in y_list must be 1D")
            if any(len(yi) != len(x) for yi in datasets):
                raise ValueError("Each y must have same length as x")

        # --- Case 2: y is ndarray ---
        else:
            y = np.asarray(y)
            if y.ndim == 1:
                datasets = [y]
            elif y.ndim == 2:
                if y.shape[0] != len(x):
                    raise ValueError("y.shape[0] must match len(x)")
                datasets = [y[:, i] for i in range(y.shape[1])]
            else:
                raise ValueError("y must be 1D, 2D, or list of 1D arrays")

        self._x = x
        self._datasets = datasets
        self._xrange = xrange
        return self

    def add_dataset(self, y):
        if self._datasets is None:
            self._datasets = []
        y = np.asarray(y)
        if len(y) != len(self._x):
            raise ValueError("Dataset length mismatch")
        self._datasets.append(y)
        return self


    # --------------------- MODELS ------------------------
    def add_model(self, func: Callable, init_params: dict, *, func_kws=None):
        """
        Add a model component.

        Args:
            func (callable): Model function f(x, ...)
            init_params (dict): lmfit-style parameter hints
            func_kws (dict, optional): Fixed keyword arguments
        """
        if not callable(func):
            raise TypeError("func must be callable")

        if not isinstance(init_params, dict):
            raise TypeError("init_params must be a dict")

        func_kws = func_kws or {}
        if not isinstance(func_kws, dict):
            raise TypeError("func_kws must be a dict")

        self._models.append(
            {
                "func_name": func,
                "init_params": init_params,
                "func_kws": func_kws,
            }
        )
        return self

    # ------------------ CONNECTORS -----------------------
    def connect(self, *ops: str):
        """
        Provide operators connecting added models.

        Example:
            .connect("+", "*")
        """
        self._connectors = list(ops)
        return self

    # ----------------------- BUILD -----------------------
    def build(self):
        """
        Build and return the `items` dictionary for LmfitGlobal.
        """
        if self._x is None or self._datasets is None:
            raise ValueError("Call set_data(x, y) before build()")

        if not self._models:
            raise ValueError("At least one model must be added")

        if self._connectors:
            expected = len(self._models) - 1
            if len(self._connectors) != expected:
                raise ValueError(
                    f"{len(self._models)} models require "
                    f"{expected} connectors, got {len(self._connectors)}"
                )

        return make_items_from_xy(
            self._x,
            self._datasets,
            func_list=self._models,
            connectors=self._connectors,
            xrange=self._xrange,
        )

    # ---------------------- DEBUG ------------------------
    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"datasets={0 if self._datasets is None else len(self._datasets)}, "
            f"models={len(self._models)}, "
            f"xrange={self._xrange})"
        )


# %%


# %%
# import numpy as np
# from scipy.special import erf, erfc
# log2 = np.log(2)
# s2pi = np.sqrt(2*np.pi)
# s2 = np.sqrt(2.0)
# # tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# # here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
# tiny = 1.0e-15

# def not_zero(value):
#     """Return value with a minimal absolute size of tiny, preserving the sign.

#     This is a helper function to prevent ZeroDivisionError's.

#     Parameters
#     ----------
#     value : scalar
#         Value to be ensured not to be zero.

#     Returns
#     -------
#     scalar
#         Value ensured not to be zero.

#     """
#     return float(np.copysign(max(tiny, abs(value)), value))

# def step(x, amplitude, center, sigma, form='linear'):
# # def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
#     """Return a step function.

#     Starts at 0.0, ends at `sign(sigma)*amplitude`, has a half-max at
#     `center`, rising or falling with `form`:

#     - `'linear'` (default) = amplitude * min(1, max(0, arg + 0.5))
#     - `'atan'`, `'arctan'` = amplitude * (0.5 + atan(arg)/pi)
#     - `'erf'`              = amplitude * (1 + erf(arg))/2.0
#     - `'logistic'`         = amplitude * [1 - 1/(1 + exp(arg))]

#     where ``arg = (x - center)/sigma``.

#     Note that ``sigma > 0`` gives a rising step, while ``sigma < 0`` gives
#     a falling step.
#     """
#     out = np.sign(sigma)*(x - center)/max(tiny*tiny, abs(sigma))

#     if form == 'erf':
#         out = 0.5*(1 + erf(out))
#     elif form == 'logistic':
#         out = 1. - 1./(1. + np.exp(out))
#     elif form in ('atan', 'arctan'):
#         out = 0.5 + np.arctan(out)/np.pi
#     elif form == 'linear':
#         out = np.minimum(1, np.maximum(0, out + 0.5))
#     else:
#         msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
#                "of 'erf', 'logistic', 'atan', 'arctan', or 'linear'.")
#         raise ValueError(msg)

#     return amplitude*out


# def linear(x, slope=1.0, intercept=0.0):
#     """Return a linear function.

#     linear(x, slope, intercept) = slope * x + intercept

#     """
#     return slope * x + intercept


# x = np.linspace(0, 10, 201)
# y = np.ones_like(x)
# y[:48] = 0.0
# y[48:77] = np.arange(77-48)/(77.0-48)
# np.random.seed(0)
# y = 110.2 * (y + 9e-3*np.random.randn(x.size)) + 12.0 + 2.22*x

# xy_dat = np.column_stack([x, y])


# data_dict = make_data_dict(
#     xy=xy_dat,
#     xrange=None
# )

# func_list = [
#     {
#         'func_name': step,
#         'init_params': {
#             'amplitude': {'value':100, 'vary':True},
#             'center': {'value':2.5, 'min':0, 'max':10},
#             'sigma': {'value':1},
#         },
#         'func_kws': {'form': 'erf'}
#     },
#     {
#         'func_name': linear,
#         'init_params' : {
#             'slope': {'value':0.0, 'vary':True},
#             'intercept': {'value':0},
#         },
#         'func_kws': {}
#     }
# ]

# function_dict = make_function_dict(
#     func_list,
#     connectors=['+']
# )

# items = make_items(data_dict, function_dict)


# items


# %%
# import numpy as np
# from scipy.special import erf, erfc
# log2 = np.log(2)
# s2pi = np.sqrt(2*np.pi)
# s2 = np.sqrt(2.0)
# # tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# # here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
# tiny = 1.0e-15

# def not_zero(value):
#     """Return value with a minimal absolute size of tiny, preserving the sign.

#     This is a helper function to prevent ZeroDivisionError's.

#     Parameters
#     ----------
#     value : scalar
#         Value to be ensured not to be zero.

#     Returns
#     -------
#     scalar
#         Value ensured not to be zero.

#     """
#     return float(np.copysign(max(tiny, abs(value)), value))

# def step(x, amplitude, center, sigma, form='linear'):
# # def step(x, amplitude=1.0, center=0.0, sigma=1.0, form='linear'):
#     """Return a step function.

#     Starts at 0.0, ends at `sign(sigma)*amplitude`, has a half-max at
#     `center`, rising or falling with `form`:

#     - `'linear'` (default) = amplitude * min(1, max(0, arg + 0.5))
#     - `'atan'`, `'arctan'` = amplitude * (0.5 + atan(arg)/pi)
#     - `'erf'`              = amplitude * (1 + erf(arg))/2.0
#     - `'logistic'`         = amplitude * [1 - 1/(1 + exp(arg))]

#     where ``arg = (x - center)/sigma``.

#     Note that ``sigma > 0`` gives a rising step, while ``sigma < 0`` gives
#     a falling step.
#     """
#     out = np.sign(sigma)*(x - center)/max(tiny*tiny, abs(sigma))

#     if form == 'erf':
#         out = 0.5*(1 + erf(out))
#     elif form == 'logistic':
#         out = 1. - 1./(1. + np.exp(out))
#     elif form in ('atan', 'arctan'):
#         out = 0.5 + np.arctan(out)/np.pi
#     elif form == 'linear':
#         out = np.minimum(1, np.maximum(0, out + 0.5))
#     else:
#         msg = (f"Invalid value ('{form}') for argument 'form'; should be one "
#                "of 'erf', 'logistic', 'atan', 'arctan', or 'linear'.")
#         raise ValueError(msg)

#     return amplitude*out


# def linear(x, slope=1.0, intercept=0.0):
#     """Return a linear function.

#     linear(x, slope, intercept) = slope * x + intercept

#     """
#     return slope * x + intercept

# x = np.linspace(0, 10, 200)
# y0 = step(x, amplitude=100, center=3, sigma=1, form='erf') + linear(x, 1.0, 0.0)
# y1 = step(x, amplitude=80, center=4, sigma=1.2, form='erf') + linear(x, 0.5, 1.0)

# y_list = y0
# # y_list = y0.tolist()  # must be array
# y_list = [y0]
# y_list = [y0, y1]
# y_list = np.column_stack([y0, y1])  # shape (N, ny)

# init_step = {
#     'amplitude': {'value':100, 'vary':True},
#     'center': {'value':2.5, 'min':0, 'max':10},
#     'sigma': {'value':1}
# }

# init_linear = {
#     'slope': {'value':0.0, 'vary':True},
#     'intercept': {'value':0}
# }


# builder = (
#     GlobalFitBuilder()
#     .set_data(x, y_list)                      # x and all y datasets
#     .add_model(step, init_step, func_kws={"form": "erf"})
#     .add_model(linear, init_linear)
#     .connect("+")                             # how to combine the 2 functions
# )

# items = builder.build()

# items


