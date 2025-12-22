# %%
import re
import copy
import logging
import inspect
import operator
import numpy as np
import functools as ft

# %%
def _r2_score(y_true, y_pred, multioutput='uniform_average'):
    """:math:`R^2` (coefficient of determination) regression score function.

    Best possible score is 1.0 and it can be negative (because the
    model can be arbitrarily worse). In the general case when the true y is
    non-constant, a constant model that always predicts the average y
    disregarding the input features would get a :math:`R^2` score of 0.0.

    Args:
        y_true (numpy.ndarray): True/Correct/Experimental data in ndarray-like of shape 
            (n_samples,) or (n_samples, n_outputs).
        y_pred (numpt.ndarray): Estimated/Target/Predicted values in ndarray-like of shape 
            (n_samples,) or (n_samples, n_outputs).
             
        multioutput (str, optional): Defines aggregating of multiple output scores. 
            Defaults to 'uniform_average'. Other options are:

            'raw_values' :
                Returns a full set of scores in case of multioutput input.

            'uniform_average' :
                Scores of all outputs are averaged with uniform weight.

            'variance_weighted' :
                Scores of all outputs are averaged, weighted by the variances
                of each individual output.

    Returns:
        (float or ndarray of floats): The :math:`R^2` score or ndarray of scores 
        if 'multioutput' is 'raw_values'.
    """
    
    def r_square_nan(y, f):
        """Calculate the :math:`R^2` statistic.

        Args:
            y (array-like): Array of observed data.
            f (array-like): Array of fitted data (model predictions).

        Returns:
            float: :math:`R^2` statistic.
        """
        # Convert inputs to numpy arrays if they aren't already
        y = np.array(y)
        f = np.array(f)
        # Mask to ignore NaN values
        mask = ~np.isnan(y)
        # mask = ~np.isnan(y) & ~np.isnan(f)

        y_mean = np.mean(y[mask])
        ss_res = np.sum((y[mask] - f[mask]) ** 2)
        ss_tot = np.sum((y[mask] - y_mean) ** 2)

        r_squared = 1 - (ss_res / ss_tot)
        return r_squared, ss_tot
    
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Normalize to 2-D
    if y_true.ndim==1:
        y_true = y_true.reshape(-1, 1)
        y_pred = y_pred.reshape(-1, 1)

    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ValueError(f'`y_true` and `y_pred` must be 2-D after reshaping')
    if y_true.ndim != y_pred.ndim:
        raise ValueError(
            f'Shape mismatch: `y_true` has ndim={y_true.ndim}, '
            f'`y_pred` has ndim={y_pred.ndim}'
        )

    r2_scores = []
    ss_tot_list = []

    # Compute per-output R^2 and ss_tot
    for i in range(y_true.shape[1]):
        r2, ss_tot = r_square_nan(y_true[:, i], y_pred[:, i])
        r2_scores.append(r2)
        ss_tot_list.append(ss_tot)

    r2_scores = np.array(r2_scores)
    ss_tot_list = np.array(ss_tot_list)

    if multioutput == 'raw_values':
        return r2_scores
    elif multioutput == 'uniform_average':
        return np.mean(r2_scores)
    elif multioutput == 'variance_weighted':
        weights = ss_tot_list / np.sum(ss_tot_list)
        return np.sum(weights * r2_scores)
    else:
        raise ValueError("Invalid multioutput option")


# %%
def wrap_model_reprstring(expr, width=80, indent=4):
    """Wrap a composite model expression string at operators for readability.

    Args:
        expr (str): The composite model expression string.
        width (int, optional): Max line width (default is 80).
        indent (int, optional): Spaces to indent continuation lines (default is 4)

    Returns:
        str: Wrapped expression string.
    """
    tokens = re.split(r'(\+|\-|\*|/)', expr)  # split but keep operators
    lines = []
    current = ""

    for tok in tokens:
        if len(current) + len(tok) + 1 > width:
            lines.append(current.rstrip())
            current = " " * indent + tok
        else:
            current += tok
    if current:
        lines.append(current.rstrip())

    return "\n".join(lines)

# %%
def ensure_2darray(arr):
    """Ensure array is always shape (N, M), and 1D becomes (N, 1).

    Args:
        arr (list): Numpy array in 1 dimension or list or list of lists

    Returns:
        numpy.array: 2d array of shape(N,M)
    """
    arr = np.asarray(arr)
    
    # Make it 2D
    arr = np.atleast_2d(arr)

    # If row vector → convert to column vector
    if arr.shape[0] == 1 and arr.ndim == 2:
        arr = arr.T

    return arr


def pad_list_with_nan(lst):
    """
    Pads a list of lists with NaN to ensure all inner lists have the same length.

    Args:
        arr (lst): List of lists of float or int.

    Returns:
        numpy.ndarray: 2D array with shorter rows padded with np.nan.
                    If all rows are the same length, returns the array as-is.
    """
    row_lengths = [len(row) for row in lst]
    if len(set(row_lengths)) == 1:
        # All rows have the same length — no padding needed
        return np.array(lst, dtype=float)

    max_len = max(row_lengths)
    pad_arr = np.full((len(lst), max_len), np.nan)
    for i, row in enumerate(lst):
        pad_arr[i, :len(row)] = row
    return pad_arr


def merge_xyerr_data_numpy(xdat_lst, ydat_lst, yerr_lst=None):
    """
    Merges multiple (x, y, yerr) datasets into a unified NumPy array with shared x-axis.
    Missing values are filled with np.nan. If yerr_lst is None, error columns are omitted.

    Args:
        xdat_lst (list of np.ndarray): List of x arrays.
        ydat_lst (list of np.ndarray): List of y arrays.
        yerr_lst (list of np.ndarray or None): List of error arrays or None.

    Returns:
        np.ndarray: 2D array with columns: x, y_1, yerr_1, y_2, yerr_2, ...
    """
    all_x = np.unique(np.concatenate(xdat_lst))
    N = len(xdat_lst)

    y_merged = np.full((len(all_x), N), np.nan)
    yerr_merged = None if yerr_lst is None else np.full((len(all_x), N), np.nan)

    for i in range(N):
        x = xdat_lst[i]
        y = ydat_lst[i]
        for xi, yi in zip(x, y):
            idx = np.where(all_x == xi)[0]
            if idx.size > 0:
                y_merged[idx[0], i] = yi
        if yerr_lst is not None:
            for xi, ei in zip(x, yerr_lst[i]):
                idx = np.where(all_x == xi)[0]
                if idx.size > 0:
                    yerr_merged[idx[0], i] = ei

    # Stack into final output: [x | y_1 | yerr_1 | y_2 | yerr_2 | ...]
    if yerr_lst is not None:
        combined = [all_x.reshape(-1, 1)]
        for i in range(N):
            combined.append(y_merged[:, i].reshape(-1, 1))
            combined.append(yerr_merged[:, i].reshape(-1, 1))
        return np.hstack(combined)
    else:
        return np.column_stack((all_x, y_merged))
    

def build_composite_model(model_lst, op_list):
    """
    Build a composite lmfit.Model from a list of models and operators.

    Parameters
    ----------
    model_lst : list of lmfit.Model
        Models to combine.
    op_list : list of str
        Operators ('+', '-', '*', '/') of length len(model_lst)-1.

    Returns
    -------
    composite_model : lmfit.Model
        Composite model object.
    """
    op_map = {
        '+': operator.add,
        '-': operator.sub,
        '*': operator.mul,
        '/': operator.truediv,
    }

    composite_model = ft.reduce(
        lambda x, y: op_map[y[1]](x, y[0]),
        zip(model_lst[1:], op_list),
        model_lst[0]
    )
    return composite_model
    
    
def build_composite_model_expr(model_lst, op_list):
    """
    Build a human-readable expression string for the composite model.

    Parameters
    ----------
    model_lst : list of lmfit.Model
        Models to describe.
    op_list : list of str
        Operators ('+', '-', '*', '/') of length len(model_lst)-1.

    Returns
    -------
    expr : str
        String representation of the composite function with arguments.
    """
    expr_parts = []
    for i, mdl in enumerate(model_lst):
        func = mdl.func
        sig = inspect.signature(func)
        args = [pname for pname in sig.parameters]
        arg_str = ', '.join(args)
        func_str = f'{func.__name__}({arg_str})'
        expr_parts.append(func_str)
        if i < len(op_list):
            expr_parts.append(op_list[i])
    return ' '.join(expr_parts)


def _evaluate_function(function, xdat, params, prefix, i, kws=None):
    """
    Evaluate a single function with parameters from lmfit and fixed options.

    Parameters
    ----------
    function : callable
        The function to evaluate (e.g. gaussian, exponential).
    xdat : array-like
        Input x data.
    params : lmfit.Parameters
        Composite parameter set containing prefixed parameter names.
    prefix : str
        Prefix for this function (e.g. 'c0_', 'c1_').
    i : int
        Index for dataset (e.g. 0 for first dataset).
    kws : dict, optional
        Fixed options (non-fit parameters like form='erf', gamma=5.0).

    Returns
    -------
    ndarray
        Evaluated function values.
    """
    if kws is None:
        kws = {}

    # Get ordered function parameters
    fn_pars = list(inspect.signature(function).parameters.keys())

    # First parameter is the x variable name
    xname = fn_pars[0]

    # Remaining parameters
    argnames = fn_pars[1:]

    kwargs = {}
    for name in argnames:
        # Case 1: name is in lmfit parameter set → pull from params
        param_key = f"{prefix}{name}_{i}"
        if param_key in params:
            kwargs[name] = params[param_key].value

        # Case 2: name is in extra kwargs (fixed parameters)
        elif name in kws:
            kwargs[name] = kws[name]

        # Case 3: leave it alone; function will handle default value
        else:
            pass

    # Evaluate function safely
    return function(xdat, **kwargs)


def evaluate_function(func, x, params, prefix, i, func_kws=None):
    """Evaluate a single function with parameters from lmfit and `func` keyward arguments (if any).

    Args:
        func (callable): The function to evaluate
        x (array, list of floats): Array-like of x data
        params (lmfit.Parameters): Contains the Parameters for the model.
        prefix (str): Prefix for the function `func` (e.g. 'c0_', 'c1_').
        i (int): Index for dataset (e.g. 0 for first dataset).
        func_kws (_type_, optional): Additional keyword arguments to pass to model function. 
            Defaults to None.

    Returns:
        ndarray: Evaluated function `func` values.
    """
    if func_kws is None:
        func_kws = {}

    x = np.array(x)

    # --- Get ordered function parameters ---
    fn_pars = list(inspect.signature(function).parameters.keys())
    argnames = fn_pars[1:]  # skip first (x variable)

    kwargs = {}
    for name in argnames:
        param_key = f'{prefix}{name}_{i}'
        if param_key in params:
            kwargs[name] = params[param_key].value
        elif name in func_kws:
            kwargs[name] = func_kws[name]
        # else: leave default
        else:
            pass # IS THIS OKAY!?

    return func(x, **kwargs)
    
    

# %%
class MyClass:
    def __init__(self, prefix_on=False, **kws):
        self._prefix_on = prefix_on
        for key, val in kws.items():
            setattr(self, key, val)

    @property
    def prefix_on(self):
        return self._prefix_on

    @prefix_on.setter
    def prefix_on(self, value):
        self._prefix_on = bool(value)


obj = MyClass(name="MUHAMMAD", age=30, active=True)

# print(obj.name)   # → "MUHAMMAD"
# print(obj.age)    # → 30
# print(obj.active) # → True





