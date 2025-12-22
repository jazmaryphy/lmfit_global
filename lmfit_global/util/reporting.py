# %%
import re
import copy
import inspect
import logging
import textwrap
import itertools
import numpy as np
import functools as ft

# %%
# --- The package lmfit is a MUST
try:
    import lmfit
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("lmfit is required. Install with `pip install lmfit`") from exc
    
# Optional imports
try:
    import numdifftools 
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False

try:
    from sklearn.metrics import r2_score as sk_r2
except ImportError:
    sk_r2 = None

# %%
def get_default_logger(
    name: str,
    *,
    level: int = logging.INFO,
    fmt: str = "%(levelname)s: %(message)s",
    propagate: bool = False,
) -> logging.Logger:
    """
    Create or return a safe default logger.

    - Adds a StreamHandler only if none exists
    - Prevents duplicate logs
    - Suitable for libraries (not applications)

    Args:
        name (str): Logger name (usually __name__ or class name)
        level (int): Logging level (default: INFO)
        fmt (str): Log format
        propagate (bool): Whether to propagate to root logger

    Returns:
        logging.Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propagate

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    return logger

# %%
def r2_score_util(y_true, y_pred, multioutput='uniform_average'):
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


def r2_safe(y_true, y_pred, **kwargs):
    """Compute :math:`R^2`. Automatically choose sklearn or fallback r2_score_util.

    Args:
    y_true, y_pred : array-like
        True and predicted values.

    **kwargs : Passed to sklearn.metrics.r2_score or r2_score_util.

    Returns:
    float or array-like: :math:`R^2` score(s).

    NOTES:
    - If NaNs are present, ALWAYS use r2_score_util
    - Else use sklearn if available
    - Fallback gracefully if sklearn fails

    """
    # --- If no NaNs, use sklearn when available ---
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    has_nan = np.isnan(y_true).any() or np.isnan(y_pred).any()


    # --- NaN-safe path ---
    if has_nan:
        return r2_score_util(y_true, y_pred, **kwargs)

    # --- Fast path ---
    if sk_r2 is not None:
        try:
            return sk_r2(y_true, y_pred, **kwargs)
        except Exception:
            pass

    # --- Final fallback ---
    return r2_score_util(y_true, y_pred, **kwargs)


def r_squared_safe(y_true, y_pred, **kwargs):
    """Compute :math:`R^2`. Automatically choose sklearn or fallback r2_score_util.

    Args:
    y_true, y_pred : array-like
        True and predicted values.

    **kwargs : Passed to sklearn.metrics.r2_score or r2_score_util.

    Returns:
    float or array-like: :math:`R^2` score(s).

    NOTES:
    - If NaNs are present, ALWAYS use r2_score_util
    - Else use sklearn if available
    - Fallback gracefully if sklearn fails

    """
    # --- If no NaNs, use sklearn when available ---
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    has_nan = np.isnan(y_true).any() or np.isnan(y_pred).any()


    # --- NaN-safe path ---
    if has_nan:
        return r2_score_util(y_true, y_pred, **kwargs)

    # --- Fast path ---
    if sk_r2 is not None:
        try:
            return sk_r2(y_true, y_pred, **kwargs)
        except Exception:
            pass

    # --- Final fallback ---
    return r2_score_util(y_true, y_pred, **kwargs)

# %%
def alphanumeric_sort(s, _nsre=re.compile('([0-9]+)')):
    """Sort alphanumeric string."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]
            
    
def getfloat_key(dct, key, length=11):
    """Format a dictionary value for printing.

    Args:
        dct (dict): Dictionary containing values.
        key (str): Key to look up in the dictionary.
        length (int, optional): Length of formatted output string (default 11).

    Returns:
        str: Formatted string representation of the value.
    """
    val = dct.get(key, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return f'{val}'
    if isinstance(val, float):
        return gformat(val, length=length).strip()
    return repr(val)


def getfloat_attr(obj, attr, length=11):
    """Format an attribute of an object for printing."""
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return f'{val}'
    if isinstance(val, float):
        return gformat(val, length=length).strip()
    return repr(val)


def gformat(val, length=11):
    """Format a number with '%g'-like format.

    Except that:
        a) the length of the output string will be of the requested length.
        b) positive numbers will have a leading blank.
        b) the precision will be as high as possible.
        c) trailing zeros will not be trimmed.

    The precision will typically be ``length-7``.

    Args:
        val (float): Value to be formatted.
        length (int, optional): Length of output string, default is 11.

    Returns:
        str: String of specified length.

    Notes
    ------
    Positive values will have leading blank.

    """
    from math import log10
    if val is None or isinstance(val, bool):
        return f'{repr(val):>{length}s}'
    try:
        expon = int(log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    except TypeError:
        return f'{repr(val):>{length}s}'

    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif ((expon > 0 and expon < (prec+4)) or
          (expon <= 0 and -expon < (prec-1))):
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    return f'{val:{length}.{prec}{form}}'

# %%
def correl_table(params):
    """Return a printable correlation table for a Parameters object."""
    varnames = [vname for vname in params if params[vname].vary]
    nwid = max(8, max([len(vname) for vname in varnames])) + 1

    def sfmt(a):
        return f" {a:{nwid}s}"

    def ffmt(a):
        return sfmt(f"{a:+.4f}")

    title = ['', sfmt('Variable')]
    title.extend([sfmt(vname) for vname in varnames])

    title = '|'.join(title) + '|'
    bar = [''] + ['-'*(nwid+1) for i in range(len(varnames)+1)] + ['']
    bar = '+'.join(bar)

    buff = [bar, title, bar]

    for vname, par in params.items():
        if not par.vary:
            continue
        line = ['', sfmt(vname)]
        for vother in varnames:
            if vother == vname:
                line.append(ffmt(1))
            elif vother in par.correl:
                line.append(ffmt(par.correl[vother]))
            else:
                line.append('unknown')
        buff.append('|'.join(line) + '|')
    buff.append(bar)
    return '\n'.join(buff)

def lmfit_report(
        inpars, 
        rsquared=None,
        modelpars=None, 
        show_correl=True, 
        min_correl=0.1,
        sort_pars=False, 
        correl_mode='list'
        ):
    """Generate a report of the fitting results.

    The report contains the best-fit values for the parameters and their
    uncertainties and correlations.

    Args:
        inpars (lmfit.Parameters): Input Parameters from fit or MinimizerResult returned from a fit.
        rsquared (float, optional): :math:`R^2` coefficient of determinations. Default None
        modelpars (lmfit.Parameters, optional): Known Model Parameters.
        show_correl (bool, optional): Whether to show list of sorted correlations (default is True).
        min_correl (float, optional): Smallest correlation in absolute value to show (default is 0.1).
        sort_pars (bool or callable, optional): Whether to show parameter names sorted in alphanumerical order. 
            If False (default), then the parameters will be listed in the order they were added to the Parameters 
            dictionary. If callable, then this (one argument) function is used to extract a comparison key from each list element.
        correl_mode ({'list', table'} str, optional): Mode for how to show correlations. Can be either 'list' (default) to show a 
            sorted (if ``sort_pars`` is True) list of correlation values, or 'table' to show a complete, formatted table of correlations.

    Returns:
        str: Multi-line text of fit report.
    """
    inpars = copy.deepcopy(inpars) # MAKE OWN COPY
    
    if isinstance(inpars, lmfit.Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params


    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        # dict.keys() returns a KeysView in py3, and they're indexed
        # further down
        parnames = list(params.keys())

    # if r2_dict is None:
    #     r2_dict = {}

    buff = []
    add = buff.append
    namelen = max(len(n) for n in parnames)
    if result is not None:
        add("[[Fit Statistics]]")
        add(f"    # fitting method   = {result.method}")
        add(f"    # function evals   = {getfloat_attr(result, 'nfev')}")
        add(f"    # data points      = {getfloat_attr(result, 'ndata')}")
        add(f"    # variables        = {getfloat_attr(result, 'nvarys')}")
        add(f"    chi-square         = {getfloat_attr(result, 'chisqr')}")
        add(f"    reduced chi-square = {getfloat_attr(result, 'redchi')}")
        add(f"    Akaike info crit   = {getfloat_attr(result, 'aic')}")
        add(f"    Bayesian info crit = {getfloat_attr(result, 'bic')}")
        if hasattr(result, 'rsquared'):
            add(f"    R-squared          = {getfloat_attr(result, 'rsquared')}")
        else:  # MAKE IT SIMPLE
            if rsquared is not None:
                add(f"    R-squared          = {gformat(rsquared, length=11)}")
        ### --- (EXPERIMENTAL) THIS PART MAY NEED IMPROVEMENT LATER --- ###
        # else:  ## ADD THIS PART
        #     mean_val = r2_dict.get('mean', None)
        #     weighted_val = r2_dict.get('weighted', None)
        #     # Only proceed if at least one value is not None
        #     if mean_val is not None and weighted_val is not None:
        #         tol = 1e-12
        #         if abs(mean_val - weighted_val) < tol:
        #             add(f"    R-squared          = {getfloat_key(r2_dict, 'mean')}")
        #         else:
        #             add(f"    R-squared (mean)   = {getfloat_key(r2_dict, 'mean')}")
        #             add(f"    R-squared (weight) = {getfloat_key(r2_dict, 'weighted')}")
        #     elif mean_val is not None:
        #         add(f"    R-squared (mean)   = {getfloat_key(r2_dict, 'mean')}")
        #     elif weighted_val is not None:
        #         # add(f"    R-squared (weighted) = {getfloat_key(r2_dict, 'weighted')}")
        #         # add(f"    R-squared (var)    = {getfloat_key(r2_dict, 'weighted')}")
        #         add(f"    R-squared (weight) = {getfloat_key(r2_dict, 'weighted')}")
        #     # else: both None → do nothing (no add)
            
        if not result.errorbars:
            add("##  Warning: uncertainties could not be estimated:")
            if result.method in ('leastsq', 'least_squares') or HAS_NUMDIFFTOOLS:
                parnames_varying = [par for par in result.params
                                    if result.params[par].vary]
                for name in parnames_varying:
                    par = params[name]
                    space = ' '*(namelen-len(name))
                    if par.init_value and np.allclose(par.value, par.init_value):
                        add(f'    {name}:{space}  at initial value')
                    if (np.allclose(par.value, par.min) or np.allclose(par.value, par.max)):
                        add(f'    {name}:{space}  at boundary')
            else:
                add("    this fitting method does not natively calculate uncertainties")
                add("    and numdifftools is not installed for lmfit to do this. Use")
                add("    `pip install numdifftools` for lmfit to estimate uncertainties")
                add("    with this fitting method.")

    add("[[Variables]]")
    for name in parnames:
        par = params[name]
        space = ' '*(namelen-len(name))
        nout = f"{name}:{space}"
        inval = '(init = ?)'
        if par.init_value is not None:
            inval = f'(init = {par.init_value:.7g})'
        if modelpars is not None and name in modelpars:
            inval = f'{inval}, model_value = {modelpars[name].value:.7g}'
        try:
            sval = gformat(par.value)
        except (TypeError, ValueError):
            sval = ' Non Numeric Value?'
        if par.stderr is not None:
            serr = gformat(par.stderr)
            try:
                spercent = f'({abs(par.stderr/par.value):.2%})'
            except ZeroDivisionError:
                spercent = ''
            sval = f'{sval} +/-{serr} {spercent}'

        if par.vary:
            add(f"    {nout} {sval} {inval}")
        elif par.expr is not None:
            add(f"    {nout} {sval} == '{par.expr}'")
        else:
            add(f"    {nout} {par.value: .7g} (fixed)")

    if show_correl and correl_mode.startswith('tab'):
        add('[[Correlations]] ')
        for line in correl_table(params).split('\n'):
            buff.append('  %s' % line)
    elif show_correl:
        correls = {}
        for i, name in enumerate(parnames):
            par = params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls[f"{name}, {name2}"] = par.correl[name2]

        sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
        sort_correl.reverse()
        if len(sort_correl) > 0:
            add('[[Correlations]] (unreported correlations are < '
                f'{min_correl:.3f})')
            maxlen = max(len(k) for k in list(correls.keys()))
        for name, val in sort_correl:
            lspace = max(0, maxlen - len(name))
            add(f"    C({name}){(' '*30)[:lspace]} = {val:+.4f}")
    return '\n'.join(buff)

# %%
def build_expr(funcs, operators):
    """Build a human-readable expression string from functions and operators.

    Args:
        funcs (list): list of callable functions to describe 
            (e.g., lmfit.Model.func or any callable).
        operators (list of str): Operators ('+', '-', '*', '/') of length len(funcs)-1.

    Returns:
        expr (str): String representation of the composite function with arguments.
    """
    # if len(operators) != len(funcs) - 1:
    #     raise ValueError('operators must have length len(funcs)-1')

    def _format_callable(func):
        """Return a string like 'func(arg1, arg2, ...)' for a callable."""
        sig = inspect.signature(func)
        args = ', '.join(sig.parameters.keys())
        return f'{func.__name__}({args})'

    parts = []
    for func, op in itertools.zip_longest(funcs, operators, fillvalue=''):
        parts.append(_format_callable(func))
        if op:
            parts.append(op)

    return ' '.join(parts)


def wrap_expr(expr: str, width: int = 80, indent: int = 4) -> str:
    """
    Wrap a mathematical expression string at binary operators for readability.

    This is intended for pretty-printing composite model expressions
    (e.g. lmfit composite models) in logs or reports.

    Wrapping occurs only at binary operators (+, -, *, /) and preserves
    operator visibility at the start of continuation lines.

    Args:
        expr (str): Expression string to wrap.
        width (int): Maximum line width (default: 80).
        indent (int): Number of spaces to indent continuation lines (default: 4).

    Returns:
        str: Wrapped expression string.
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a string")

    if indent >= width:
        raise ValueError("indent must be smaller than width")

    # Split only on binary operators (keep operators)
    tokens = re.split(r"\s*([+\-*/])\s*", expr)

    if len(tokens) == 1:
        return expr  # nothing to wrap

    lines = []
    current = tokens.pop(0).strip()

    for op, term in zip(tokens[::2], tokens[1::2]):
        piece = f" {op} {term}"

        if len(current) + len(piece) > width:
            lines.append(current)
            current = " " * indent + f"{op} {term}"
        else:
            current += piece

    lines.append(current)
    return "\n".join(lines)


def pretty_expr(expr, line_style="#", width=80, logger=None):
    """
    Pretty-print or log a boxed expression string.

    Args:
        expr (str): Expression string to display.
        line_style (str): Character(s) used for the box border.
        width (int): Total width of the box (including borders).
        logger: Logger with an `.info()` method. If None, falls back to print().
    """
    if not isinstance(expr, str):
        raise TypeError("expr must be a string")
    
    def _out(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)

    # Wrap text to fit inside the box
    wrapped = textwrap.wrap(expr, width=width - 4)

    # Determine box width
    box_width = max(len(line) for line in wrapped) + 4

    # Top border
    _out(line_style * box_width)

    # Middle lines
    for line in wrapped:
        _out(f"{line_style} {line.ljust(box_width - 4)} {line_style}")

    # Bottom border
    _out(line_style * box_width)


def pretty_repr_params(params):
    """
    Return a pretty string representation of lmfit Parameters.

    Args:
        params (lmfit.Parameters): Parameters object (e.g. result.params).

    Returns:
        str
    """

    s = "Parameters({\n"
    for key in params.keys():
        s += f"    '{key}': {params[key]}, \n"
    s += "    })\n"
    return s


def pretty_print_params(
    params,
    *,
    logger=None,
    colwidth=8,
    precision=4,
    fmt="g",
    columns=["value", "min", "max", "stderr", "vary", "expr", "brute_step"],
):
    """
    Pretty-print lmfit Parameters.

    Args:
        params (lmfit.Parameters): Parameters object (e.g. result.params).
        logger (logging.Logger | None): Logger to use. If None, falls back to print().
        oneline (bool): If True, prints a one-line representation.
        colwidth (int): Column width.
        precision (int): Floating-point precision.
        fmt (str): Numeric format: 'g', 'e', or 'f'.
        columns (tuple[str]): Parameter attributes to display.

    Returns:
        None
    """

    def _out(msg):
        if logger is not None:
            logger.info(msg)
        else:
            print(msg, flush=True)


    name_len = max(len(s) for s in params)
    allcols = ['name'] + columns
    title = '{:{name_len}} ' + len(columns) * ' {:>{n}}'
    _out(title.format(*allcols, name_len=name_len, n=colwidth).title())

    numstyle = '{%s:>{n}.{p}{f}}'
    otherstyles = dict(
        name='{name:<{name_len}} ',
        stderr='{stderr!s:>{n}}',
        vary='{vary!s:>{n}}',
        expr='{expr!s:>{n}}',
        brute_step='{brute_step!s:>{n}}'
    )
    line = ' '.join(otherstyles.get(k, numstyle % k) for k in allcols)

    for name, values in sorted(params.items()):
        pvalues = {k: getattr(values, k) for k in columns}
        pvalues['name'] = name
        if 'stderr' in columns and pvalues['stderr'] is not None:
            pvalues['stderr'] = (numstyle % '').format(
                pvalues['stderr'], n=colwidth, p=precision, f=fmt)
        elif 'brute_step' in columns and pvalues['brute_step'] is not None:
            pvalues['brute_step'] = (numstyle % '').format(
                pvalues['brute_step'], n=colwidth, p=precision, f=fmt)
        _out(line.format(name_len=name_len, n=colwidth, p=precision, f=fmt, **pvalues))


