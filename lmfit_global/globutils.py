# %%
import re
import logging
import textwrap
import operator
import numpy as np
import functools as ft

# %%

class TqdmLoggingHandler(logging.Handler):
    """Logging handler that writes via tqdm.write to avoid breaking progress bars."""
    import tqdm
    from tqdm import tqdm
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
            

# %%
def print_in_box(expr, line_style='#', width=80):
    """Log a boxed expression using logger.info(), similar to print_in_box() above.

    Args:
        expr (str): The expression string to display.
        line_style (str, optional):  Character(s) used for the box border default to '#'.
        width (int, optional): Maximum width of the box default to 80.
    """
    # Wrap the text to the given width
    wrapped_lines = textwrap.wrap(expr, width=width-4)  # leave space for borders

    # Determine box width
    box_width = max(len(line) for line in wrapped_lines) + 4

    # Top border
    print(line_style * box_width)

    # Each line with padding, using line_style for sides
    for line in wrapped_lines:
        print(f"{line_style} {line.ljust(box_width-4)} {line_style}")

    # Bottom border
    print(line_style * box_width, flush=True)


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

    
def get_reducer(option):
    """Factory function to build a parser for complex numbers.

    Args:
        option ({'real', 'imag', 'abs', 'angle'}):  Implements the NumPy function with the same name.

    Returns:
        callable: See docstring for `reducer` below.
    """
    if option not in ['real', 'imag', 'abs', 'angle']:
        raise ValueError(f"Invalid option ('{option}') for function 'propagate_err'.")

    def reducer(array):
        """Convert a complex array to a real array.

        Several conversion methods are available and it does nothing to a
        purely real array.

        Args:
            array (array-like): Input array. If complex, will be converted to real array via
                one of the following NumPy functions: :numpydoc:`real`, 
                :numpydoc:`imag`, :numpydoc:`abs`, or :numpydoc:`angle`.

        Returns:
            ndarray: Returned array will be purely real.

        """
        array = np.asarray(array)
        if np.iscomplexobj(array):
            parsed_array = getattr(np, option)(array)
        else:
            parsed_array = array
        return parsed_array

    return reducer


def propagate_err(z, dz, option):
    """Perform error propagation on a vector of complex uncertainties.

    Required to get values for magnitude (abs) and phase (angle)
    uncertainty.

    Args:
        z (array-like): Array of complex or real numbers.
        dz (array-like): Array of uncertainties corresponding to `z`. Must satisfy
            ``numpy.shape(dz) == numpy.shape(z)``.
        option ({'real', 'imag', 'abs', 'angle'}): How to convert the array `z` to an array with real numbers.

    Returns:
        numpy.array: Returned array will be purely real.

    Notes
    -----
    Uncertainties are ``1/weights``. If the weights provided are real,
    they are assumed to apply equally to the real and imaginary parts. If
    the weights are complex, the real part of the weights are applied to
    the real part of the residual and the imaginary part is treated
    correspondingly.

    In the case where ``option='angle'`` and ``numpy.abs(z) == 0`` for any
    value of `z` the phase angle uncertainty becomes the entire circle and
    so a value of `math:pi` is returned.

    In the case where ``option='abs'`` and ``numpy.abs(z) == 0`` for any
    value of `z` the magnitude uncertainty is approximated by
    ``numpy.abs(dz)`` for that value.

    """
    z = np.asarray(z)
    dz = np.asarray(dz)

    if option not in ['real', 'imag', 'abs', 'angle']:
        raise ValueError(f"Invalid option ('{option}') for function 'propagate_err'.")

    if z.shape != dz.shape:
        raise ValueError(f"shape of z: {z.shape} != shape of dz: {dz.shape}")

    # If z is complex
    if np.iscomplexobj(z):
        # If dz is real, apply equally to real and imag parts
        if np.isrealobj(dz):
            dz = dz + 1j*dz

        if option == 'real':
            err = np.real(dz)

        elif option == 'imag':
            err = np.imag(dz)

        elif option in ['abs', 'angle']:
            rz, iz = np.real(z), np.imag(z)
            rdz, idz = np.real(dz), np.imag(dz)

            with np.errstate(divide='ignore', invalid='ignore'):
                if option == 'abs':
                    # error propagation for |z|
                    err = np.true_divide(np.sqrt((iz*idz)**2 + (rz*rdz)**2), np.abs(z))
                    # handle |z|=0
                    err[np.isinf(err)] = np.abs(dz)[np.isinf(err)]

                elif option == 'angle':
                    # error propagation for angle(z)
                    err = np.true_divide(np.sqrt((rz*idz)**2 + (iz*rdz)**2), np.abs(z)**2)
                    # handle |z|=0
                    err[np.isinf(err)] = np.pi
    else:
        # purely real case
        err = dz

    return np.asarray(err)

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


# # Create a reducer that extracts the real part
# real_reducer = get_reducer('real')

# arr1d = np.array([1+2j, 3+4j, 5])
# print('real 1D = ', real_reducer(arr1d))
# # Output: [1. 3. 5.]

# # Create a reducer that extracts the magnitude
# abs_reducer = get_reducer('abs')
# print('abs 1D = ', abs_reducer(arr1d))
# # Output: [2.23606798 5.         5.        ]

# arr2d = np.array([[1+2j, 3+4j],
#                   [5+6j, 7+8j]])

# real_reducer = get_reducer('real')
# print('real 2D = ', real_reducer(arr2d))
# # [[1. 3.]
# #  [5. 7.]]

# abs_reducer = get_reducer('abs')
# print('abs 2D = ', abs_reducer(arr2d))
# # [[2.23606798 5.        ]
# #  [7.81024968 10.63014581]]
# arrr = abs_reducer(arr2d)

# arrr
# arrr

# propagate_err(arr2d, arr2d, 'real'), arr1d.ndim

# z1 = np.array([1+2j, 3+4j])
# dz1 = np.array([0.1, 0.2])
# print(propagate_err(z1, dz1, "abs"))

# z2 = np.array([[1+2j, 3+4j],[5+6j, 7+8j]])
# dz2 = np.array([[0.1, 0.2],[0.3, 0.4]])
# print(propagate_err(z2, dz2, "abs"))

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


# %%
def reduce_with_operators(items, ops, operator_map=None):
    """Reduce a sequence of items into a composite using operators.

    Args:
        items (list): Sequence of items to combine.
        ops (list of str): Operators of length len(items)-1.
        operator_map (dict): Mapping of operator symbols to functions (e.g. {'+': operator.add}).
            Default is None

    Returns:
        obj (object): Composite result after applying operators.
    """
    if operator_map is None:
        operator_map = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            }
        
    obj = ft.reduce(
        lambda x, y: operator_map[y[1]](x, y[0]),
        zip(items[1:], ops),
        items[0]
        )
    return obj


def build_composite_model(models, operators, operator_map=None):
    """Build a composite similar to lmfit.CompositeModel from a list of models and operators
    to connect the models in the list.

    Build a composite lmfit.Model from a list of models and operators.

    Args:
    models (list of lmfit.Model): list of lmfit.Model Models to combine.
    operators (list of str): Operators ('+', '-', '*', '/') of length len(model_lst)-1.
    operator_map (dict): Mapping of operator symbols to functions (e.g. {'+': operator.add}).
        Default is None

    Returns:
    model (lmfit.CompositeModel) Composite model object.
    """

    if operator_map is None:
        operator_map = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            '/': operator.truediv,
            }

    model = reduce_with_operators(
        items=models, 
        ops=operators, 
        operator_map=operator_map
        )
    return model

# %%
def apply_ax_kws(ax, ax_kws=None):
    """Apply axis customizations from a dictionary.

    Args:
        ax (matplotlib.axes.Axes): Axis object to customize.
        ax_kws (dict, optional): Keys are method names or special strings, values are args/kwargs.
        Examples:
            {
              "figsize": None,  
              "minorticks_on": {},
              "tick_params_major": {"which":"major","direction":"in","length":8,"width":1.0,"top":True,"right":True},
              "tick_params_minor": {"which":"minor","direction":"in","length":4,"width":1.0,"top":True,"right":True},
              "tick_params_xlabels": {"axis":"x","labelsize":18,"labelcolor":"k"},
              "tick_params_ylabels": {"axis":"y","labelsize":18,"labelcolor":"k"},
              "set_xscale": {"value":"log"},
              "set_xlim": ([1.0,200],),
              "axhline": {"y":0,"color":"black","linestyle":"dotted"},
              "axvline": {"x":12,"color":"black","linestyle":"dashed"},
              "formatter": "log_plain",
              "spines": {"linewidth": 1.5, "color": "black"}
            }
    """
    import matplotlib.ticker as ticker

    if not ax_kws:
        return

    for method_name, args in ax_kws.items():
        # --- Special case: formatter ---
        if method_name == "formatter":
            if args == "log_plain":
                ax.xaxis.set_major_formatter(ticker.LogFormatter(base=10, labelOnlyBase=False))
            elif callable(args):
                ax.xaxis.set_major_formatter(args)
            continue

        # --- Tick params (major/minor/labels) ---
        if method_name in ("tick_params_major", "tick_params_minor",
                           "tick_params_xlabels", "tick_params_ylabels"):
            ax.tick_params(**args)
            continue

        # --- Spines ---
        if method_name == "spines":
            for spine in ax.spines.values():
                if "linewidth" in args:
                    spine.set_linewidth(args["linewidth"])
                if "color" in args:
                    spine.set_color(args["color"])
            continue

        # --- Figure size ---
        if method_name == "figsize":
            if args is not None:  # only apply if not None
                fig = ax.figure
                fig.set_size_inches(*args)
            continue

        # --- Normal methods ---
        if hasattr(ax, method_name):
            method = getattr(ax, method_name)
            if isinstance(args, dict):
                method(**args)
            elif isinstance(args, (list, tuple)):
                method(*args)
            else:
                method(args)


