# %%
import numpy as np
from pathlib import Path

# %%
def build_ascii_columns(
    x_data: np.ndarray,
    y_data: np.ndarray,
    x_fit: np.ndarray,
    y_fit: np.ndarray,
) -> np.ndarray:
    """
    Build a column-stacked ASCII array combining data and fitted model values.

    If data and model grids have different lengths, shorter columns are padded
    with NaNs so all columns have equal length.

    Args:
        x_data (np.ndarray): 1D array of true data
        y_data (np.ndarray): 2D array of true data with shape (N, ny).
        x_fit  (np.ndarray): 1D array of x-grid, maybe denser than x_data
        y_fit  (np.ndarray): 2D array of model evaluated on x_data/x_fit with shape (M, ny)

    Raises:
        ValueError:
            If y_data or y_fit is not a 2D array.
        ValueError:
            If y_data and y_fit have different numbers of datasets.
    Returns:
        np.ndarray:
            2D array with shape (max(N, M), 2 + 2*ny), suitable for ASCII export.
            Columns are padded with NaNs where needed.
            columns:
                x_data, x_fit, y_data0, y_fit0, y_data1, y_fit1, ...
    """
    x_data = np.asarray(x_data, float)
    x_fit  = np.asarray(x_fit, float)
    y_data = np.asarray(y_data, float)
    y_fit  = np.asarray(y_fit, float)

    if y_data.ndim != 2 or y_fit.ndim != 2:
        raise ValueError("y_data and y_fit must be 2D arrays")

    if y_data.shape[1] != y_fit.shape[1]:
        raise ValueError("y_data and y_fit must have same number of datasets")

    ny = y_data.shape[1]
    nrows = max(len(x_data), len(x_fit))

    def pad(arr, n):
        out = np.full((n,) + arr.shape[1:], np.nan)
        out[: len(arr)] = arr
        return out

    cols = [pad(x_data[:, None], nrows), pad(x_fit[:, None], nrows)]

    for j in range(ny):
        cols.append(pad(y_data[:, j:j+1], nrows))
        cols.append(pad(y_fit[:, j:j+1], nrows))

    return np.hstack(cols)


def grid_and_eval(
    x_data: np.ndarray,
    eval_func,
    params,
    numpoints: int | None = None,
    x_fit: np.ndarray | None = None,
):
    """Evaluate function of x-data grid

    Args:
        x_data (np.ndarray): 1D array of true data
        eval_func (callable): function to evaluate data, LmfitGlobal.eval()
        params (lmfit.Parameters): Parameter object to passed to eval_func
        numpoints (int | None, optional): Number of points for dense grid
            Ignored if x_fit is provided. Defaults to None.
        x_fit (np.ndarray | None, optional): Explicit x-grid. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            x_model:
                1D array of x values used for evaluation.
            y_model:
                2D array of evaluated model values with shape (len(x_model), ny).
    """
    x_data = np.asarray(x_data, float)

    if x_fit is not None:
        x_model = np.asarray(x_fit, float)

    elif numpoints is not None and numpoints > x_data.size:
        x_model = np.linspace(x_data.min(), x_data.max(), numpoints)

    else:
        x_model = x_data

    y_model = eval_func(x=x_model, params=params)

    return x_model, y_model

def export_ascii(
    lg,
    filename: str,
    *,
    numpoints: int | None = None,
    x_fit: np.ndarray | None = None,
    header: bool = True,
    fmt: str = "% .8e",
):
    """
    Export data and fitted model to an ASCII column file.

    The output columns are arranged as::

        x_data | x_fit | y_data[0] | y_fit[0] | y_data[1] | y_fit[1] | ...

    Shorter arrays are padded with NaNs to form a rectangular table
    (compatible with musrfit-style ASCII files).

    Args:
        lg:
            LmfitGlobal instance (already fitted).
        filename (str):
            Output file path.
        numpoints (int | None, optional):
            Number of points for a dense fit grid.
            If None, uses data grid.
        x_fit (np.ndarray | None, optional):
            Explicit x-grid for model evaluation.
            Overrides numpoints if provided.
        header (bool, optional):
            If True, write a descriptive header.
        fmt (str, optional):
            Numeric format passed to np.savetxt.
    """
    if not getattr(lg, "fit_success", False):
        raise RuntimeError("Cannot export: fit has not been performed successfully")

    # Obtain FitData (single authoritative source)
    fitdata = lg.get_fitdata(numpoints=numpoints)

    x_data = np.asarray(fitdata.x_data, float)
    y_data = np.asarray(fitdata.y_data, float)

    if fitdata.has_fit:
        x_model = np.asarray(fitdata.x_model, float)
        y_fit = np.asarray(fitdata.y_fit, float)
    else:
        raise RuntimeError("FitData has no fitted model")

    # Optional override of x_fit
    if x_fit is not None:
        x_model = np.asarray(x_fit, float)
        y_fit = lg.eval(x=x_model, params=lg.result.params)

    if y_data.ndim != 2 or y_fit.ndim != 2:
        raise ValueError("y_data and y_fit must be 2D arrays")

    if y_data.shape[1] != y_fit.shape[1]:
        raise ValueError("Mismatch in number of datasets between data and fit")

    ny = y_data.shape[1]
    nrows = max(len(x_data), len(x_model))

    # Padding helper
    def pad(arr, n):
        out = np.full((n,) + arr.shape[1:], np.nan)
        out[: len(arr)] = arr
        return out

    # Build column matrix
    columns = [
        pad(x_data[:, None], nrows),
        pad(x_model[:, None], nrows),
    ]

    for j in range(ny):
        columns.append(pad(y_data[:, j:j + 1], nrows))
        columns.append(pad(y_fit[:, j:j + 1], nrows))

    table = np.hstack(columns)

    # Header
    hdr = None
    if header:
        # names = ["xData", "xTheory"]
        names = ["xData", "xFit"]
        for j in range(ny):
            # names += [f"data{j}", f"theory{j}"]
            names += [f"Data{j}", f"yFit{j}"]
        hdr = ", ".join(names)

    # Write file
    filename = Path(filename)
    np.savetxt(
        filename,
        table,
        delimiter=", ",
        header=hdr if hdr else "",
        # comments="",
        fmt=fmt,
    )


def ascii_header(ny: int) -> str:
    cols = ["xData", "xTheory"]
    for j in range(ny):
        cols += [f"data{j}", f"theory{j}"]
    return ", ".join(cols)