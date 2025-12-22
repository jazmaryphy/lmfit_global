# %%
"""Utilities for generating nicer plots.

https://github.com/materialsproject/pymatgen/blob/master/src/pymatgen/util/plotting.py

"""

from __future__ import annotations

import math
import logging
import importlib
import numpy as np
import functools as ft
from functools import wraps
import matplotlib.pyplot as plt
from string import ascii_letters
from typing import TYPE_CHECKING
# import palettable.colorbrewer.diverging

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

logger = logging.getLogger(__name__)

# %%
# Optional imports
try:
    import matplotlib 
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False
    
def _ensureMatplotlib(function):
    if _HAS_MATPLOTLIB:
        @ft.wraps(function)
        def wrapper(*args, **kws):
            return function(*args, **kws)
        return wrapper

    def no_op(*args, **kwargs):
        print('matplotlib module is required for plotting the results')
    return no_op

# %%
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

# %%
@_ensureMatplotlib
def pretty_plot(
    width: float = 8,
    height: float | None = None,
    ax: Axes | None = None,
    dpi: float | None = 100,
    color_cycle: tuple[str, str] = ("qualitative", "Set1_9"),
) -> Axes:
    """Get a publication quality plot, with nice defaults for font sizes etc.

    Args:
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        ax (Axes): If ax is supplied, changes will be made to an
            existing plot. Otherwise, a new plot will be created.
        dpi (float): Sets dot per inch for figure. Defaults to 300.
        color_cycle (tuple): Set the color cycle for new plots to one of the
            color sets in palettable. Defaults to a qualitative Set1_9.

    Returns:
        Axes: matplotlib axes object with properly sized fonts.
    """

    tick_size  = int(width * 2.5)
    label_size = int(width * 3)
    title_size = int(width * 4)

    golden_ratio = (np.sqrt(5) - 1) / 2
    if height is None:
        height = width * golden_ratio

    if ax is None:
        # --- Try palettable, fallback to tab10 ---
        from cycler import cycler
        colors = None
        try:
            mod = importlib.import_module(
                f"palettable.colorbrewer.{color_cycle[0]}"
            )
            colors = getattr(mod, color_cycle[1]).mpl_colors
        except Exception:
            cmap = plt.get_cmap("tab10")
            if hasattr(cmap, "colors"):
                colors = cmap.colors
            else:
                colors = [cmap(i) for i in np.linspace(0, 1, 10)]

        plt.figure(figsize=(width, height), facecolor="w", dpi=dpi)
        ax = plt.gca()
        ax.set_prop_cycle(cycler("color", colors))
    else:
        # fig = plt.gcf()
        # fig.set_size_inches(width, height)
        fig = ax.figure
        fig.set_size_inches(width, height)
        if dpi is not None:
            fig.set_dpi(dpi)

    ax.tick_params(labelsize=tick_size)
    ax.set_title(ax.get_title(), fontsize=title_size)
    ax.set_xlabel(ax.get_xlabel(), fontsize=label_size)
    ax.set_ylabel(ax.get_ylabel(), fontsize=label_size)

    return ax


@_ensureMatplotlib
def pretty_plot_two_axis(
    x,
    y1,
    y2,
    xlabel=None,
    y1label=None,
    y2label=None,
    width: float = 8,
    height: float | None = None,
    dpi=300,
    **plot_kwargs,
):
    """Variant of pretty_plot that does a dual axis plot. Adapted from matplotlib
    examples. Makes it easier to create plots with different axes.

    Args:
        x (Sequence[float]): Data for x-axis.
        y1 (Sequence[float] | dict[str, Sequence[float]]): Data for y1 axis (left). If a dict, it will
            be interpreted as a {label: sequence}.
        y2 (Sequence[float] | dict[str, Sequence[float]]): Data for y2 axis (right). If a dict, it will
            be interpreted as a {label: sequence}.
        xlabel (str): If not None, this will be the label for the x-axis.
        y1label (str): If not None, this will be the label for the y1-axis.
        y2label (str): If not None, this will be the label for the y2-axis.
        width (float): Width of plot in inches. Defaults to 8in.
        height (float): Height of plot in inches. Defaults to width * golden
            ratio.
        dpi (int): Sets dot per inch for figure. Defaults to 300.
        plot_kwargs: Passthrough kwargs to matplotlib's plot method. e.g.
            linewidth, etc.

    Returns:
        plt.Axes: matplotlib axes object with properly sized fonts.
    """
    try:
        mod = importlib.import_module(
            "palettable.colorbrewer.diverging"
        )
        colors = mod.RdYlBu_4.mpl_colors
        c1, c2 = colors[0], colors[-1]
    except Exception:
        cmap = plt.get_cmap("RdYlBu")
        c1 = cmap(0.1)
        c2 = cmap(0.9)

        golden_ratio = (math.sqrt(5) - 1) / 2

    # width = 12

    if not height:
        height = int(width * golden_ratio)

    label_size = int(width * 3)
    tick_size = int(width * 2.5)
    styles = ["-", "--", "-.", "."]

    fig, ax1 = plt.subplots()
    fig.set_size_inches((width, height))
    if dpi:
        fig.set_dpi(dpi)
    if isinstance(y1, dict):
        for idx, (key, val) in enumerate(y1.items()):
            ax1.plot(
                x,
                val,
                c=c1,
                marker="s",
                ls=styles[idx % len(styles)],
                label=key,
                **plot_kwargs,
            )
        ax1.legend(fontsize=label_size)
    else:
        ax1.plot(x, y1, c=c1, marker="s", ls="-", **plot_kwargs)

    if xlabel:
        ax1.set_xlabel(xlabel, fontsize=label_size)

    if y1label:
        # Make the y-axis label, ticks and tick labels match the line color.
        ax1.set_ylabel(y1label, color=c1, fontsize=label_size)

    ax1.tick_params("x", labelsize=tick_size)
    ax1.tick_params("y", colors=c1, labelsize=tick_size)

    ax2 = ax1.twinx()
    if isinstance(y2, dict):
        for idx, (key, val) in enumerate(y2.items()):
            ax2.plot(x, val, c=c2, marker="o", ls=styles[idx % len(styles)], label=key)
        ax2.legend(fontsize=label_size)
    else:
        ax2.plot(x, y2, c=c2, marker="o", ls="-")

    if y2label:
        # Make the y-axis label, ticks and tick labels match the line color.
        ax2.set_ylabel(y2label, color=c2, fontsize=label_size)

    ax2.tick_params("y", colors=c2, labelsize=tick_size)
    return ax1


@_ensureMatplotlib
def get_ax_fig(ax: Axes = None, **kwargs) -> tuple[Axes, Figure]:
    """Helper function used in plot functions supporting an optional Axes argument.
    If ax is None, we build the `matplotlib` figure and create the Axes else
    we return the current active figure.

    Args:
        ax (Axes, optional): Axes object. Defaults to None.
        kwargs: keyword arguments are passed to plt.figure if ax is not None.

    Returns:
        tuple[Axes, Figure]: matplotlib Axes object and Figure objects
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.gca()
    else:
        fig = plt.gcf()

    return ax, fig


@_ensureMatplotlib
def get_ax3d_fig(ax: Axes = None, **kwargs) -> tuple[Axes3D, Figure]:
    """Helper function used in plot functions supporting an optional Axes3D
    argument. If ax is None, we build the `matplotlib` figure and create the
    Axes3D else we return the current active figure.

    Args:
        ax (Axes3D, optional): Axes3D object. Defaults to None.
        kwargs: keyword arguments are passed to plt.figure if ax is not None.

    Returns:
        tuple[Axes3D, Figure]: matplotlib Axes3D and corresponding figure objects
    """
    if ax is None:
        fig = plt.figure(**kwargs)
        ax = fig.add_subplot(projection="3d")
    else:
        fig = plt.gcf()

    return ax, fig


@_ensureMatplotlib
def get_axarray_fig_plt(
    ax_array,
    nrows=1,
    ncols=1,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw,
):
    """Helper function used in plot functions that accept an optional array of Axes
    as argument. If ax_array is None, we build the `matplotlib` figure and
    create the array of Axes by calling plt.subplots else we return the
    current active figure.

    Returns:
        ax: Array of Axes objects
        figure: matplotlib figure
        plt: matplotlib pyplot module.
    """
    if ax_array is None:
        fig, ax_array = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            sharex=sharex,
            sharey=sharey,
            squeeze=squeeze,
            subplot_kw=subplot_kw,
            gridspec_kw=gridspec_kw,
            **fig_kw,
        )
    else:
        # fig = plt.gcf()
        # ax_array = np.reshape(np.array(ax_array), (nrows, ncols))
        # if squeeze:
        #     if ax_array.size == 1:
        #         ax_array = ax_array[0]
        #     elif any(s == 1 for s in ax_array.shape):
        #         ax_array = ax_array.ravel()

        ax_array = np.asarray(ax_array)
        ax_array = ax_array.reshape(nrows, ncols)
        if squeeze:
            if ax_array.size == 1:
                ax_array = ax_array[0]
            elif 1 in ax_array.shape:
                ax_array = ax_array.ravel()

        fig = ax_array.flat[0].figure  # safer than plt.gcf()

    return ax_array, fig, plt


@_ensureMatplotlib
def get_pretty_axarray(
    ax_array=None,
    nrows: int = 1,
    ncols: int = 1,
    sharex: bool = False,
    sharey: bool = False,
    squeeze: bool = True,
    pretty_kw: dict | None = None,
    subplot_kw=None,
    gridspec_kw=None,
    **fig_kw,
):
    """
    Create or reshape an array of Axes and apply pretty_plot() styling.

    Parameters
    ----------
    ax_array : Axes or array-like, optional
        Existing axes. If None, new subplots are created.
    nrows, ncols : int
        Grid dimensions.
    pretty_kw : dict, optional
        Keyword arguments forwarded to pretty_plot().
    subplot_kw : dict, optional
        Passed to plt.subplots().
    gridspec_kw : dict, optional
        Passed to plt.subplots() for GridSpec configuration.
    **fig_kw :
        Additional figure keyword arguments (e.g., figsize, dpi).

    Returns
    -------
    ax_array : Axes or ndarray
    fig : Figure
    plt : module
    """

    pretty_kw = pretty_kw or {}

    # Create or reshape axes
    ax_array, fig, plt_mod = get_axarray_fig_plt(
        ax_array=ax_array,
        nrows=nrows,
        ncols=ncols,
        sharex=sharex,
        sharey=sharey,
        squeeze=squeeze,
        subplot_kw=subplot_kw,
        gridspec_kw=gridspec_kw,
        **fig_kw,
    )

    # Apply pretty styling
    if isinstance(ax_array, np.ndarray):
        for ax in ax_array.flat:
            pretty_plot(ax=ax, **pretty_kw)
    else:
        pretty_plot(ax=ax_array, **pretty_kw)

    return ax_array, fig, plt_mod


def add_fig_kwargs(func):
    """Decorator that adds keyword arguments for functions returning matplotlib
    figures.

    The function should return either a matplotlib figure or None to signal
    some sort of error/unexpected event.
    See doc string below for the list of supported options.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # pop the kwds used by the decorator.
        title = kwargs.pop("title", None)
        size_kwargs = kwargs.pop("size_kwargs", None)
        show = kwargs.pop("show", True)
        savefig = kwargs.pop("savefig", None)
        tight_layout = kwargs.pop("tight_layout", False)
        ax_grid = kwargs.pop("ax_grid", None)
        ax_annotate = kwargs.pop("ax_annotate", None)
        fig_close = kwargs.pop("fig_close", False)

        # Call func and return immediately if None is returned.
        fig = func(*args, **kwargs)
        if fig is None:
            return fig

        # Operate on matplotlib figure.
        if title is not None:
            fig.suptitle(title)

        if size_kwargs is not None:
            fig.set_size_inches(size_kwargs.pop("w"), size_kwargs.pop("h"), **size_kwargs)

        if ax_grid is not None:
            for ax in fig.axes:
                ax.grid(bool(ax_grid))

        if ax_annotate:
            tags = ascii_letters
            if len(fig.axes) > len(tags):
                tags = (1 + len(ascii_letters) // len(fig.axes)) * ascii_letters
            for ax, tag in zip(fig.axes, tags, strict=True):
                ax.annotate(f"({tag})", xy=(0.05, 0.95), xycoords="axes fraction")

        if tight_layout:
            try:
                fig.tight_layout()
            except Exception:
                # For some unknown reason, this problem shows up only on travis.
                # https://stackoverflow.com/questions/22708888/valueerror-when-using-matplotlib-tight-layout
                # pass
                logger.exception("Ignoring Exception raised by fig.tight_layout\n")

        if savefig:
            fig.savefig(savefig)

        if show:
            plt.show()
        if fig_close:
            plt.close(fig=fig)

        return fig

    # Add docstring to the decorated method.
    doc_str = """\n\n
        Keyword arguments controlling the display of the figure:

        ================  ====================================================
        kwargs            Meaning
        ================  ====================================================
        title             Title of the plot (Default: None).
        show              True to show the figure (default: True).
        savefig           "abc.png" or "abc.eps" to save the figure to a file.
        size_kwargs       Dictionary with options passed to fig.set_size_inches
                          e.g. size_kwargs=dict(w=3, h=4)
        tight_layout      True to call fig.tight_layout (default: False)
        ax_grid           True (False) to add (remove) grid from all axes in fig.
                          Default: None i.e. fig is left unchanged.
        ax_annotate       Add labels to subplots e.g. (a), (b).
                          Default: False
        fig_close         Close figure. Default: False.
        ================  ====================================================
        """

    if wrapper.__doc__ is not None:
        # Add s at the end of the docstring.
        wrapper.__doc__ += f"\n{doc_str}"
    else:
        # Use s
        wrapper.__doc__ = doc_str

    return wrapper

# %%
def plot_from_fitdata(
    fitdata,
    *,
    ax,
    plotwhat: str,
    fmt: str,
    label: str,
    yerr=None,
    parse_complex="abs",
    zorder=1,
    plot_kws=None,
):
    """
    Generic plotting helper using FitData.

    Args:
        fitdata (FitData)
        ax (matplotlib.axes.Axes)
        plotwhat ('data' | 'init' | 'fit' | 'resid')
        fmt (str): matplotlib format string
        label (str): label
        yerr (ndarray, optional):
            Must have the same shape as fitdata.y_data
        parse_complex (str)
        zorder (int)
        plot_kws (dict, optional)
    """
    if plot_kws is None:
        plot_kws = {}

    reducer = get_reducer(parse_complex)

    DATA_MAP = {
        "data":  lambda fd: (fd.x_data,  fd.y_data),
        "init":  lambda fd: (fd.x_model, fd.y_init),
        "fit":   lambda fd: (fd.x_model, fd.y_fit),
        "resid": lambda fd: (fd.x_data,  fd.resid_fit),
    }

    if plotwhat not in DATA_MAP:
        raise ValueError(
            f"Invalid plotwhat='{plotwhat}'. "
            f"Expected one of {tuple(DATA_MAP)}"
        )

    x, y = DATA_MAP[plotwhat](fitdata)

    if y is None:
        return

    # y = reduce_creducer omplex(y)

    # if yerr is not None:
    #     yerr = np.asarray(yerr)
    #     if yerr.shape != y.shape:
    #         raise ValueError(
    #             f"yerr shape {yerr.shape} does not match y shape {y.shape}"
    #         )

    # --- Strict yerr validation ---
    if yerr is not None:
        yerr = np.asarray(yerr)
        if yerr.shape != fitdata.y_data.shape:
            raise ValueError(
                "yerr must have the same shape as fitdata.y_data: "
                f"{yerr.shape} != {fitdata.y_data.shape}"
            )
    
    # ny = y.shape[1] if y.ndim > 1 else 1

    # for i in range(ny):
    #     yi = y[:, i] if ny > 1 else y
    #     lbl = f"{label}{i}" if ny > 1 else label

    #     if yerr is not None and plotwhat in ("data", "resid"):
    #         err = propagate_err(y, yerr, parse_complex)
    #         ei = err[:, i] if ny > 1 else err
    #         ax.errorbar(x, yi, yerr=ei, fmt=fmt, label=lbl, zorder=zorder, **plot_kws)
    #     else:
    #         ax.errorbar(x, yi, yerr=None, fmt=fmt, label=lbl, zorder=zorder, **plot_kws)
    #         # ax.plot(x, yi, fmt, label=lbl, zorder=zorder, **plot_kws)


    ny = y.shape[1] if y.ndim > 1 else 1

    for i in range(ny):
        yi_raw = y[:, i] if ny > 1 else y
        yi = reducer(yi_raw)

        lbl = f"{label}{i}" if ny > 1 else label

        if yerr is not None and plotwhat in ("data", "resid"):
            err_i = yerr[:, i] if ny > 1 else yerr
            err_i = propagate_err(yi_raw, err_i, parse_complex)

            ax.errorbar(
                x, yi, yerr=err_i,
                fmt=fmt, label=lbl,
                zorder=zorder, **plot_kws
            )
        else:
            ax.plot(
                x, yi, fmt,
                label=lbl, zorder=zorder,
                **plot_kws
            )

# %%
_PLOT_RULES = {
    "data": {"fmt": "o", "overlay_data": False, "zorder": 1,},
    "init": {"fmt": "--", "overlay_data": True, "zorder": 2,},
    "fit": {"fmt": "-", "overlay_data": True, "zorder": 3,},
    "resid": {"fmt": "o", "overlay_data": False,"zorder": 2,},
}

class FitPlotter:
    """
    Handles all plotting for FitData objects:
    - figure creation
    - pretty styling
    - data/init/fit/residual layers
    - subplot layout
    """

    def __init__(self, fitdata):
        self.fitdata = fitdata
        self.rules = _PLOT_RULES

    # ------------------------------------------------------------
    # Figure / axes creation
    # ------------------------------------------------------------
    def make_axes(self, plotwhat, plot_residual, pretty_kw=None):
        if plotwhat == "resid" or plot_residual:
            pretty_kw = pretty_kw or {"width": 8, "height": 8, "dpi": 100}
            ax, fig, _ = get_pretty_axarray(
                nrows=2,
                ncols=1,
                sharex=True,
                gridspec_kw={"height_ratios": [1, 4]},
                pretty_kw=pretty_kw,
            )
            ax_res, ax_main = ax
        else:
            pretty_kw = pretty_kw or {"width": 8, "height": 6, "dpi": 100}
            ax_main, fig, _ = get_pretty_axarray(
                nrows=1, ncols=1, pretty_kw=pretty_kw,
            )
            ax_res = None

        return ax_main, ax_res, fig

    # ------------------------------------------------------------
    # Core plotting logic
    # ------------------------------------------------------------
    def _overlay_data(self, plotwhat: str) -> bool:
        return _PLOT_RULES[plotwhat]["overlay_data"]
    
    def _plot(
        self, 
        plotwhat, 
        *, 
        ax=None, 
        yerr=None,
        xlabel=None, 
        ylabel=None, 
        xlim=None,
        ylim=None,
        plot_residual=True,
        show=True,
        data_kws=None, 
        init_kws=None, 
        fit_kws=None, 
        resid_kws=None,
        pretty_kw=None
    ):

        data_kws  = data_kws  or {}
        init_kws  = init_kws  or {}
        fit_kws   = fit_kws   or {}
        resid_kws = resid_kws or {}

        if plotwhat not in self.rules:
            raise ValueError(f"Invalid plotwhat '{plotwhat}'")

        fitdata = self.fitdata
        rules = self.rules[plotwhat]

        # Axes creation
        if ax is None:
            ax_main, ax_res, fig = self.make_axes(
                plotwhat, plot_residual, pretty_kw
            )
        else:
            ax_main = ax
            ax_res = None

        # Overlay data if needed
        if rules["overlay_data"]:
            plot_from_fitdata(
                fitdata, ax=ax_main, plotwhat="data",
                fmt=self.rules["data"]["fmt"],
                label="data", yerr=yerr,
                zorder=self.rules["data"]["zorder"],
                plot_kws=data_kws,
            )

        # Main layer
        plot_from_fitdata(
            fitdata, ax=ax_main, plotwhat=plotwhat,
            fmt=rules["fmt"], label=plotwhat,
            yerr=yerr, zorder=rules["zorder"],
            plot_kws={"data": data_kws,
                      "init": init_kws,
                      "fit": fit_kws,
                      "resid": resid_kws}[plotwhat],
        )

        # Residuals
        if plot_residual and plotwhat != "resid" and ax_res is not None:
            plot_from_fitdata(
                fitdata, ax=ax_res, plotwhat="resid",
                fmt=self.rules["resid"]["fmt"],
                label="resid", yerr=yerr,
                plot_kws=resid_kws,
            )
            ax_res.axhline(0, color="k", lw=1)

        # Labels
        if xlabel:
            ax_main.set_xlabel(xlabel)
        if ylabel:
            ax_main.set_ylabel(ylabel)

        if xlim:
            ax_main.set_xlim(xlim)
        if ylim:
            ax_main.set_ylim(ylim)

        ax_main.legend()
        if ax_res:
            ax_res.legend()
            ax_res.set_ylabel('resid.')
            if xlim:
                ax_res.set_xlim(xlim)

        plt.tight_layout() 

        if show:
            plt.show()

        return ax_main

# %%
# pretty_plot(width=8, height=6, dpi=100)
# x=np.linspace(0, 100)
# y1=np.sin(x)
# y2=np.cos(x)


# # pretty_plot_two_axis(x=x, y1=y1, y2=y2, width=6.5, height=6.5, dpi=100)

# gridspec_kw={
#     # 'width_ratios': [1.5, 1.2],
#     'height_ratios': [1.0, 4]
#     }

# # ax_array, fig, plt = get_axarray_fig_plt(
# #     ax_array=None,
# #     nrows=2,
# #     ncols=1,
# #     sharex=True,
# #     sharey= False,
# #     squeeze= True,
# #     subplot_kw=None,
# #     gridspec_kw=gridspec_kw,
# #     # **fig_kw,
# # )

# # width, height = ax_array[0].figure.get_size_inches()
# # width, height

# ax, fig, plt = get_pretty_axarray(
#     nrows=2, ncols=1,
#     sharex=True, sharey=False,
#     gridspec_kw=gridspec_kw,
#     pretty_kw={"width": 8, "height":8}
# )
# ax
# ax[0].plot(x, y1)
# ax[1].plot(x, y2)


# ax, fig, plt = get_pretty_axarray(
#     nrows=1, ncols=1,
#     sharex=False, sharey=False,
#     gridspec_kw=None,
#     pretty_kw={"width": 8, "height":6, "dpi":100}
# )

# %%



