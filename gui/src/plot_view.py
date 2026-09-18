# %%
from __future__ import annotations

import numpy as np
import streamlit as st

from lmfit_global.utils.plotting import FitPlotter
from gui.src.utils import component_label, render_fancy_header

# %%
def render_plot(
    lg,
    ny: int,
    dataset_labels: list[str],
    component_choices: list[str],
    x_min_eval: float,
    x_max_eval: float,
    n_points_eval: int,
):
    """Renders the plot display controls and the fit figure itself
    (data + fit + optional residuals + component overlay).

    Returns:
        (fig, fitdata, x_model_custom, dpi_val) -- everything the export
        section needs afterward.
    """
    render_fancy_header(
        title="Plot settings", step_number=None,
        title_color="#7dd3fc", title_size="0.95rem",
        title_margin="0.1rem 0 0.4rem",
    )

    with st.container(border=True):
        # Residuals + DPI first (structural choices), legend pair last and
        # adjacent to each other since "show residual legend" only makes
        # sense in the context of "show legend" / "show residuals".
        # vcol1, vcol2, vcol3, vcol4 = st.columns([1, 1, 1, 1])
        vcol1, vcol2, vcol3 = st.columns(3)

        show_m_leg = vcol1.checkbox("🏷️ Show legend", value=True)
        show_res = vcol2.checkbox("📉 Show residuals", value=False)
        show_r_leg = vcol3.checkbox(
            "🏷️ Show residual legend", value=False,
            disabled=not show_res,
            help="Only applies when 'Show Residuals' is enabled.",
        )

    # Re-evaluate internal model grid for current resolution
    if x_max_eval <= x_min_eval:
        st.error("x-max must be greater than x-min in the Evaluation Grid settings.")
        st.stop()
    x_model_custom = np.linspace(x_min_eval, x_max_eval, int(n_points_eval))

    lg.x_model = x_model_custom
    fitdata = lg.get_fitdata()
    fitdata.x_model = x_model_custom
    fitdata.y_fit = lg.eval(x=x_model_custom)
    fd = fitdata

    # Generate Plot
    plotter = FitPlotter(fitdata)

    pretty_kw = {
        "width": 8.0,
        "height": 5.0 if show_res else 5.0,
        "dpi": 100
    }

    ax_main, ax_res, fig = plotter.make_axes(
        plotwhat="fit",
        plot_residual=show_res,
        pretty_kw=pretty_kw
    )

    plotter._plot(
        plotwhat="fit",
        ax=ax_main,
        ax_res=ax_res,
        plot_residual=show_res,
        show_legend=show_m_leg,
        show_resid_legend=show_r_leg,
        show=False,
        xlim=(x_min_eval, x_max_eval),
        fit_kws={"linewidth": 1.5},
        data_kws={"markersize": 4, "alpha": 0.7},
        resid_kws={"markersize": 3, "alpha": 0.7}
    )

    # Deconstruct multi-component lines on plot
    if lg.is_multicomponent:
        comps = lg.eval_components(x_model=x_model_custom)
        for comp_name, comp_data in comps.items():
            comp_lbl = component_label(comp_name, component_choices)

            if isinstance(comp_data, dict):
                for j in range(ny):
                    c_vals = comp_data.get(j, comp_data.get(f"ds_{j}"))
                    if c_vals is not None:
                        ax_main.plot(
                            x_model_custom, c_vals, ":",
                            color=plotter.colors[j % len(plotter.colors)],
                            alpha=0.7, label=f"{dataset_labels[j]}: {comp_lbl}"
                        )
            elif isinstance(comp_data, np.ndarray):
                if comp_data.ndim == 2 and comp_data.shape[1] == ny:
                    for j in range(ny):
                        ax_main.plot(
                            x_model_custom, comp_data[:, j], ":",
                            color=plotter.colors[j % len(plotter.colors)],
                            alpha=0.7, label=f"{dataset_labels[j]}: {comp_lbl}"
                        )
                else:
                    ax_main.plot(x_model_custom, comp_data.ravel(), ":", alpha=0.7, label=comp_lbl)

    # Legend Styling & Entry Cap -- runs regardless of component count, so
    # single- and multi-component fits get identically styled legends.
    if show_m_leg:
        MAX_LEGEND_ENTRIES = 20
        handles, labels = ax_main.get_legend_handles_labels()
        legend_kws = dict(
            loc="best", frameon=True, edgecolor="black",
            fontsize="small", markerscale=0.8,
            ncol=2 if len(labels) > 6 else 1,
        )
        if len(labels) > MAX_LEGEND_ENTRIES:
            # ax_main.legend(
            #     handles[:MAX_LEGEND_ENTRIES], labels[:MAX_LEGEND_ENTRIES],
            #     loc="best", frameon=True, edgecolor="black", fontsize="small",
            #     title=f"showing {MAX_LEGEND_ENTRIES}/{len(labels)} entries"
            # )
            ax_main.legend(
                handles[:MAX_LEGEND_ENTRIES], labels[:MAX_LEGEND_ENTRIES],
                title=f"showing {MAX_LEGEND_ENTRIES}/{len(labels)} entries",
                **legend_kws,
            )
        else:
            # ax_main.legend(loc="best", frameon=True, edgecolor="black")
            ax_main.legend(**legend_kws)
    elif ax_main.get_legend() is not None:
        ax_main.get_legend().remove()

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    return fig, fitdata, x_model_custom