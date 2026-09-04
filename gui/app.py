# %%
from __future__ import annotations

import io
import contextlib
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gui.library import FUNCTION_LIBRARY, CONNECTORS
from lmfit_global import LmfitGlobal
from lmfit_global.utils.plotting import FitPlotter

from gui.src.data_loader import render_data_sidebar
from gui.src.parameter_editor import render_parameter_editor
from gui.src.utils import (component_label, component_short_tag, 
                           sanitize_label, to_fixed_width, render_fancy_header
)

# %%
# Page Configuration
# st.set_page_config(
#     page_title="lmfitgedit — lmfit-global Studio", 
#     layout="wide"
# )
st.set_page_config(
     page_title="lmfit-global GUI",
    # page_title="lmfit-global Studio", 
    # page_icon="📈",
    layout="wide"
)
# st.title("`lmfitgedit` — Multi-Dataset Fitting Interface")
st.title("`lmfitgedit` — Interactive Multi-Dataset / Multi-Component Fitting")
st.caption(
    "`lmfitgedit`: The GUI Based Interface to `LmfitGlobal` class from "
    "[jazmaryphy/lmfit_global](https://github.com/jazmaryphy/lmfit_global)."
)

# %%
# 1. Data Input Sidebar
#
xy, dataset_labels, export_labels, source = render_data_sidebar()
if xy is None or xy.shape[1] <= 1:
    st.info("Upload data or select a demo preset to proceed.")
    st.stop()

ny = xy.shape[1] - 1

# Invalidate stale fit state if data signature changes
data_sig = (xy.shape, float(np.nansum(xy)))
if st.session_state.get("_data_signature") != data_sig:
    st.session_state.pop("fitted_lg", None)
    st.session_state["_data_signature"] = data_sig

# 2. Sidebar: Model Configuration
with st.sidebar:
    # st.header("2. Model Construction")
    render_fancy_header(
        title="Model Construction", 
        step_number=2, 
        level=2, 
        title_color="#38bdf8"  # Universal Electric Blue
    )
    n_components = st.number_input("Number of components", min_value=1, max_value=6, value=1, step=1)

    component_choices, connectors, all_selected = [], [], True
    for i in range(n_components):
        fname = st.selectbox(f"Component {i+1}", list(FUNCTION_LIBRARY.keys()), index=None, key=f"func_{i}")
        if fname is None:
            all_selected = False
        else:
            component_choices.append(fname)
        if i > 0:
            connectors.append(st.selectbox(f"Connector {i+1}", CONNECTORS, key=f"conn_{i}"))

    # st.header("3. Evaluation Grid")
    render_fancy_header(
        title="X-Grid", 
        step_number=3, 
        level=2, 
        title_color="#38bdf8"  # Universal Electric Blue
    )
    x_min_eval = st.number_input("x-min", value=float(np.nanmin(xy[:, 0])), format="%.4f")
    x_max_eval = st.number_input("x-max", value=float(np.nanmax(xy[:, 0])), format="%.4f")
    n_points_eval = st.number_input("numpoints (N)", min_value=50, max_value=10000, value=500, step=50)

if not all_selected:
    st.info("Please choose a function for every component in the sidebar.")
    st.stop()

# %%
# 3. Parameter Editors
#
param_df = render_parameter_editor(xy, component_choices)

# 4. Global Linkings
#
global_param_bases = []
if ny > 1:
    # st.header("5. Shared Parameters Across Datasets")
    render_fancy_header(
        title="Shared Parameters Across Datasets", 
        step_number=5, 
        level=2, 
        title_color="#38bdf8"  # Universal Electric Blue
    )
    unique_params = sorted(set(param_df["Parameter"].tolist()))
    global_param_bases = st.multiselect("Select base parameters to link globally:", unique_params)

# %%
# 5. Fit Execution
#
# st.header("6. Optimization & Results")
render_fancy_header(
    title="Optimization & Results", 
    step_number=6, 
    level=2, 
    title_color="#38bdf8"  # Universal Electric Blue
)

runfit_str = "Fit"
if st.button(runfit_str, type="primary"):
    func_lst = []
    for c_idx, fname in enumerate(component_choices):
        spec = FUNCTION_LIBRARY[fname]
        comp_df = param_df[
            (param_df["Component_ID"] == c_idx) & (param_df["Dataset_ID"] == 0)
        ]

        init_params = {}
        for _, row in comp_df.iterrows():
            init_params[row["Parameter"]] = {
                "value": row["Value"],
                "vary": bool(row["Vary"]),
                "min": row["Min"],
                "max": row["Max"],
            }

        func_lst.append({
            "func_name": spec["func"],
            "init_params": init_params,
            "func_kws": {},
        })

    try:
        items = {
            "data": {"xy": xy, "xrange": None},
            "functions": {
                "theory": func_lst,
                "theory_connectors": connectors if connectors else None,
            },
        }
        lg = LmfitGlobal(items=items, log_level="warning")
        lg.rebuild()

        override_params = {}
        for _, row in param_df.iterrows():
            val = row["Value"]
            parsed_val = float(val) if isinstance(val, (int, float)) else val

            override_params[row["Target_Key"]] = {
                "value": parsed_val,
                "vary": bool(row["Vary"]),
                "min": float(row["Min"]),
                "max": float(row["Max"]),
            }

        lg.update_params(override_params, overwrite_expr=True)

        if ny > 1 and global_param_bases:
            for base in global_param_bases:
                for c_idx in range(n_components):
                    parlist = [f"{base}_{j}" if n_components == 1 else f"c{c_idx}_{base}_{j}" for j in range(ny)]
                    if parlist[0] in lg.init_params:
                        lg.set_global(parlist, reference=parlist[0], overwrite_expr=True)

        lg.fit(verbose=False)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            lg.report()

        st.session_state["fitted_lg"] = lg
        st.session_state["report_text"] = buf.getvalue()
        st.success("Optimization completed successfully.")
    except Exception as e:
        st.error(f"Fit failed: {e}")

# %%
# 6. Plotting & Results Section
if "fitted_lg" in st.session_state:
    lg = st.session_state["fitted_lg"]
    report_text = st.session_state["report_text"]

    # st.subheader("Fit Report")
    render_fancy_header(
        "Fit Report", 
        level=3, 
    )
    st.code(report_text, language="text")

    # Display Controls
    vcol1, vcol2, vcol3, vcol4 = st.columns(4)
    show_res = vcol1.checkbox("Show Residuals", value=False)
    dpi_val = vcol2.number_input("DPI Resolution", min_value=100, max_value=600, value=100, step=50)
    show_m_leg = vcol3.checkbox("Show legend", value=True)
    show_r_leg = vcol4.checkbox(
        "Show residual legend", value=False, 
        disabled=not show_res, 
        help="Only applies when 'Show Residuals Plot' is enabled.",
    )

    # Re-evaluate internal model grid for current resolution
    if x_max_eval <= x_min_eval:
        st.error("x-max must be greater than x-min in the Evaluation Grid settings.")
        st.stop()
    x_model_custom = np.linspace(x_min_eval, x_max_eval, int(n_points_eval))

    x_model_custom = np.linspace(x_min_eval, x_max_eval, int(n_points_eval))
    lg.x_model = x_model_custom
    fitdata = lg.get_fitdata()
    fitdata.x_model = x_model_custom
    fitdata.y_fit = lg.eval(x=x_model_custom)
    fd = fitdata

    # 3. Generate Plot
    plotter = FitPlotter(fitdata)

    pretty_kw = {
        "width": 6.0,
        "height": 6.0 if show_res else 5.5,
        "dpi": dpi_val
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

    # 4. Deconstruct multi-component lines on plot
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

    # 5. Legend Styling & Entry Cap
    if show_m_leg:
        MAX_LEGEND_ENTRIES = 20
        handles, labels = ax_main.get_legend_handles_labels()
        if len(labels) > MAX_LEGEND_ENTRIES:
            ax_main.legend(
                handles[:MAX_LEGEND_ENTRIES], labels[:MAX_LEGEND_ENTRIES],
                loc="best", frameon=True, edgecolor="black", fontsize="small",
                title=f"showing {MAX_LEGEND_ENTRIES}/{len(labels)} entries"
            )
        else:
            ax_main.legend(loc="best", frameon=True, edgecolor="black")
    elif ax_main.get_legend() is not None:
        ax_main.get_legend().remove()

    fig.tight_layout()
    st.pyplot(fig, width="stretch")

    # ------------------------------------------------------------------
    # Data Export Section
    # ------------------------------------------------------------------
    # st.subheader("Export Results")
    render_fancy_header(
        "Export Results", 
        level=3, 
    )

    y_fit_custom = fd.y_fit

    # Model evaluation export DataFrame
    model_export = pd.DataFrame({"x_eval": x_model_custom})
    if y_fit_custom.ndim == 1:
        model_export["yfit1"] = y_fit_custom
    else:
        for j in range(fd.ny):
            model_export[f"yfit{j+1}"] = y_fit_custom[:, j]

    # Append individual component evaluations
    if lg.is_multicomponent:
        comps = lg.eval_components(x_model=x_model_custom)
        for comp_name, comp_data in comps.items():
            comp_tag = component_short_tag(comp_name, component_choices)
            if isinstance(comp_data, dict):
                for j in range(fd.ny):
                    c_vals = comp_data.get(j, comp_data.get(f"ds_{j}"))
                    if c_vals is not None:
                        model_export[f"yfit{j+1}_{comp_tag}"] = c_vals
            elif isinstance(comp_data, np.ndarray):
                if comp_data.ndim == 2 and comp_data.shape[1] == fd.ny:
                    for j in range(fd.ny):
                        model_export[f"yfit{j+1}_{comp_tag}"] = comp_data[:, j]
                else:
                    model_export[comp_tag] = comp_data.ravel()

    # Raw data & calculated residuals DataFrame
    raw_export = pd.DataFrame({"x_data": fd.x_data})
    for j in range(fd.ny):
        raw_export[export_labels[j]] = fd.y_data[:, j] if fd.y_data.ndim > 1 else fd.y_data
        raw_export[f"{export_labels[j]}_resid"] = fd.resid_fit[:, j] if fd.resid_fit.ndim > 1 else fd.resid_fit

    # Filename tags & precision configuration
    _model_desc = "-".join(sanitize_label(c) for c in component_choices) or "model"
    _timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{_model_desc}_ny{ny}_{_timestamp}"

    # n_decimals = st.number_input(
    #     "Decimal precision for exported numbers", min_value=2, max_value=12,
    #     value=6, step=1,
    #     help="Controls the number of decimal places used in whitespace-aligned .dat exports."
    # )

    # Top control row: Decimal precision input aligned nicely on the left
    col_prec, col_spacer = st.columns([0.3, 0.7])
    with col_prec:
        n_decimals = st.number_input(
            "Decimal precision",
            min_value=2,
            max_value=12,
            value=6,
            step=1,
            help="Controls the number of decimal places used in whitespace-aligned .dat exported data"
        )

    float_fmt = f"%.{int(n_decimals)}e"

    if source == "Upload file" and any(el != dl for el, dl in zip(export_labels, dataset_labels)):
        with st.expander("Export label key (short name → original column)"):
            st.table(pd.DataFrame({
                "Export label": export_labels,
                "Original column": dataset_labels,
            }))

    # Single horizontal row for all 4 download actions
    dcol1, dcol2, dcol3, dcol4 = st.columns(4)
    
    # Downloads Layout
    # dcol1, dcol2 = st.columns(2)

    with dcol1:
        file_name=f"fit_report_{base_name}.txt"
        file_name=f"fit_report.txt"
        st.download_button(
            "Report (.txt)",
            data=report_text,
            file_name=file_name,
            mime="text/plain"
        )
 
    with dcol2:
        img_buf = io.BytesIO()
        fig.savefig(img_buf, format="png", dpi=int(dpi_val), bbox_inches="tight")
        file_name=f"fit_plot_{base_name}.png"
        file_name=f"fit_plot.png"
        st.download_button(
            "Plot (.png)",
            data=img_buf.getvalue(),
            file_name=file_name,
            mime="image/png"
        )

    # st.caption(
    #     "Exports below are whitespace-aligned, fixed-width `.dat` files for compatibility with external software."
    # )

    # dcol3, dcol4 = st.columns(2)

    with dcol3:
        file_name=f"fit_curves_{base_name}.dat"
        file_name=f"fit_curves.dat"
        st.download_button(
            # "Fit Curves (.dat, aligned)",
            "Fit Curves (.dat)",
            data=to_fixed_width(model_export, float_fmt=float_fmt).encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )
 
    with dcol4:
        file_name=f"data_residuals_{base_name}.dat"
        file_name=f"data_residuals.dat"
        st.download_button(
            # "Data & Residuals (.dat, aligned)",
            "Data & Residuals (.dat)",
            data=to_fixed_width(raw_export, float_fmt=float_fmt).encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )

    plt.close(fig)