# %%
from __future__ import annotations

import numpy as np
import streamlit as st

from gui.library import FUNCTION_LIBRARY
from gui.src.plot_view import render_plot
from gui.src.export_view import render_export
from gui.src.fit_runner import run_global_fit
from gui.src.parameter_view import render_parameter
from gui.src.sidebar_view import render_data, render_model

from gui.src.utils import render_fancy_header

# %%
# Page Configuration
#
st.set_page_config(
    page_title="lmfit-global GUI",
    layout="wide"
)
st.title("`lmfitgedit` — Interactive Multi-Dataset / Multi-Component Fitting")
st.caption(
    "`lmfitgedit`: The GUI Based Interface to `LmfitGlobal` class from "
    "[jazmaryphy/lmfit_global](https://github.com/jazmaryphy/lmfit_global)."
)

# %%
### SIDEBAR VIEW
# DATA INPUT
#
xy, dataset_labels, export_labels, source = render_data()
if xy is None or xy.shape[1] <= 1:
    st.info("Upload data or select a demo preset to proceed.")
    st.stop()

ny = xy.shape[1] - 1

# Invalidate stale fit state if data signature changes
data_sig = (xy.shape, float(np.nansum(xy)))
if st.session_state.get("_data_signature") != data_sig:
    st.session_state.pop("fitted_lg", None)
    st.session_state["_data_signature"] = data_sig

# %%
### SIDEBAR VIEW
# MODEL: construction, x-grid, advanced fitting settings
#
(
    n_components, component_choices, connectors, all_selected,
    x_min_eval, x_max_eval, n_points_eval,
    nan_policy_choice, fit_method_choice, log_level_choice,
) = render_model(xy)

if not all_selected:
    st.info("Please choose a function for every component in the sidebar.")
    st.stop()

# %%
### MAIN VIEW
# PARAMETERS: view and edit parameters
#
param_df = render_parameter(xy, component_choices)

# %%
### MAIN VIEW
# GLOBAL PARAMETERS: global links between parameters (EXPERIMENTAL)
#
global_param_bases = []
if ny > 1:
    render_fancy_header(
        title="Shared Parameters Across Datasets",
        step_number=5,
        level=2,
        title_color="#38bdf8"  # Universal Electric Blue
    )
    unique_params = sorted(set(param_df["Parameter"].tolist()))
    global_param_bases = st.multiselect("Select base parameters to link globally:", unique_params)

# %%
### MAIN VIEW
# FIT EXECUTION
#
render_fancy_header(
    title="Optimization & Results",
    step_number=6,
    level=2,
    title_color="#38bdf8"  # Universal Electric Blue
)

runfit_str = "🚀 Run Fit Optimization"
if st.button(runfit_str, type="primary", use_container_width=True):
    with st.spinner("⚡ Running fit, please wait..."):
        try:
            lg, report_text = run_global_fit(
                xy=xy,
                param_df=param_df,
                component_choices=component_choices,
                connectors=connectors,
                global_param_bases=global_param_bases,
                ny=ny,
                n_components=n_components,
                function_library=FUNCTION_LIBRARY,
                nan_policy_choice=nan_policy_choice,
                fit_method_choice=fit_method_choice,
                log_level_choice=log_level_choice,
            )
            st.session_state["fitted_lg"] = lg
            st.session_state["report_text"] = report_text
            st.success("Optimization completed successfully.")
        except Exception as e:
            st.error(f"Fit failed: {e}")

# %%
### MAIN VIEW
# PLOT & EXPORT
#
if "fitted_lg" in st.session_state:
    lg = st.session_state["fitted_lg"]
    report_text = st.session_state["report_text"]

    fig, fitdata, x_model_custom, dpi_val = render_plot(
        lg=lg,
        report_text=report_text,
        xy=xy,
        ny=ny,
        dataset_labels=dataset_labels,
        component_choices=component_choices,
        x_min_eval=x_min_eval,
        x_max_eval=x_max_eval,
        n_points_eval=n_points_eval,
    )

    render_export(
        lg=lg,
        fd=fitdata,
        x_model_custom=x_model_custom,
        ny=ny,
        dataset_labels=dataset_labels,
        export_labels=export_labels,
        component_choices=component_choices,
        report_text=report_text,
        fig=fig,
        dpi_val=dpi_val,
        source=source,
    )