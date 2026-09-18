# %%
from __future__ import annotations

import numpy as np
import streamlit as st

from gui.src.plot_view import render_plot
from gui.src.export_view import render_export
from gui.src.fit_view import render_fit_execution
from gui.src.sidebar_view import render_data, render_model
from gui.src.parameter_view import render_parameter, render_shared_parameters

from gui.src.utils import invalidate_stale_fit_state

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

# %%
### SIDEBAR VIEW
# MODEL: construction, x-grid, advanced fitting settings
#
(
    n_components, component_choices, connectors, all_selected,
    x_min_fit, x_max_fit,
    x_min_eval, x_max_eval, n_points_eval,
    nan_policy_choice, fit_method_choice, log_level_choice,
) = render_model(xy)

if not all_selected:
    st.info("Please choose a function for every component in the sidebar.")
    st.stop()

# Invalidate stale fit/preview state if the data OR the model/fit-range
# changes -- both checked together so a stale result can't survive
# either kind of change without the user noticing it's gone.
data_sig = (xy.shape, float(np.nansum(xy)))
model_sig = (tuple(component_choices), tuple(connectors), x_min_fit, x_max_fit)
invalidate_stale_fit_state(data_sig, model_sig)

# %%
### MAIN VIEW
# PARAMETERS: view and edit parameters
#
param_df = render_parameter(xy, component_choices)

# %%
### MAIN VIEW
# GLOBAL PARAMETERS: global links between parameters (EXPERIMENTAL)
#
global_param_selections: list[tuple[int, str]] = []
if ny > 1:
    global_param_selections = render_shared_parameters(param_df, component_choices)

# %%
### MAIN VIEW
# FIT EXECUTION
#
render_fit_execution(
    xy=xy,
    param_df=param_df,
    component_choices=component_choices,
    connectors=connectors,
    global_param_selections=global_param_selections,
    ny=ny,
    n_components=n_components,
    nan_policy_choice=nan_policy_choice,
    fit_method_choice=fit_method_choice,
    log_level_choice=log_level_choice,
    x_min_fit=x_min_fit,
    x_max_fit=x_max_fit,
    n_points_eval=n_points_eval,
)

# %%
### MAIN VIEW
# FIT REPORT, PLOT & EXPORT
#
if "fitted_lg" in st.session_state:
    lg = st.session_state["fitted_lg"]
    report_text = st.session_state["report_text"]

    just_fit = st.session_state.pop("_just_fit", False)
    just_fit = False  # DEBUG: always expand fit report for now

    with st.container(key="fit_report_box"):
        st.markdown(
            """
            <style>
            .st-key-fit_report_box div[data-testid="stExpander"] summary p,
            .st-key-fit_report_box div[data-testid="stExpander"] summary span {
                color: #ff4b4b !important;
                font-weight: 700 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        text_str = "Fit report"
        text_str = "📄 Fit report"
        with st.expander(text_str, expanded=just_fit):
            st.code(report_text, language="text")

    fig, fitdata, x_model_custom = render_plot(
        lg=lg,
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
        source=source,
    )