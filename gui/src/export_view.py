# %%
from __future__ import annotations

import io
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gui.src.utils import component_short_tag, sanitize_label, to_fixed_width, render_fancy_header

# %%
def render_export(
    lg,
    fd,
    x_model_custom: np.ndarray,
    ny: int,
    dataset_labels: list[str],
    export_labels: list[str],
    component_choices: list[str],
    report_text: str,
    fig,
    dpi_val: int,
    source: str,
):
    """Renders the Export Results section: builds the fit-curve and
    raw-data/residual export tables, and the download buttons for the
    report, plot, and whitespace-aligned .dat files.
    """
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

    with dcol3:
        file_name=f"fit_curves_{base_name}.dat"
        file_name=f"fit_curves.dat"
        st.download_button(
            "Fit Curves (.dat)",
            data=to_fixed_width(model_export, float_fmt=float_fmt).encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )

    with dcol4:
        file_name=f"data_residuals_{base_name}.dat"
        file_name=f"data_residuals.dat"
        st.download_button(
            "Data & Residuals (.dat)",
            data=to_fixed_width(raw_export, float_fmt=float_fmt).encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
        )

    plt.close(fig)