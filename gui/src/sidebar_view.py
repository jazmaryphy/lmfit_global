# %%
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st

from gui.library import FUNCTION_LIBRARY, CONNECTORS
from gui.demo_data import DEMO_DATASETS, make_demo_data
from gui.src.utils import render_fancy_header, sanitize_label

# %%
def render_data() -> tuple[np.ndarray | None, list[str], list[str], str]:
    """Renders Sidebar section 1 and handles data extraction."""
    with st.sidebar:
        render_fancy_header(
            "Data Input", 
            step_number=1, 
            level=2,
            title_color="#38bdf8"
        )
        source = st.radio("Data source", ["Built-in demo", "Upload file"])
        xy, y_cols = None, []

        if source == "Upload file":
            uploaded = st.file_uploader("Data file (CSV / DAT / TSV / TXT)", type=None)
            delim = st.selectbox("Delimiter", ["auto", ",", "\t", "whitespace"], index=0)
            comment_char = st.text_input("Comment prefix", value="#")

            if uploaded is not None:
                content = uploaded.getvalue().decode("utf-8")
                lines = content.splitlines()

                comment_prefix = comment_char.strip() if comment_char.strip() else "#"
                clean_lines = [l for l in lines if not l.strip().startswith(comment_prefix)]
                clean_data = "\n".join(clean_lines)

                # 1. Smart Delimiter Auto-Detection
                if delim == "auto":
                    first_line = clean_lines[0] if clean_lines else ""
                    if "," in first_line:
                        sep = ","
                    elif "\t" in first_line:
                        sep = "\t"
                    else:
                        sep = r"\s+"
                elif delim == "whitespace":
                    sep = r"\s+"
                else:
                    sep = delim

                raw = pd.read_csv(
                    io.StringIO(clean_data),
                    sep=sep,
                    engine="python",
                    header=None,
                    skip_blank_lines=True,
                )

                first_row_numeric = raw.iloc[0].apply(
                    lambda v: pd.to_numeric(v, errors="coerce")
                ).notna().all()
                has_header = st.checkbox("File has header row", value=not first_row_numeric)

                if has_header:
                    df = raw.iloc[1:].reset_index(drop=True)
                    df.columns = raw.iloc[0].astype(str)
                else:
                    df = raw.copy()
                    df.columns = [f"col{i}" for i in range(df.shape[1])]

                df = df.apply(pd.to_numeric, errors="coerce")
                n_before = len(df)
                df = df.dropna(how="all")
                col_names = list(df.columns.astype(str))

                st.write("Preview:", df.head(3))
                if n_before != len(df):
                    st.caption(f"Dropped {n_before - len(df)} fully non-numeric row(s).")

                x_col = st.selectbox("X column", col_names, index=0)
                y_cols = st.multiselect(
                    "Y column(s)",
                    [c for c in col_names if c != x_col],
                    default=[c for c in col_names if c != x_col][:1]
                )

                if y_cols:
                    # 2. Preserve Sparse Datasets with NaNs
                    # Ensure X column itself is valid
                    x_valid_df = df.dropna(subset=[x_col])
                    
                    if len(x_valid_df) > 0:
                        # Extract X and replace individual NaN entries with np.nan for global array
                        x_arr = x_valid_df[x_col].to_numpy()
                        y_arrs = [x_valid_df[c].to_numpy() for c in y_cols]
                        
                        xy = np.column_stack([x_arr] + y_arrs)
                        
                        # Inform user if sparse data exists
                        has_nans = np.isnan(xy[:, 1:]).any()
                        if has_nans:
                            st.warning(
                                "⚠️ **Warning:** Sparse dataset detected: "
                                "`NaN` entries will be ignored individually during fitting. "
                                "See `Advanced Fitting Setting` below."
                            )
                            
        else:
            selected_demo = st.selectbox("Select Demo Preset", list(DEMO_DATASETS.keys()))
            col_ny, col_pts = st.columns(2)
            ny_input = col_ny.number_input("N-datasets", min_value=1, max_value=20, value=5, step=1)
            n_pts_input = col_pts.number_input("Points / Data", min_value=50, max_value=2000, value=151, step=25)
            xy = make_demo_data(
                name=selected_demo,
                ny=int(ny_input),
                n_points=int(n_pts_input)
            )

    ny = xy.shape[1] - 1 if xy is not None else 0

    if source == "Upload file" and y_cols:
        dataset_labels = [sanitize_label(c) for c in y_cols]
    else:
        dataset_labels = [f"dataset{j+1}" for j in range(ny)]
    export_labels = [f"ydat{j+1}" for j in range(ny)]

    return xy, dataset_labels, export_labels, source

# %%
def render_model(xy: np.ndarray):
    """Renders Sidebar sections 2 & 3: model construction (component
    functions + connectors), the X evaluation grid, and the advanced
    fitting settings expander (NaN policy / optimizer / log level).

    Returns everything app.py needs to proceed to the parameter editor
    and fit execution steps.
    """
    with st.sidebar:
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

        render_fancy_header(
            title="X-Grid",
            step_number=3,
            level=2,
            title_color="#38bdf8"  # Universal Electric Blue
        )
        x_min_eval = st.number_input("x-min", value=float(np.nanmin(xy[:, 0])), format="%.4f")
        x_max_eval = st.number_input("x-max", value=float(np.nanmax(xy[:, 0])), format="%.4f")
        n_points_eval = st.number_input("numpoints (N)", min_value=50, max_value=10000, value=500, step=50)

        with st.expander("Advanced Fitting Settings", expanded=False):
            nan_policy_choice = st.selectbox(
                "NaN Policy",
                options=["omit", "raise", "propagate"],
                index=0,  # Defaults to "omit" (ignores NaNs)
                help="How to handle NaN/missing values: 'omit' ignores NaNs, 'raise' throws an error, 'propagate' returns NaN."
            )

            fit_method_choice = st.selectbox(
                "Optimization Algorithm",
                options=[
                    "leastsq",       # Levenberg-Marquardt (default)
                    "least_squares", # Least-Squares (Trust Region Reflective)
                    "nelder",        # Nelder-Mead
                    "powell",        # Powell
                    "cobyla",        # COBYLA
                    "bfgs",          # BFGS
                    "lbfgsb",        # L-BFGS-B
                    "cg",            # Conjugate Gradient
                    "differential_evolution"  # Global optimization
                ],
                index=0,
                help="Algorithm used by scipy.optimize / lmfit to minimize residuals."
            )

            log_level_choice = st.selectbox(
                "Logging Level",
                options=["warning", "info", "debug", "error"],
                index=0,
                help="Console/logger verbosity level."
            )

    return (
        n_components, component_choices, connectors, all_selected,
        x_min_eval, x_max_eval, n_points_eval,
        nan_policy_choice, fit_method_choice, log_level_choice,
    )