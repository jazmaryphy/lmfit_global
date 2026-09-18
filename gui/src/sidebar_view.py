# %%
from __future__ import annotations

import io
import uuid
import numpy as np
import pandas as pd
import streamlit as st

from gui.src.demo_data import DEMO_DATASETS, make_demo_data
from gui.src.library import FUNCTION_LIBRARY, CONNECTORS, _CONNECTOR_LABELS
from gui.src.utils import render_fancy_header, render_thin_divider, sanitize_label
from gui.src.custom_functions import (
    make_library_entry, 
    extract_parameter_names, 
    validate_custom_function,
    available_special_functions,
)

# %%
def render_data() -> tuple[np.ndarray | None, list[str], list[str], str]:
    """Renders Sidebar section 1 and handles data extraction."""
    with st.sidebar:
        render_fancy_header(
            "Data input", 
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
def _remove_component_row(row_id: str) -> None:
    """Button callback: drop a component row and its stale widget state.

    Uses on_click (not an inline `if button:` check) because mutating a
    session_state list mid-render-loop, keyed by row_id (a uuid) rather
    than positional index, is what keeps a middle-row removal from
    corrupting the widget state of the rows around it.
    """
    st.session_state.component_rows = [
        r for r in st.session_state.component_rows if r["id"] != row_id
    ]
    st.session_state.pop(f"func_{row_id}", None)
    st.session_state.pop(f"conn_{row_id}", None)

# %%
def render_model(xy: np.ndarray):
    """Renders Sidebar section 2: model construction (component functions
    + connectors), fit range, X-eval grid, and advanced fitting settings.

    Fit range and x-fine-grid are deliberately unnumbered (step_number=None)
    and nested inside the same bordered container as Fit Model -- they are
    sub-settings of the model step, not independent workflow stages.

    Returns everything app.py needs to proceed to the parameter editor
    and fit execution steps.
    """
    x_data_min = float(np.nanmin(xy[:, 0]))
    x_data_max = float(np.nanmax(xy[:, 0]))

    # reset the grid_from/grid_to session state if the data range has changed
    grid_data_sig = (round(x_data_min, 6), round(x_data_max, 6))
    if st.session_state.get("_grid_data_sig") != grid_data_sig:
        st.session_state["grid_from"] = x_data_min
        st.session_state["grid_to"] = x_data_max
        st.session_state["_grid_data_sig"] = grid_data_sig

    if "custom_functions" not in st.session_state:
        st.session_state.custom_functions = {}
    # Built up front so the component pickers below always see the
    # latest custom functions, including ones just added this rerun.
    combined_library = {**FUNCTION_LIBRARY, **st.session_state.custom_functions}

    with st.sidebar:
        render_fancy_header(title="Fit Model", step_number=2, level=2, title_color="#38bdf8")

        with st.container(border=True, key="fit_model_box"):
            # Pin the icon-button columns to a fixed width so they never
            # shrink below button size when the sidebar is narrowed, and
            # stop the row from wrapping onto two lines.
            st.markdown(
                """
                <style>
                .st-key-fit_model_box div[data-testid="stHorizontalBlock"] {
                    flex-wrap: nowrap !important;
                }
                .st-key-fit_model_box div[data-testid="column"]:has(
                    div[data-testid="stButton"]
                ) {
                    flex: 0 0 44px !important;
                    min-width: 44px !important;
                    width: 44px !important;
                }
                .st-key-fit_model_box div[data-testid="column"]:has(
                    div[data-testid="stSelectbox"]
                ) {
                    flex: 1 1 auto !important;
                    min-width: 0 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            if "component_rows" not in st.session_state:
                st.session_state.component_rows = [{"id": str(uuid.uuid4())}]

            MAX_COMPONENTS = 6
            component_choices, connectors, all_selected = [], [], True

            for i, row in enumerate(st.session_state.component_rows):
                rid = row["id"]

                if i == 0:
                    col_fn, col_add = st.columns([5, 1])
                    fname = col_fn.selectbox(
                        "Model", list(combined_library.keys()), index=None,
                        key=f"func_{rid}", label_visibility="collapsed",
                        placeholder="Select model…",
                    )
                    if len(st.session_state.component_rows) < MAX_COMPONENTS:
                        col_add.button(
                            "➕", key="add_component", help="Add another component",
                            use_container_width=True,
                            on_click=lambda: st.session_state.component_rows.append(
                                {"id": str(uuid.uuid4())}
                            ),
                        )
                else:
                    col_conn, col_fn, col_rm = st.columns([1.5, 3.0, 0.6])
                    connector = col_conn.selectbox(
                        "Combine using",
                        CONNECTORS,
                        key=f"conn_{rid}",
                        label_visibility="collapsed",
                        format_func=lambda op: _CONNECTOR_LABELS[op],
                        help="How this component combines with the one above it",
                    )
                    fname = col_fn.selectbox(
                        "Model", list(combined_library.keys()), index=None,
                        key=f"func_{rid}", label_visibility="collapsed",
                        placeholder="Select model…",
                    )
                    col_rm.button(
                        "🗑️", key=f"rm_{rid}", help="Remove this component",
                        use_container_width=True,
                        on_click=_remove_component_row, args=(rid,),
                    )
                    connectors.append(connector)

                if fname is None:
                    all_selected = False
                else:
                    component_choices.append(fname)

            n_components = len(st.session_state.component_rows)

            # Custom function definition 
            with st.expander("➕ Define custom function", expanded=False):
                st.caption(
                    "Write a formula using `x` as the independent variable and "
                    "any other names as fittable parameters, e.g.\n\n"
                    "`amplitude * exp(-(x-center)**2/(2*sigma**2)) + slope*x`\n\n"
                    f"Also available: `{', '.join(available_special_functions())}`."
                )
                custom_name = st.text_input("Function name", key="custom_name_input")
                custom_expr = st.text_area(
                    "Formula",
                    placeholder="amplitude * exp(-(x-center)**2/(2*sigma**2)) + slope*x + intercept",
                    height=80, key="custom_expr_input",
                )

                if st.button("Validate & Add", key="add_custom_func"):
                    name = custom_name.strip()
                    expr = custom_expr.strip()
                    if not name or not expr:
                        st.error("Both a name and a formula are required.")
                    elif name in FUNCTION_LIBRARY:
                        st.error(f"'{name}' collides with a built-in function name — choose a different name.")
                    else:
                        param_names = extract_parameter_names(expr)
                        if not param_names:
                            st.error("No fittable parameters detected in the formula.")
                        else:
                            error = validate_custom_function(expr, param_names)
                            if error:
                                st.error(f"Formula error: {error}")
                            else:
                                st.session_state.custom_functions[name] = make_library_entry(expr, param_names)
                                st.success(f"Added '{name}' — parameters: {', '.join(param_names)}")
                                st.rerun()  # refresh so the new entry appears in the pickers immediately

                if st.session_state.custom_functions:
                    st.caption("Your custom functions: " + ", ".join(st.session_state.custom_functions.keys()))
                    remove_choice = st.selectbox(
                        "Remove a custom function",
                        [""] + list(st.session_state.custom_functions.keys()),
                        label_visibility="collapsed", key="remove_custom_choice",
                    )
                    if remove_choice and st.button(f"🗑️ Remove '{remove_choice}'", key="remove_custom_btn"):
                        st.session_state.custom_functions.pop(remove_choice, None)
                        st.rerun()

            # Fit range: sub-setting of Fit Model, no badge
            render_thin_divider()
            render_fancy_header(
                title="Fit range", step_number=None,
                title_color="#7dd3fc", title_size="0.95rem",
                title_margin="0.2rem 0 0.15rem",
            )
            col_from, col_to = st.columns(2)
            x_min_fit = col_from.number_input(
                "From", value=x_data_min,
                min_value=x_data_min, max_value=x_data_max,
                format="%.4f",
            )
            x_max_fit = col_to.number_input(
                "To", value=x_data_max,
                min_value=x_data_min, max_value=x_data_max,
                format="%.4f",
            )
            if x_min_fit >= x_max_fit:
                st.warning(
                    f"⚠️ Fit range 'From' ({x_min_fit:.4f}) must be less than "
                    f"'To' ({x_max_fit:.4f}). Swapping them for the fit."
                )
                x_min_fit, x_max_fit = x_max_fit, x_min_fit

            # x-fine-grid: sub-setting of Fit Model, no badge 
            render_thin_divider()
            render_fancy_header(
                title="Fine grid", step_number=None,
                title_color="#7dd3fc", title_size="0.95rem",
                title_margin="0.2rem 0 0.15rem",
            )
            col_gmin, col_gmax, col_gn = st.columns(3)
            x_min_eval = col_gmin.number_input(
                "From", value=x_data_min, format="%.4f", key="grid_from"
            )
            x_max_eval = col_gmax.number_input(
                "To", value=x_data_max, format="%.4f", key="grid_to"
            )
            n_points_eval = col_gn.number_input(
                "N", min_value=50, max_value=10000, value=500, step=50, key="grid_n"
            )

        with st.container(key="fit_settings_box"):
            st.markdown(
                """
                <style>
                .st-key-fit_settings_box div[data-testid="stExpander"] summary p {
                    color: #ff4b4b !important;
                    font-weight: 700 !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("⚙️ Fit settings", expanded=False):
                nan_policy_choice = st.selectbox(
                    "NaN Policy",
                    options=["omit", "raise", "propagate"],
                    index=0,
                    # help="How to handle NaN/missing values.",
                    help=f"How to handle NaN/missing values: "
                        f"'omit' ignores NaNs, 'raise' throws an error, 'propagate' returns NaN.",
                )
                fit_method_choice = st.selectbox(
                    "Optimization Algorithm",
                    options=[
                        "leastsq", "least_squares", "nelder", "powell",
                        "cobyla", "bfgs", "lbfgsb", "cg", "differential_evolution",
                    ],
                    index=0,
                    help="Algorithm used by scipy.optimize / lmfit to minimize residuals.",
                )
                log_level_choice = st.selectbox(
                    "Logging Level",
                    options=["warning", "info", "debug", "error"],
                    index=0,
                    help="Console/logger verbosity level.",
                )

    return (
        n_components, component_choices, connectors, all_selected,
        x_min_fit, x_max_fit,
        x_min_eval, x_max_eval, n_points_eval,
        nan_policy_choice, fit_method_choice, log_level_choice,
        combined_library,
    )