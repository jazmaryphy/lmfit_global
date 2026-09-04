# %%
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import streamlit as st
from gui.demo_data import DEMO_DATASETS, make_demo_data
from gui.src.utils import render_fancy_header, sanitize_label

# %%
def render_data_sidebar() -> tuple[np.ndarray | None, list[str], list[str], str]:
    """Renders Sidebar section 1 and handles data extraction."""
    with st.sidebar:
        render_fancy_header(
            "Data Input", 
            step_number=1, 
            level=2,
            title_color="#38bdf8"  # Universal Electric Blue
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

                sep = r"\s+" if delim in ("auto", "whitespace") else delim

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
                    sub = df[[x_col] + y_cols].apply(pd.to_numeric, errors="coerce")
                    n_before2 = len(sub)
                    sub = sub.dropna(how="any")
                    if n_before2 != len(sub):
                        st.warning(
                            f"Dropped {n_before2 - len(sub)} row(s) with non-numeric "
                            f"or missing values in the selected columns."
                        )
                    if len(sub) > 0:
                        xy = np.column_stack([sub[x_col].to_numpy()] + [sub[c].to_numpy() for c in y_cols])
        else:
            selected_demo = st.selectbox("Select Demo Preset", list(DEMO_DATASETS.keys()))
            col_ny, col_pts = st.columns(2)
            ny_input = col_ny.number_input("Datasets (N)", min_value=1, max_value=20, value=5, step=1)
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