# %%
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gui.library import FUNCTION_LIBRARY
from gui.src.utils import render_fancy_header

# %%
def render_parameter(xy: np.ndarray, component_choices: list[str]) -> pd.DataFrame:
    """Builds interactive tabbed data-editors for model bounds/initial parameters."""
    # st.header("4. Initial Parameter Editor & Bounds")
    # st.caption("Edit value / bounds / vary flag for each component's parameters.")
    render_fancy_header(
        "Initial Parameter Editor & Bounds", 
        step_number=4, 
        subtitle="Edit values, min/max bounds, and vary flags for each component.", 
        level=2,
        title_color="#38bdf8"  # Universal Electric Blue
    )

    ny = xy.shape[1] - 1
    x_data_flat = xy[:, 0]
    edited_dfs = []
    
    # Expected columns schema
    columns = [
        "Component_ID", "Dataset_ID", "Target_Key", "Component", 
        "Parameter", "Value", "Vary", "Min", "Max"
    ]
    
    ds_tabs = st.tabs([f"Dataset {j+1}" for j in range(ny)])

    for ds_idx, tab in enumerate(ds_tabs):
        with tab:
            ds_param_rows = []
            for c_idx, fname in enumerate(component_choices):
                spec = FUNCTION_LIBRARY.get(fname, {})
                params = spec.get("params", {})
                
                for pname, pdefault in params.items():
                    val = pdefault.get("value", 1.0)
                    is_num = isinstance(val, (int, float))
                    y_data_ds = xy[:, ds_idx + 1]
                    ds_val = float(val) if is_num else val

                    if is_num and c_idx == 0 and fname in ("Gaussian", "Lorentzian", "Voigt"):
                        if pname == "amplitude":
                            ds_val = float(np.nanmax(y_data_ds) - np.nanmin(y_data_ds))
                        elif pname == "center":
                            ds_val = float(x_data_flat[np.nanargmax(y_data_ds)])

                    key_name = f"{pname}_{ds_idx}" if len(component_choices) == 1 else f"c{c_idx}_{pname}_{ds_idx}"
                    
                    ds_param_rows.append({
                        "Component_ID": c_idx,
                        "Dataset_ID": ds_idx,
                        "Target_Key": key_name,
                        "Component": f"Comp {c_idx+1}: {fname}",
                        "Parameter": pname,
                        "Value": ds_val,
                        "Vary": pdefault.get("vary", True) if is_num else False,
                        "Min": float(pdefault.get("min", -np.inf)) if is_num else -np.inf,
                        "Max": float(pdefault.get("max", np.inf)) if is_num else np.inf
                    })

            # Explicitly enforce column structure even if ds_param_rows is empty
            ds_df = pd.DataFrame(ds_param_rows, columns=columns)

            # Guard clause against empty DataFrame
            if ds_df.empty:
                st.warning("No parameters found for the selected components.")
                edited_dfs.append(ds_df)
                continue

            is_numeric = ds_df["Value"].apply(lambda v: isinstance(v, (int, float)))

            edited_numeric = st.data_editor(
                ds_df[is_numeric],
                column_config={
                    "Component_ID": None, "Dataset_ID": None, "Target_Key": None,
                    "Component": st.column_config.TextColumn("Component", disabled=True),
                    "Parameter": st.column_config.TextColumn("Parameter", disabled=True),
                    "Value": st.column_config.NumberColumn("Initial Value", format="%.6g"),
                    "Vary": st.column_config.CheckboxColumn("Vary", default=True),
                    "Min": st.column_config.NumberColumn("Min Bound", format="%.6g"),
                    "Max": st.column_config.NumberColumn("Max Bound", format="%.6g"),
                },
                width="stretch",
                hide_index=True,
                key=f"editor_ds_{ds_idx}"
            )
            edited_dfs.append(pd.concat([edited_numeric, ds_df[~is_numeric]], ignore_index=True))

    return pd.concat(edited_dfs, ignore_index=True) if edited_dfs else pd.DataFrame(columns=columns)