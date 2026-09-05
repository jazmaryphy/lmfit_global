# %%
from __future__ import annotations

import io
import contextlib

import numpy as np
import pandas as pd

from lmfit_global import LmfitGlobal

# %%
def run_global_fit(
    xy: np.ndarray,
    param_df: pd.DataFrame,
    component_choices: list[str],
    connectors: list[str],
    global_param_bases: list[str],
    ny: int,
    n_components: int,
    function_library: dict,
    nan_policy_choice: str,
    fit_method_choice: str,
    log_level_choice: str,
):
    """Runs one global (multi-dataset / multi-component) fit.

    This is deliberately kept free of any Streamlit calls (no st.spinner,
    st.success, st.error, etc.) so it can be unit-tested directly with
    plain pytest -- the caller (app.py's button handler) is responsible
    for wrapping this in a spinner and catching/reporting exceptions.

    Returns:
        (lg, report_text): the fitted LmfitGlobal instance and its
        captured text report.

    Raises:
        Whatever LmfitGlobal itself raises on a bad model/fit -- the
        caller should catch and display this.
    """
    func_lst = []
    for c_idx, fname in enumerate(component_choices):
        spec = function_library[fname]
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

    items = {
        "data": {"xy": xy, "xrange": None},
        "functions": {
            "theory": func_lst,
            "theory_connectors": connectors if connectors else None,
        },
    }

    lg = LmfitGlobal(items=items, log_level=log_level_choice)

    if hasattr(lg, "set_nan_policy"):
        lg.set_nan_policy(nan_policy_choice)

    # IMPORTANT: rebuild() must be called BEFORE set_global(), not after --
    # calling rebuild() after tying parameters discards the tie.
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

    if hasattr(lg, "fit_kws") and isinstance(lg.fit_kws, dict):
        lg.fit_kws["method"] = fit_method_choice
    elif hasattr(lg, "minimize_kws") and isinstance(lg.minimize_kws, dict):
        lg.minimize_kws["method"] = fit_method_choice
    elif hasattr(lg, "fit_method"):
        lg.fit_method = fit_method_choice

    # Method can also be explicitly passed or overridden during .fit()
    lg.fit(verbose=False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lg.report()

    return lg, buf.getvalue()