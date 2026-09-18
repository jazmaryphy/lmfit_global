# %%
from __future__ import annotations

import io
import contextlib

import numpy as np
import pandas as pd

from lmfit_global import LmfitGlobal

# %%
def _configure_lg(
    xy: np.ndarray,
    param_df: pd.DataFrame,
    component_choices: list[str],
    connectors: list[str],
    global_param_selections: list[tuple[int, str]],
    ny: int,
    n_components: int,
    function_library: dict,
    nan_policy_choice: str,
    x_min_fit: float,
    x_max_fit: float,
    log_level_choice: str = "warning",
) -> tuple["LmfitGlobal", list[str]]:
    """Builds and configures an LmfitGlobal instance -- parses inputs,
    rebuilds the lmfit backend, applies parameter overrides, and ties any
    globally-shared parameters. Stops short of calling .fit(), so the
    same setup backs both a live 'preview the initial guess' plot and
    the actual optimization run.

    Returns:
        (lg, link_warnings)
    """
    func_lst = []
    for c_idx, fname in enumerate(component_choices):
        spec = function_library[fname]
        comp_df = param_df[
            (param_df["Component_ID"] == c_idx) & (param_df["Dataset_ID"] == 0)
        ]

        init_params = {
            row["Parameter"]: {
                "value": row["Value"],
                "vary": bool(row["Vary"]),
                "min": row["Min"],
                "max": row["Max"],
            }
            for _, row in comp_df.iterrows()
        }

        func_lst.append({
            "func_name": spec["func"],
            "init_params": init_params,
            "func_kws": {},
        })

    items = {
        # NOTE: LmfitGlobal expects xrange as {"xmin": ..., "xmax": ...},
        # not a plain (min, max) tuple -- confirmed against core.py.
        "data": {"xy": xy, "xrange": {"xmin": x_min_fit, "xmax": x_max_fit}},
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

    override_params = {
        row["Target_Key"]: {
            "value": float(row["Value"]) if isinstance(row["Value"], (int, float)) else row["Value"],
            "vary": bool(row["Vary"]),
            "min": float(row["Min"]),
            "max": float(row["Max"]),
        }
        for _, row in param_df.iterrows()
    }
    lg.update_params(override_params, overwrite_expr=True)

    link_warnings: list[str] = []

    if ny > 1 and global_param_selections:
        for c_idx, base in global_param_selections:
            parlist = [
                f"{base}_{j}" if n_components == 1 else f"c{c_idx}_{base}_{j}"
                for j in range(ny)
            ]
            if parlist[0] in lg.init_params:
                lg.set_global(parlist, reference=parlist[0], overwrite_expr=True)
            else:
                link_warnings.append(
                    f"Could not link Component {c_idx + 1}.{base} across "
                    f"datasets: parameter key '{parlist[0]}' was not found "
                    f"in the model."
                )

    return lg, link_warnings

# %%
def build_initial_model(
    xy: np.ndarray,
    param_df: pd.DataFrame,
    component_choices: list[str],
    connectors: list[str],
    global_param_selections: list[tuple[int, str]],
    ny: int,
    n_components: int,
    function_library: dict,
    nan_policy_choice: str,
    x_min_fit: float,
    x_max_fit: float,
) -> tuple["LmfitGlobal", list[str]]:
    """Builds an LmfitGlobal configured with the user's current initial
    parameters, WITHOUT running the fit -- for previewing the starting
    guess before committing to Run Fit.

    LmfitGlobal.plot_init()/eval() are safe to call on the returned
    instance: the initial-guess curve is computed from self.init_params
    unconditionally, while the fitted curve is gated behind
    self.fit_success (False until .fit() runs) -- see core.py's
    _build_fit_arrays().

    Returns:
        (lg, link_warnings) -- same shape as the first two return
        values of run_global_fit.
    """
    return _configure_lg(
        xy=xy, param_df=param_df, component_choices=component_choices,
        connectors=connectors, global_param_selections=global_param_selections,
        ny=ny, n_components=n_components, function_library=function_library,
        nan_policy_choice=nan_policy_choice, x_min_fit=x_min_fit, x_max_fit=x_max_fit,
        log_level_choice="warning",  # keep preview quiet; it may rerun often
    )

# %%
def run_global_fit(
    xy: np.ndarray,
    param_df: pd.DataFrame,
    component_choices: list[str],
    connectors: list[str],
    global_param_selections: list[tuple[int, str]],
    ny: int,
    n_components: int,
    function_library: dict,
    nan_policy_choice: str,
    fit_method_choice: str,
    log_level_choice: str,
    x_min_fit: float,
    x_max_fit: float,
) -> tuple["LmfitGlobal", str, list[str]]:
    """Runs one global (multi-dataset / multi-component) fit.

    This is deliberately kept free of any Streamlit calls (no st.spinner,
    st.success, st.error, etc.) so it can be unit-tested directly with
    plain pytest -- the caller (fit_view.py's button handler) is
    responsible for wrapping this in a spinner and catching/reporting
    exceptions.

    Returns:
        (lg, report_text, link_warnings): the fitted LmfitGlobal instance,
        its captured text report, and any (component, parameter) tie in
        global_param_selections that could not be applied.
    """
    lg, link_warnings = _configure_lg(
        xy=xy, param_df=param_df, component_choices=component_choices,
        connectors=connectors, global_param_selections=global_param_selections,
        ny=ny, n_components=n_components, function_library=function_library,
        nan_policy_choice=nan_policy_choice, x_min_fit=x_min_fit, x_max_fit=x_max_fit,
        log_level_choice=log_level_choice,
    )

    # NOTE: fit_method must be passed as the explicit `fit_method` kwarg,
    # NOT stashed in lg.fit_kws["method"] -- LmfitGlobal.fit() pops any
    # "method" key out of fit_kws (to avoid a duplicate-keyword collision
    # with lmfit.minimize()'s own `method=` argument) and instead falls
    # back to self.fit_method, which never gets updated that way.
    lg.fit(fit_method=fit_method_choice, verbose=False)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        lg.report()

    return lg, buf.getvalue(), link_warnings