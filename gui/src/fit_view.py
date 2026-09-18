# %%
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# from gui.src.library import FUNCTION_LIBRARY # REMOVE:
from gui.src.utils import render_fancy_header
from gui.src.fit_runner import build_initial_model, run_global_fit

# %%
def render_fit_execution(
    xy, param_df, component_choices, connectors, global_param_selections,
    ny, n_components, nan_policy_choice, fit_method_choice, log_level_choice,
    x_min_fit, x_max_fit, n_points_eval, function_library,
):
    """Renders the 'Fit & Results' section: a Preview button (shows the
    current initial guess against the data without fitting) and the Run
    Fit button. On fit success, stores the fitted LmfitGlobal instance
    and report text in st.session_state, fires a one-shot toast, and
    sets a one-shot flag so the Fit Report expander opens automatically
    exactly once -- on the rerun right after the fit completes, not on
    every subsequent rerun.
    """
    render_fancy_header(
        title="Fit & Results", 
        step_number=5, 
        level=2, 
        title_color="#38bdf8"
    )

    st.markdown(
        """
        <style>
        /* Shared sizing for BOTH buttons -- keeps Preview and Fit visually
           paired (same height/padding/radius) even though they carry
           different semantic weight, which is expressed via color below
           instead of via mismatched sizing. */
        div.stButton > button {
            padding: 14px 20px !important;
            min-height: 58px !important;
            border-radius: 10px !important;
            width: 100% !important;
            font-weight: 700 !important;
            transition: filter 0.15s ease;
        }
        div.stButton > button:hover {
            filter: brightness(1.08);
        }

        /* Fit: primary action -- solid red, largest text */
        div.stButton > button[kind="primary"] {
            background-color: #ff4b4b !important;
            border: none !important;
        }
        div.stButton > button[kind="primary"] p,
        div.stButton > button[kind="primary"] span {
            font-size: 22px !important;
            font-weight: 800 !important;
            color: #ffffff !important;
            text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.4);
            line-height: 1.2 !important;
        }

        /* Preview: secondary action -- outlined, same accent blue used
           elsewhere (badges, "Fit range"/"x-fine-grid" labels), so it
           reads as part of the workflow without competing with Fit. */
        div.stButton > button[kind="secondary"] {
            background-color: rgba(56, 189, 248, 0.10) !important;
            border: 1.5px solid #38bdf8 !important;
        }
        div.stButton > button[kind="secondary"] p,
        div.stButton > button[kind="secondary"] span {
            font-size: 18px !important;
            font-weight: 700 !important;
            color: #38bdf8 !important;
            line-height: 1.2 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        # Range/DOF sanity check, upfront -- catches a fit that's
        # guaranteed to fail (or be meaningless) before the user clicks
        # Fit, rather than after it silently produces garbage.
        x_col = xy[:, 0]
        n_in_range = int(((x_col >= x_min_fit) & (x_col <= x_max_fit)).sum())
        n_free_params = int((param_df["Vary"] == True).sum())  # noqa: E712

        if n_in_range == 0:
            st.error(
                f"No data points fall inside the fit range "
                f"[{x_min_fit:.4f}, {x_max_fit:.4f}]. Widen the range before fitting."
            )
            return
        elif n_in_range * ny <= n_free_params:
            st.warning(
                f"⚠️ Only {n_in_range} point(s) per dataset fall inside the fit "
                f"range, but the model has {n_free_params} free parameter(s) "
                f"across {ny} dataset(s). The fit is likely to fail or be "
                f"poorly constrained — consider widening the range."
            )

        col_preview, col_fit = st.columns(2, gap="medium")

        with col_preview:
            if st.button(
                "👁️ Preview Initial Model", 
                use_container_width=True
                ):
                try:
                    preview_lg, preview_warnings = build_initial_model(
                        xy=xy,
                        param_df=param_df,
                        component_choices=component_choices,
                        connectors=connectors,
                        global_param_selections=global_param_selections,
                        ny=ny,
                        n_components=n_components,
                        function_library=function_library,
                        nan_policy_choice=nan_policy_choice,
                        x_min_fit=x_min_fit,
                        x_max_fit=x_max_fit,
                    )
                    st.session_state["preview_lg"] = preview_lg
                    for w in preview_warnings:
                        st.warning(w)
                except Exception as e:
                    st.error(f"Could not build preview: {e}")


        with col_fit:
            text_str = "🚀 Run Fit Optimization"
            text_str = "🚀 Fit"
            if st.button(text_str, type="primary", use_container_width=True):
                with st.spinner("⚡ Running fit, please wait..."):
                    try:
                        lg, report_text, link_warnings = run_global_fit(
                            xy=xy,
                            param_df=param_df,
                            component_choices=component_choices,
                            connectors=connectors,
                            global_param_selections=global_param_selections,
                            ny=ny,
                            n_components=n_components,
                            function_library=function_library,
                            nan_policy_choice=nan_policy_choice,
                            fit_method_choice=fit_method_choice,
                            log_level_choice=log_level_choice,
                            x_min_fit=x_min_fit,
                            x_max_fit=x_max_fit,
                        )

                        st.session_state["fitted_lg"] = lg
                        st.session_state["report_text"] = report_text
                        # A completed fit supersedes the initial-guess preview.
                        st.session_state.pop("preview_lg", None)
                        # One-shot: forces the Fit Report expander open on the
                        # very next render, then app.py consumes (pops) it.
                        st.session_state["_just_fit"] = True

                        st.toast("Fit completed successfully!", icon="✅")
                        for w in link_warnings:
                            st.warning(w)

                    except Exception as e:
                        st.error(f"Fit failed: {e}")

        # Persisted preview: survives reruns, steps aside once a real fit exists.
        if "preview_lg" in st.session_state and "fitted_lg" not in st.session_state:
            st.caption("Preview: current initial guess vs. data (not yet fitted)")
            try:
                plt.close("all")  # ensure a clean slate before plot_init() creates its figure
                pretty_kw={'width': 8, 'height':5, 'dpi':100} # width and height and dpi of figure, or None to use default settings
                st.session_state["preview_lg"].plot_init(
                    show=False, plot_residual=False, 
                    numpoints=int(n_points_eval), pretty_kw=pretty_kw
                )
                # plot_init() returns None -- grab matplotlib's current
                # figure instead (see core.py's _plot_what, which never
                # returns the Axes it builds).
                fig = plt.gcf()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Could not render preview: {e}")