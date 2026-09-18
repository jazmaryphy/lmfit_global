# %%
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

# %%
def render_thin_divider(margin: str = "0.35rem 0") -> None:
    """Compact horizontal rule with a controllable margin -- st.divider()
    has a fixed, larger built-in margin that's too loose for tightly
    packed sidebar sections."""
    st.markdown(
        f'<hr style="margin:{margin}; border:none; '
        f'border-top:1px solid rgba(250,250,250,0.15);">',
        unsafe_allow_html=True,
    )


def render_subsection_label(
    title: str,
    color: str = "#7dd3fc",
    font_size: str = "0.95rem",
) -> None:
    """Bold, colored label for a setting nested under a numbered header
    (no badge) -- keeps it visually subordinate to its parent step."""
    st.markdown(
        f"<span style='font-weight:700; color:{color}; font-size:{font_size};'>{title}</span>",
        unsafe_allow_html=True,
    )
    

def invalidate_stale_fit_state(data_sig: tuple, model_sig: tuple) -> None:
    """Clears cached fit/preview results in session_state whenever the
    data or model signature has changed since the last check, and
    records the new signatures for the next comparison.

    Uses .get(...) (not [...]) so the very first call -- with no prior
    signature recorded -- doesn't raise, and .pop(key, None) (not del)
    so clearing a key that was never set doesn't raise either.
    """
    stale = (
        st.session_state.get("_data_signature") != data_sig
        or st.session_state.get("_model_signature") != model_sig
    )
    if stale:
        for key in ("fitted_lg", "report_text", "preview_lg"):
            st.session_state.pop(key, None)
        st.session_state["_data_signature"] = data_sig
        st.session_state["_model_signature"] = model_sig

# %%
def render_fancy_header(
    title: str,
    step_number: int | str | None = None,
    subtitle: str | None = None,
    level: int = 2,
    title_color: str | None = None,
    title_size: str | None = None,
    badge_size: str | None = None,
    subtitle_size: str | None = None,
    title_margin: str = "0.5rem 0 0.25rem",
):
    """Renders styled headers using native Streamlit color syntax for subheaders.

    title_size / badge_size / subtitle_size / title_margin are optional
    overrides on top of the level-based defaults -- pass e.g.
    title_size="0.95rem" to render a small, unnumbered sub-label nested
    under a numbered section (step_number=None forces this HTML path
    instead of native "### :color[...]" markdown, since that syntax has
    no way to set an arbitrary font size).
    """
    # Level 3+ Subheaders: native Streamlit color syntax prevents DOM/code-block
    # bugs, but only supports heading levels, not arbitrary font sizes -- so a
    # custom title_size falls back to HTML, same as the level 1/2 branch.
    if level >= 3 or step_number is None:
        active_color = title_color if title_color is not None else "gray"
        if title_size is None:
            st.markdown(f"### :{active_color}[{title}]")
        else:
            st.markdown(
                f'<div style="color:{active_color}; font-size:{title_size}; '
                f'font-weight:700; margin:{title_margin};">{title}</div>',
                unsafe_allow_html=True,
            )
        if subtitle:
            st.caption(subtitle)
        return

    # Level 1 & 2 Main Sections: Flat HTML with step badges
    default_colors = {1: "#f9fafb", 2: "#38bdf8"}
    active_color = title_color or default_colors.get(level, "#38bdf8")

    default_title_sizes = {1: "2.0rem", 2: "1.3rem"}
    default_badge_sizes = {1: "1.0rem", 2: "0.85rem"}

    active_title_size = title_size or default_title_sizes.get(level, "1.3rem")
    active_badge_size = badge_size or default_badge_sizes.get(level, "0.85rem")
    active_subtitle_size = subtitle_size or "0.85rem"

    badge_style = (
        "background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); "
        "color: #ffffff; "
        f"font-size: {active_badge_size}; "
        "font-weight: 700; "
        "padding: 0.15rem 0.55rem; "
        "border-radius: 6px; "
        "display: inline-flex; "
        "align-items: center; "
        "justify-content: center;"
    )
    badge_html = f'<span style="{badge_style}">{step_number}</span>'

    subtitle_html = (
        f'<div style="color: #9ca3af; font-size: {active_subtitle_size}; '
        f'margin-top: 0.25rem; font-weight: 400;">{subtitle}</div>'
        if subtitle else ""
    )

    title_style = (
        f"color: {active_color}; "
        f"font-size: {active_title_size}; "
        "font-weight: 600; "
        "margin: 0; "
        "padding: 0; "
        "line-height: 1.2;"
    )

    flat_html = (
        f'<div style="margin-top: 1.0rem; margin-bottom: 0.6rem;">'
        f'<div style="display: flex; align-items: center; gap: 0.5rem;">'
        f'{badge_html}<span style="{title_style}">{title}</span>'
        f'</div>{subtitle_html}</div>'
    )

    st.markdown(flat_html, unsafe_allow_html=True)

# %%
def sanitize_label(s: str) -> str:
    """Turn an arbitrary string into a safe column-name / filename fragment."""
    s = re.sub(r"[^0-9A-Za-z_]+", "_", str(s).strip())
    return re.sub(r"_+", "_", s).strip("_") or "dataset"


def component_label(comp_name: str, component_choices: list[str]) -> str:
    """Map internal component key to readable function label."""
    m = re.search(r"(\d+)", str(comp_name))
    if m:
        idx = int(m.group(1))
        if 0 <= idx < len(component_choices):
            return sanitize_label(component_choices[idx])
    return sanitize_label(str(comp_name).replace("c", "Comp_").title())


def component_short_tag(comp_name: str, component_choices: list[str]) -> str:
    """Compact tag for export headers."""
    m = re.search(r"(\d+)", str(comp_name))
    idx = int(m.group(1)) if m else 0
    if 0 <= idx < len(component_choices):
        base = re.sub(r"[^0-9A-Za-z]", "", component_choices[idx]).lower()[:4] or "comp"
    else:
        base = "comp"
    return f"{base}{idx}"


def to_fixed_width(df: pd.DataFrame, float_fmt: str = "%14.6e", min_col_width: int = 16) -> str:
    """Render DataFrame as whitespace-padded, right-aligned plain text dat format."""
    sample_width = len(float_fmt % 0)
    col_widths = [
        max(min_col_width, sample_width + 2, len(str(col)) + 2)
        for col in df.columns
    ]

    lines = ["# " + "".join(f"{col:>{w}}" for col, w in zip(df.columns, col_widths)).strip()]
    for _, row in df.iterrows():
        cells = [
            f"{(float_fmt % v):>{w}}" if isinstance(v, (int, float, np.floating, np.integer))
            else f"{str(v):>{w}}"
            for v, w in zip(row, col_widths)
        ]
        lines.append("".join(cells))
    return "\n".join(lines)