# %%
from __future__ import annotations

import re
import numpy as np
import pandas as pd
import streamlit as st

# %%
def render_fancy_header(
    title: str, 
    step_number: int | str | None = None, 
    subtitle: str | None = None, 
    level: int = 2,
    title_color: str | None = None
):
    """Renders styled headers using native Streamlit color syntax for subheaders."""
    
    # Level 3+ Subheaders: Native Streamlit color syntax prevents DOM/code-block bugs
    if level >= 3 or step_number is None:
        active_color = title_color if title_color is not None else "gray"
        st.markdown(f"### :{active_color}[{title}]")
        if subtitle:
            st.caption(subtitle)
        return

    # Level 1 & 2 Main Sections: Flat HTML with step badges
    default_colors = {1: "#f9fafb", 2: "#38bdf8"}
    active_color = title_color or default_colors.get(level, "#38bdf8")
    
    font_sizes = {1: "2.0rem", 2: "1.3rem"}
    badge_sizes = {1: "1.0rem", 2: "0.85rem"}
    
    title_size = font_sizes.get(level, "1.3rem")
    badge_size = badge_sizes.get(level, "0.85rem")

    badge_style = (
        "background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%); "
        "color: #ffffff; "
        f"font-size: {badge_size}; "
        "font-weight: 700; "
        "padding: 0.15rem 0.55rem; "
        "border-radius: 6px; "
        "display: inline-flex; "
        "align-items: center; "
        "justify-content: center;"
    )
    badge_html = f'<span style="{badge_style}">{step_number}</span>'
    
    subtitle_html = (
        f'<div style="color: #9ca3af; font-size: 0.85rem; margin-top: 0.25rem; font-weight: 400;">{subtitle}</div>'
        if subtitle else ""
    )
    
    title_style = (
        f"color: {active_color}; "
        f"font-size: {title_size}; "
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