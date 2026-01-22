# docs/conf.py
from __future__ import annotations
import os
import sys

# -------------------------------------------------
# Path setup
# -------------------------------------------------
sys.path.insert(0, os.path.abspath(".."))

# -------------------------------------------------
# Project information
# -------------------------------------------------
project = "lmfit-global"
author = "lmfit-global developers"
copyright = "2025"
release = "0.1.0"

# -------------------------------------------------
# Extensions
# -------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True

# -------------------------------------------------
# MyST configuration
# -------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
    "dollarmath",
]

# -------------------------------------------------
# Napoleon (Google-style docstrings)
# -------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# -------------------------------------------------
# HTML output
# -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_title = "lmfit-global Documentation"

# -------------------------------------------------
# Warnings
# -------------------------------------------------
suppress_warnings = ["myst.header"]
