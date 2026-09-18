# %%
from __future__ import annotations

import re
import numpy as np
from scipy import special as sps
from asteval import Interpreter

from gui.src.utils import sanitize_label

# %%
# Curated scipy.special functions commonly needed in physics/spectroscopy
# line shapes. Each is added explicitly to asteval's symbol table --
# asteval has NO import mechanism, so nothing outside this whitelist
# (plus asteval's own built-in numpy/math set) is ever reachable.
_SPECIAL_FUNCTIONS = {
    "erf": sps.erf,
    "erfc": sps.erfc,
    "gamma": sps.gamma,
    "gammaln": sps.gammaln,
    "wofz": sps.wofz,               # Faddeeva function -- exact Voigt profiles
    "j0": sps.j0,
    "j1": sps.j1,
    "jv": sps.jv,
    "expi": sps.expi,
    "voigt_profile": sps.voigt_profile,
}

_PYTHON_KEYWORDS = {
    "and", "or", "not", "if", "else", "elif", "for", "in", "is",
    "lambda", "None", "True", "False",
}

_MAX_EXPR_LENGTH = 500  # cheap guard against pathological input

# %%
def _build_base_interpreter() -> Interpreter:
    """A fresh, sandboxed interpreter pre-loaded with the scipy.special
    whitelist -- the shared foundation both parameter-detection and
    actual evaluation are built on."""
    aeval = Interpreter(minimal=True, use_numpy=True)
    for name, fn in _SPECIAL_FUNCTIONS.items():
        aeval.symtable[name] = fn
    return aeval


_RESERVED_NAMES_CACHE: set[str] | None = None


def _get_reserved_names() -> set[str]:
    """Reserved identifiers = whatever the interpreter itself already
    knows (numpy funcs/constants asteval exposes, our scipy.special
    additions) plus Python keywords plus 'x'. Derived from the real
    interpreter rather than a hand-maintained list, so it stays correct
    across asteval versions without manual upkeep. Cached at module
    level since it's identical for every formula and every rerun.
    """
    global _RESERVED_NAMES_CACHE
    if _RESERVED_NAMES_CACHE is None:
        aeval = _build_base_interpreter()
        _RESERVED_NAMES_CACHE = set(aeval.symtable.keys()) | _PYTHON_KEYWORDS | {"x"}
    return _RESERVED_NAMES_CACHE

# %%
def extract_parameter_names(expr: str) -> list[str]:
    """Identifiers in the formula that aren't x, a keyword, or a known
    math/special function -- these become the fittable parameters."""
    identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr))
    return sorted(identifiers - _get_reserved_names())


def build_custom_function(expr: str, param_names: list[str]):
    """Wraps an asteval-evaluated formula as a callable with signature
    (x, **kwargs) -> np.ndarray, matching what FUNCTION_LIBRARY entries
    already expect.

    Efficiency: the interpreter and the formula's parsed AST are built
    ONCE per function (closed over), not re-parsed on every call --
    important since lmfit's minimizer may call this hundreds of times
    per fit (once per residual evaluation per iteration).
    """
    aeval = _build_base_interpreter()
    compiled = aeval.parse(expr)

    def _custom(x, **kwargs):
        x_arr = np.asarray(x, dtype=float)
        aeval.symtable["x"] = x_arr
        for p in param_names:
            aeval.symtable[p] = kwargs.get(p, 1.0)

        aeval.error = []  # clear stale errors from any prior call
        result = aeval.run(compiled)

        if aeval.error:
            msgs = "; ".join(e.get_error()[1] for e in aeval.error)
            raise ValueError(f"Error evaluating custom function: {msgs}")

        result = np.asarray(result, dtype=complex if np.iscomplexobj(result) else float)

        if np.iscomplexobj(result):
            raise ValueError(
                "Formula produced complex values (e.g. sqrt of a negative "
                "number) -- adjust the formula or parameter bounds."
            )

        # Broadcast a scalar (x-independent formula, e.g. just "amplitude")
        # up to x's shape, so it plugs into the fitting machinery the same
        # way an x-dependent formula does.
        if result.shape != x_arr.shape:
            result = np.broadcast_to(result, x_arr.shape).astype(float)

        return result

    _custom.__name__ = "custom_" + sanitize_label(expr)[:24]
    return _custom


def validate_custom_function(expr: str, param_names: list[str]) -> str | None:
    """Evaluates the formula on a small sample x with default parameter
    values, to catch syntax/name/shape errors before it's added to the
    model. Returns an error message, or None if valid.
    """
    if len(expr) > _MAX_EXPR_LENGTH:
        return f"Formula is too long ({len(expr)} chars, max {_MAX_EXPR_LENGTH})."
    try:
        func = build_custom_function(expr, param_names)
        sample_x = np.array([0.0, 1.0, 2.0])
        y = func(sample_x, **{p: 1.0 for p in param_names})
        if not np.all(np.isfinite(y)):
            return "Formula produced non-finite values (inf/nan) at the test points."
    except Exception as e:
        return str(e)
    return None


def make_library_entry(expr: str, param_names: list[str]) -> dict:
    """Builds a FUNCTION_LIBRARY-shaped entry for a validated custom
    formula -- plugs directly into the existing component picker,
    Initial Parameter Editor, and shared-parameters UI unmodified.
    """
    func = build_custom_function(expr, param_names)
    return {
        "func": func,
        "func_name": func.__name__,
        "params": {p: {"value": 1.0} for p in param_names},
        "doc": f"Custom function: {expr}",
    }


def available_special_functions() -> list[str]:
    """Names of whitelisted scipy.special functions, for help text."""
    return sorted(_SPECIAL_FUNCTIONS.keys())