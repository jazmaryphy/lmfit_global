# %%
"""
Runtime dependency loader for lmfit-global.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import lmfit
else:
    def _require_lmfit():
        try:
            import lmfit
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "lmfit-global requires the 'lmfit' package.\n\n"
                "Install it with:\n"
                "    pip install lmfit"
            ) from exc

        return lmfit

    lmfit = _require_lmfit()

__all__ = ["lmfit"]