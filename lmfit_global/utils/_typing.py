# %%
from __future__ import annotations
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from lmfit_global.core import LmfitGlobal

LmfitGlobalLike: TypeAlias = "LmfitGlobal"