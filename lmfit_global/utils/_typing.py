# %%
from typing import TYPE_CHECKING, TypeAlias

# %%
if TYPE_CHECKING:
    from lmfit_global.lmfit_global import LmfitGlobal


LmfitGlobalLike: TypeAlias = "LmfitGlobal"