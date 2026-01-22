# %%
import warnings

# %%
def deprecated(name: str, replacement: str | None = None):
    msg = f"'{name}' is deprecated"
    if replacement:
        msg += f"; use '{replacement}' instead"
    msg += ". This will be removed in a future release."

    warnings.warn(msg, DeprecationWarning, stacklevel=3)