# %%
import functools as ft

# %%
def ensureDependency(pkg_name, import_name=None):
    import_name = import_name or pkg_name

    def decorator(func):
        @ft.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                __import__(import_name)
            except (ImportError, ModuleNotFoundError):
                raise ImportError(
                    f"{func.__name__} requires the '{pkg_name}' package. "
                    f"Install it with: pip install {pkg_name}"
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def ensureMatplotlib(func):
    """
    Decorator that ensures matplotlib is installed before calling the function.

    Raises:
        ImportError:
            If matplotlib is not available.
    """
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                f"{func.__name__} requires the 'matplotlib' package. "
                "Install it with: pip install matplotlib"
            )
        return func(*args, **kwargs)

    return wrapper


def ensurePandas(func):
    """
    Decorator that ensures pandas is installed before calling the function.

    Raises:
        ImportError:
            If pandas is not available.
    """
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import pandas as pd  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            raise ImportError(
                f"{func.__name__} requires the 'pandas' package. "
                "Install it with: pip install pandas"
            )
        return func(*args, **kwargs)

    return wrapper