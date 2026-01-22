# LmfitGlobal

`LmfitGlobal` is the **core engine** of the **lmfit-global** package.

It provides a unified interface for performing **global (single-run) curve
fitting** across **multiple datasets** and **multiple model components**
using [`lmfit`](https://lmfit.github.io/lmfit-py/).

---

## Responsibilities

`LmfitGlobal` manages the complete fitting lifecycle:

- Parsing and validating input data
- Constructing lmfit `Model` and `CompositeModel` objects
- Initializing, sharing, and constraining parameters
- Performing global minimization
- Computing diagnostics and goodness-of-fit metrics
- Evaluating model predictions and uncertainty bands

---

## API Reference

```{autoclass} lmfit_global.LmfitGlobal
:members:
:undoc-members:
:show-inheritance:
```

---

## Class Overview

```python
class LmfitGlobal:
```

```python
LmfitGlobal(
    items: dict,
    independent_vars: list[str] | None = None,
    nan_policy: str = "raise",
    fit_method: str = "leastsq",
    logger=None,
    log_level: str = "",
    **fit_kws
)
```

## Input data handling

```python
items["data"] = {
    "xy": np.column_stack([x, y1, y2, ...]),
    "xrange": (xmin, xmax) | None
}

items["functions"] = {
    "theory": [
        {
            "func_name": gaussian,
            "init_params": {...},
            "func_kws": {}
        },
        {
            "func_name": background,
            "init_params": {...}
        }
    ],
    "theory_connectors": ["+"]
}


lg = LmfitGlobal(items, log_level="info")
# lg = LmfitGlobal(items, log_level="info", nan_polity="raise")
lg.set_data(x, np.column_stack([y1, y2, ...]))   # if not define in items above 
lg.set_xrange(xmin, xmax)                        # if not define in items above  
# lg.set_nan_polity("omit")    # if data has nans
lg.fit(nan_polity="omit")      # or set nan poilicy here
lg.report()
lg.plot()
```

