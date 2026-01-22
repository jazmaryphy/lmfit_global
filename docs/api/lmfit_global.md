# LmfitGlobal

`LmfitGlobal` is the **core engine** of the **lmfit-global** package.

It provides a unified interface for performing **global (single-run) curve fitting** across **multiple datasets** and **multiple model components** using [`lmfit`](https://lmfit.github.io/lmfit-py/).

---

## Parameter Naming Convention

`LmfitGlobal` uses a standardized **Prefix-Suffix** system to organize parameters. This allows the engine to handle complex fits where multiple models are applied to multiple datasets simultaneously.

### The Structure
A fully qualified parameter name follows this pattern: **`[Component]_[ParameterName]_[Dataset]`**

| Type | Identifier | Position | Description |
| :--- | :--- | :--- | :--- |
| **Component** | `c0`, `c1`, ... | **Prefix** | Distinguishes individual model components (e.g., Peak 1 vs Peak 2). |
| **Dataset** | `0`, `1`, ... | **Suffix** | Distinguishes the dataset index (Used for both single and multi-dataset fits). |


---

### Examples in Action

#### 1. Dataset Indexing (Suffix)
Whether you have one dataset or many, the dataset index is always appended at the end with an underscore.
* **Single Dataset:** `sigma_0`, `center_0`
* **Multi-Dataset:** `sigma_0`, `sigma_1`, `sigma_2`

#### 2. Component Labeling (Prefix)
When a model consists of multiple parts (like two Gaussians), components are labeled at the beginning.
* **Multi-Component:** `c0_amplitude_0`, `c1_amplitude_0`

#### 3. The Global Fit (Multi-Component + Multi-Dataset)
For the **center** of the **second component** (`c1`) in the **third dataset** (`2`), the parameter name is:
> **`c1_center_2`**

---

### Why this structure?

1.  **Consistency:** Using `_0` even for single datasets means your analysis scripts don't have to change when you add a second dataset.
2.  **Explicit Scoping:** It is immediately clear which component and which dataset a parameter belongs to just by reading its name.
3.  **Easy Regex Parsing:** The use of underscores as delimiters (e.g., `c0_` and `_0`) makes it trivial to extract metadata from parameter lists using Python's `split()` or `re` module.

> [!NOTE]  
> The engine automatically handles the mapping of these names to the underlying mathematical functions during the minimization process.

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
lg.fit(nan_polity="omit")      # or set nan policy here
lg.report()
lg.plot()
```

