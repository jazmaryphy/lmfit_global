
# FitData

`FitData` is a lightweight, immutable container that holds **evaluated fit results**
produced by an `LmfitGlobal` instance.

It separates **numerical results** from the fitting engine itself, making it ideal
for plotting, analysis, and exporting.

---

## Purpose

`FitData` is designed to:

- Store raw input data
- Store evaluated model curves
- Store residuals
- Store per-component contributions
- Provide a clean interface for downstream analysis

It contains **no fitting logic** and performs **no optimization**.

---

## When should you use `FitData`?

Use `FitData` when you want to:

- Plot fit results without touching `LmfitGlobal`
- Access model curves on a dense grid
- Inspect residuals
- Analyze multi-dataset or multi-component fits
- Export results to files or external tools

---

## Structure

```python
@dataclass(slots=True)
class FitData:
    x_data: np.ndarray
    y_data: np.ndarray
    x_model: np.ndarray
    y_init: np.ndarray
    y_fit: np.ndarray | None
    resid_init: np.ndarray
    resid_fit: np.ndarray | None
    components: dict | None
    rsquared: float | None
```