# ModelSpec

`ModelSpec` defines a **single model component** used by `LmfitGlobal` to construct composite models for global fitting.

It is a **pure specification object**: it describes *what* a model is,
not *how* it is fitted.

---

## Purpose

`ModelSpec` is responsible for:

- Coupling a **model function** with:
  - its **initial parameter definitions**
  - optional **fixed keyword arguments**
- Providing a uniform interface for building lmfit `Model` objects
- Enabling clean composition of multi-component models

Each `ModelSpec` corresponds to **one functional component** in the fit
(e.g. Gaussian peak, background, exponential decay, label as "`c0`", "`c1`", "`c2`", respectively).

---

## Structure

```python
@dataclass(slots=True)
class ModelSpec:
    func: Callable
    init_params: dict
    func_kws: dict = field(default_factory=dict)
```