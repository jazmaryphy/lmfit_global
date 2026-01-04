# lmfit-global

[lmfit-global](https://github.com/jazmaryphy/lmfit_global/tree/main) is a **global (single-run) curve fitting** framework built on top of
[LMFIT](https://lmfit.github.io/lmfit-py/index.html) for solving **multi-component** and/or **multi-dataset** fitting problems in Python.

It is designed for cases where:
- multiple datasets must be fitted **simultaneously**
- parameters are **shared or constrained** across datasets
- models are composed of **multiple functional components**

[LMFIT](https://lmfit.github.io/lmfit-py/index.html) refers to the Python fitting library based on
**L**evenberg–**M**arquardt **FIT**ting and related optimization algorithms.

> **Please note**  
> This project is currently in an **experimental stage**...  
> APIs and internal behavior may change.

## Dependencies

### Required
- [**lmfit**](https://lmfit.github.io/lmfit-py/installation.html)
- [**numpy**](https://numpy.org/)

### Optional
- [**scikit-learn**](https://scikit-learn.org/) — statistics and analysis utilities
- [**matplotlib**](https://matplotlib.org/) — plotting and visualization


## Code structure and utilities
```
lmfit_global/
│
├── lmfit_global/                
│   ├── __init__.py
│   │
│   ├── utils/                   
│   │   ├── __init__.py
│   │   ├── builders.py
│   │   ├── io_utils.py
│   │   ├── plotting.py
│   │   ├── reporting.py
│   │   ├── parameters.py
│   │   └── lineshapes.py
│   │
│   ├── lmfit_global.py       # Core LmfitGlobal implementation       
│
├── tests/                    # Test Suite           
├── examples/                 # example folder             
```

## Installation
Clone the repository and install in editable mode:

```
git clone https://github.com/jazmaryphy/lmfit_global.git
cd lmfit_global/
pip install -e .
```
## Example

You can many template examples in examples folder by:

```
cd examples/

```