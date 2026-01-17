<p align="center">
  <img src="https://raw.githubusercontent.com/jazmaryphy/lmfit_global/main/docs/source/images/logo2.png"
       alt="lmfit-global"
       width="400px"
       height="300px">
</p>


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
> Internal behavior may change.

## Dependencies

### Required
- [**lmfit**](https://lmfit.github.io/lmfit-py/installation.html)
- [**numpy**](https://numpy.org/)

### Optional
- [**scipy**](https://scipy.org/) — used for evaluating uncertainty bands  
- [**scikit-learn**](https://scikit-learn.org/) — provides additional statistical and analysis utilities  
- [**matplotlib**](https://matplotlib.org/) — enables plotting and visualization  
> To run the test suite, the [**pytest**](https://docs.pytest.org/en/stable/) package is required.

## Code structure and utilities
```
lmfit_global/
│
├── lmfit_global/
│   ├── __init__.py
│   │
│   ├── lmfit_global.py     # Core global fitting engine
│   ├── simplefit.py        # Lightweight SciPy-like fitting interface
│   │
│   ├── utils/              # Internal utilities (I/O, plotting, builders, etc)
│   │   └── ...
│
├── tests/                  # Test suite
├── examples/               # Usage examples and tutorials
```

<!--
```
lmfit_global/
│
├── lmfit_global/                
│   ├── __init__.py
│   │
│   ├── utils/                   
│   │   ├── __init__.py
│   │   ├── fitdata.py
│   │   ├── builders.py
│   │   ├── .
│   │   ├── .
│   │   ├── .
│   │   └── lineshapes.py
│   │
│   ├── lmfit_global.py       # Core LmfitGlobal implementation     
│   ├── simplefit.py          # Analogous to scipy.optimize implementation  
│
├── tests/                    # Test Suite           
├── examples/                 # example folder    
```
-->

<!--
```
lmfit_global/
│
├── lmfit_global/                
│   ├── __init__.py
│   │
│   ├── utils/                   
│   │   ├── __init__.py
│   │   ├── fitdata.py
│   │   ├── builders.py
│   │   ├── io_utils.py
│   │   ├── plotting.py
│   │   ├── modelspec.py
│   │   ├── reporting.py
│   │   ├── parameters.py
│   │   └── lineshapes.py
│   │
│   ├── lmfit_global.py       # Core LmfitGlobal implementation     
│   ├── simplefit.py          # Analogous to scipy.optimize implementation  
│
├── tests/                    # Test Suite           
├── examples/                 # example folder             
```
-->

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