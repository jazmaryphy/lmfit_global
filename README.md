# lmfit-global
[lmfit-global](https://github.com/jazmaryphy/lmfit_global/tree/main) is a Global (or one-run-fit) curve fitting code built on top of [LMFIT](https://lmfit.github.io/lmfit-py/index.html) to fit multi-components or multi-datasets problems for Python.

[LMFIT](https://lmfit.github.io/lmfit-py/index.html) stands for **L**evenberg–**M**arquardt **FIT**ting.

**Please note**: the code is at experimental stages...

## Dependencies
To run the LmFitClobal, lmfit and numpy library are required.

## Code structure and utilities
```
lmfit_global/
├── __init__.py
│
├── lmfit_global/                
│   ├── __init__.py
│   │
│   ├── utils/                   
│   │   ├── __init__.py
│   │   ├── builders.py
│   │   ├── io_utils.py
│   │   ├── parameters.py
│   │   ├── lineshapes.py
│   │   ├── plotting.py
│   │   └── reporting.py
│   │
│   ├── lmfit_global.py             
│
├── tests/                       
├── examples/                   
```

## Installation
install this repository as:

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