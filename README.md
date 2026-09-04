<p align="center">
  <img src="https://raw.githubusercontent.com/jazmaryphy/lmfit_global/main/docs/source/images/logo2.png"
       alt="lmfit-global logo"
       width="400px" 
       height="400px"
  >
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
- [**scipy**](https://scipy.org/)

### Optional
- [**scipy**](https://scipy.org/) — used for evaluating uncertainty bands  
- [**scikit-learn**](https://scikit-learn.org/) — provides additional statistical and analysis utilities  
- [**matplotlib**](https://matplotlib.org/) — enables plotting and visualization  
> To run the test suite, the [**pytest**](https://docs.pytest.org/en/stable/) package is required.
> To run the GUI, the [**streamlit**](https://streamlit.io/) package is required.


## Code structure and utilities
```bash
lmfit_global/
│
├── lmfit_global/
│   ├── __init__.py
│   │
│   ├── core.py             # Core global fitting engine
│   ├── simplefit.py        # Lightweight SciPy-like fitting interface
│   │
│   ├── utils/              # Internal utilities (I/O, plotting, builders, etc)
│   │   └── ...
│
├── tests/                  # Test suite
├── examples/               # Usage examples and tutorials
│
├── gui/                    # <-- GUI Application Package with Streamlit (EXPERIMENTAL!!!)
```


## Installation
Clone the repository and install in editable mode:

```bash
git clone https://github.com/jazmaryphy/lmfit_global.git
cd lmfit_global/
pip install -e .
```
## Example

Many template examples in examples folder by:

```bash
cd examples/
```

## License

MIT — Created by [Muhammad Maikudi ISAH](https://jazmaryphy.github.io/)

### 📬 Contact & Links

- **Website:** [jazmaryphy.github.io](https://jazmaryphy.github.io/)
- **LinkedIn:** [linkedin.com/in/iammisah](https://www.linkedin.com/in/iammisah)
- **X (Formally Twitter):** [@heyitsmisah](https://x.com/heyitsmisah)