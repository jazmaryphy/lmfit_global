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

---


## Dependencies

### Core Requirements
- [**lmfit**](https://lmfit.github.io/lmfit-py/installation.html)
- [**numpy**](https://numpy.org/)
- [**scipy**](https://scipy.org/)

### Optional Extras
Optional functionality can be enabled via `pip` install tags:
- **`.[gui]`** — Installs [**streamlit**](https://streamlit.io/), [**pandas**](https://pandas.pydata.org/), and [**matplotlib**](https://matplotlib.org/) to run the `lmfitgedit` web interface.
- **`.[plot]`** — Installs [**matplotlib**](https://matplotlib.org/) and [**palettable**](https://jiffyclub.github.io/palettable/) for advanced visualizations.
- **`.[ml]`** — Installs [**scikit-learn**](https://scikit-learn.org/) for additional statistical utilities.

---


## Code Structure
```bash
lmfit_global/
│
├── lmfit_global/           # Core Python Library
│   ├── core.py             # Core global fitting engine
│   ├── simplefit.py        # Lightweight SciPy-like fitting interface
│   └── utils/              # Internal utilities (I/O, plotting, builders)
│
├── gui/                    # Streamlit Web Application (lmfitgedit), EXPERIMENTAL!!!
│   ├── app.py              # Main Streamlit UI layout
│   ├── cli.py              # CLI launcher entry point
│   └── src/                # UI components and session handlers
│
├── tests/                  # Test suite
└── examples/               # Code usage examples and tutorials
```


<!--
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

---
-->


## Installation
Clone the repository and install in editable mode:

```bash
git clone https://github.com/jazmaryphy/lmfit_global.git 
cd lmfit_global/
pip install -e .
```

To include the GUI interface during installation, use the `[gui]` extra:

```bash
pip install -e .[gui]
```

## GUI (`lmfitgedit`)
`lmfit-global` includes an interactive, browser-based editor and visualizer called `lmfitgedit`.

Once installed with the `[gui]` extra, launch the interactive application directly from any terminal window:


```bash
lmfitgedit
```

## Example

Explore multi-dataset templates and tutorials in the `examples/` directory:

```bash
cd examples/
```

## License

MIT — Created by [Muhammad Maikudi ISAH](https://jazmaryphy.github.io/)

### Contact & Links

- **Website:** [jazmaryphy.github.io](https://jazmaryphy.github.io/)
- **LinkedIn:** [linkedin.com/in/iammisah](https://www.linkedin.com/in/iammisah)
- **X (Formally Twitter):** [@heyitsmisah](https://x.com/heyitsmisah)