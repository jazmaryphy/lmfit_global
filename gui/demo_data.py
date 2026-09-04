# %%
from __future__ import annotations

import numpy as np
import lmfit_global.utils.lineshapes as ls

# %%
def multi_gaussian_linear(ny: int = 1, n_points: int = 201) -> np.ndarray:
    """Generates `ny` synthetic Gaussian datasets with a linear background."""
    np.random.seed(42)
    
    x = np.linspace(0, 10, n_points)
    xy = np.zeros((len(x), ny + 1))
    xy[:, 0] = x

    test_gaussian = dict(amplitude=8.5, center=6.66, sigma=0.68)
    test_linear   = dict(slope=0.25, intercept=-1.0)

    for i in range(ny):
        # Add slight variations if generating multiple datasets (ny > 1)
        amp_scale = 1.0 if i == 0 else (1.0 + 0.1 * np.random.randn())
        
        y = ls.gaussian(x, amplitude=test_gaussian["amplitude"] * amp_scale, 
                           center=test_gaussian["center"], 
                           sigma=test_gaussian["sigma"])
        y += ls.linear(x, **test_linear)
        y += np.random.normal(scale=0.1, size=x.size)
        
        xy[:, i + 1] = y

    return xy
    

def multi_gaussian(ny: int = 5, n_points: int = 151) -> np.ndarray:
    """Generates `ny` reproducible synthetic Gaussian datasets."""
    np.random.seed(2021)
    
    x = np.linspace(-1, 2, n_points)
    xy = np.zeros((len(x), ny + 1))
    xy[:, 0] = x

    for i in range(ny):
        amplitude = 0.60 + 9.50 * np.random.rand()
        center    = -0.20 + 1.20 * np.random.rand()
        sigma     = 0.25 + 0.03 * np.random.rand()
        
        y = ls.gaussian(x, amplitude, center, sigma) + np.random.normal(scale=0.1, size=x.size)
        xy[:, i + 1] = y

    return xy


def multi_lorentzian(ny: int = 3, n_points: int = 200) -> np.ndarray:
    """Generates `ny` reproducible synthetic Lorentzian datasets."""
    np.random.seed(42)
    
    x = np.linspace(-5, 5, n_points)
    xy = np.zeros((len(x), ny + 1))
    xy[:, 0] = x

    for i in range(ny):
        amplitude = +1.0 + 3.0 * np.random.rand()
        center    = -0.5 + 1.0 * np.random.rand()
        sigma     = +0.3 + 0.1 * np.random.rand()
        
        y = ls.lorentzian(x, amplitude, center, sigma) + np.random.normal(scale=0.05, size=x.size)
        xy[:, i + 1] = y

    return xy

# %%
# Registry mapping friendly UI names to generator functions
DEMO_DATASETS = {
    "Gaussian + Linear Background": multi_gaussian_linear,
    "N-Dataset Gaussian": multi_gaussian,
    "N-Dataset Lorentzian": multi_lorentzian,
}


# def make_demo_data(
#     name: str = "Gaussian + Linear Background", 
#     ny: int = 1, 
#     n_points: int = 201
# ) -> np.ndarray:
#     """Wrapper function executing dataset generator matching target registry name."""
#     generator = DEMO_DATASETS.get(name, multi_gaussian_linear)
#     return generator(ny=ny, n_points=n_points)


def make_demo_data(
    name: str = "N-Dataset Gaussian", 
    ny: int = 5, 
    n_points: int = 151
) -> np.ndarray:
    """Wrapper function executing dataset generator matching target registry name."""
    generator = DEMO_DATASETS.get(name, multi_gaussian)
    return generator(ny=ny, n_points=n_points)