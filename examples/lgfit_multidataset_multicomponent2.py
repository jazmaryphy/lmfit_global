# %% [markdown]
# # Global multi-dataset fit example
# 
# We consider $N=2$ datasets sharing a common Gaussian component
# with dataset-dependent amplitudes and independent linear backgrounds.
# 
# $$
# y_d(x) =
# A_d \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)
# + m_d x + c_d
# $$
# 
# The parameters $\mu$ and $\sigma$ are shared across datasets,
# while $A_d$, $m_d$, and $c_d$ vary per dataset.

# %%
try:
    from lmfit_global import LmfitGlobal
    from lmfit_global.utils.builders import GlobalFitBuilder
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..")  # parent of examples
    sys.path.insert(0, ROOT)

    from lmfit_global import LmfitGlobal
    from lmfit_global.utils.builders import GlobalFitBuilder

import numpy as np
import matplotlib.pyplot as plt

# %% [markdown]
# ### (1) Generate synthetic multi-dataset, multi-component data

# %%
rng = np.random.default_rng(4)

# -------------------------
# True shared parameters
# -------------------------
mu_true = 2.0
sigma_true = 0.35

# Dataset-specific parameters
params_true = {
    0: dict(A=3.0, m=0.3, c=0.5),
    1: dict(A=2.0, m=-0.2, c=1.0),
}

# -------------------------
# Component functions
# -------------------------
def gaussian(x, amplitude, center, sigma):
    return amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))

def background(x, slope, intercept):
    return slope * x + intercept

# -------------------------
# Dataset x-grids (partially overlapping)
# -------------------------
xdat_lst = [
    np.linspace(0.0, 4.0, 101),
    np.linspace(1.0, 5.0, 150),
]

ydat_lst = []
yerr_lst = []

noise_level = 0.15

for i, x in enumerate(xdat_lst):
    p = params_true[i]

    y_gauss = gaussian(x, p["A"], mu_true, sigma_true)
    y_bg =    background(x, p["m"], p["c"])

    y = y_gauss + y_bg
    yerr = noise_level * np.ones_like(y)

    y_noisy = y + rng.normal(0, yerr)

    ydat_lst.append(y_noisy)
    yerr_lst.append(yerr)

    plt.errorbar(x, y_noisy, yerr=yerr, fmt='o')

plt.title("multi-data with un-equal x size")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

# %% [markdown]
# ### (2) Merge datasets onto a shared x-grid
# 
# using `utils.io_utils.merge_xyerr_data`

# %%
from lmfit_global.utils.io_utils import merge_xyerr_data

x, y, yerr = merge_xyerr_data(
    xdat_lst=xdat_lst,
    ydat_lst=ydat_lst,
    yerr_lst=yerr_lst,
)

print(x.shape, y.shape)


# %% [markdown]
# ### (3) Build `items` dictionary for `LmfitGlobal`

# %%
items_without_xy = {
    # "data": {
    #     "xy": np.column_stack([x, y]),
    # },
    "functions": {
        "theory": [
            {
                "func_name": gaussian,
                "init_params": {
                    "amplitude": dict(value=2.5, min=0),
                    "center": dict(value=1.8),
                    "sigma": dict(value=0.4, min=0.05),                    
                },
            },
            {
                "func_name": background,
                "init_params": {
                    "slope": dict(value=0.1),
                    "intercept": dict(value=0.3),                 
                },
            },
        ],
        "theory_connectors": ["+"],
    },
}

lg = LmfitGlobal(items_without_xy, log_level='info')
lg.theory_expr # or lg.pretty_expr

# %% [markdown]
# ### (4) `items: WARNINGS` set data as `.set_data(x, y)`

# %%
lg.set_data(x, y)

# %% [markdown]
# ### (4)* `items: WARNINGS` 2 set nan policy `.set_nan_policy("omit")`

# %%
lg.set_nan_policy("omit")

# %% [markdown]
# ### (5) pretty print initial parameters
# 
# shared global parameters:
# 
# ```bash
# c0_center_0 = c0_center_1
# c0_sigma_0 =  c0_sigma_1
# ```

# %%
def link_global(lg, names):
    lg.set_global(names, overwrite_expr=True)

ny = 2 # number of datasets
link_global(lg, [f"c0_center_{i}" for i in range(ny)])
link_global(lg, [f"c0_sigma_{i}" for i in range(ny)])

lg.init_params.pretty_print()

# %% [markdown]
# ### (6) fit, report and plot ...

# %%
lg.fit(nan_policy="omit")  # or use nan_policy="omit" in fit()
result = lg.result # result Minimizer

# %%
lg.report()

# %%
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings

# lg.plot()
lg.plot(numpoints=1024, xlabel='x', ylabel='y', pretty_kw=pretty_kw)
plt.show()

# %% [markdown]
# ### (7) Extract structured results with `FitData`

# %%
fitdata = lg.get_fitdata(numpoints=600)

print("Number of Datasets:", fitdata.n_datasets)
print("Components Names:", fitdata.component_names)

# %% [markdown]
# ### (8) Plot components per dataset (multi-panel)

# %%
n = fitdata.n_datasets
fig, axes = plt.subplots(1, n, figsize=(5*n, 4), sharey=True)

if n == 1:
    axes = [axes]

for dset, ax in enumerate(axes):
    ax.errorbar(
        fitdata.x_data,
        fitdata.y_data[:, dset],
        yerr=yerr[:, dset],
        fmt="o",
        label="data",
        alpha=0.7,
    )

    ax.plot(
        fitdata.x_model,
        fitdata.y_fit[:, dset],
        "k-",
        lw=2,
        label="total fit",
    )

    for cname, comp in fitdata.components[dset].items():
        ax.plot(
            fitdata.x_model,
            comp["model"],
            "--",
            lw=1.8,
            label=cname,
        )

    ax.set_title(f"Dataset {dset}")
    ax.set_xlabel("x")
    ax.legend()

axes[0].set_ylabel("y")
plt.tight_layout()
plt.show()


# %% [markdown]
# ### (9) (Optional) Uncertainty bands

# %%
lg.eval_uncertainty(sigma=3)

dely = lg.dely
dely_comps = lg.dely_comps

# %% [markdown]
# ### (10) (Optional) Exports

# %%
lg.export("params")
## OR WITH ###
# lg.to_dataframe()

# %%
res_dict = lg.export("dict")
## OR WITH ###
# res_dict = lg.to_dict()
# res_dict

# %%
df = lg.export("data")  
df = lg.export("data", fitdata_kws={"numpoints": 1024})
## OR WITH ###
# df = lg.data_to_dataframe()
# df = lg.data_to_dataframe(fitdata_kws={"numpoints": 1024})
df

# %%
res_json = lg.export("json", indent=2)
## OR WITH ###
# res_json = lg.to_json()
# res_json

# %%
x_data, x_fit, y_data, y_fit, resid = lg.export("numpy")
x_data, x_fit, y_data, y_fit, resid = lg.export("numpy", fitdata_kws={"numpoints": 1024})
## OR WITH ###
# x_data, x_fit, y_data, y_fit, resid = lg.to_numpy()
# x_data, x_fit, y_data, y_fit, resid = lg.to_numpy(fitdata_kws={"numpoints": 1024})
# resid