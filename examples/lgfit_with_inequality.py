#!/usr/bin/env python
# coding: utf-8

# # Fit Using Inequality Constraint
# 
# This example shows how to use `lmfit_global.LmfitGlobal` class to fit data with inequality constraints by making use of `lmfit Parameter`(s) `expr`. 
# 
# This example is similar to `example_fit_with_inequality.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# In[1]:


try:
    from lmfit_global import LmfitGlobal
    from lmfit_global.utils.builders import GlobalFitBuilder
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..")  # parent of examples
    sys.path.insert(0, ROOT)

    from lmfit_global import LmfitGlobal
    from lmfit_global.utils.builders import GlobalFitBuilder

import matplotlib
import matplotlib.pyplot as plt


# # Define `LmfitGlobal` class `items` data
# 
# First, create raw data... (always in nd.array with shape N, ny)

# In[2]:


import numpy as np
log2 = np.log(2)
s2pi = np.sqrt(2*np.pi)
s2 = np.sqrt(2.0)
# tiny had been numpy.finfo(numpy.float64).eps ~=2.2e16.
# here, we explicitly set it to 1.e-15 == numpy.finfo(numpy.float64).resolution
tiny = 1.0e-15

def not_zero(value):
    """Return value with a minimal absolute size of tiny, preserving the sign.

    This is a helper function to prevent ZeroDivisionError's.

    Args::
        value (float): Value to be ensured not to be zero.

    Returns:
        float: Value ensured not to be zero.

    """
    return float(np.copysign(max(tiny, abs(value)), value))

def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * np.exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))

def lorentzian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Lorentzian function.

    lorentzian(x, amplitude, center, sigma) =
        (amplitude/(1 + ((1.0*x-center)/sigma)**2)) / (pi*sigma)

    """
    return ((amplitude/(1 + ((1.0*x-center)/max(tiny, sigma))**2))
            / max(tiny, (np.pi*sigma)))


np.random.seed(0)
x = np.linspace(0, 20.0, 601)

data = (
    gaussian(x, 21, 6.1, 1.2) + lorentzian(x, 10, 9.6, 1.3) +
    np.random.normal(scale=0.1, size=x.size)
        )

xy = np.column_stack([x, data])

plt.plot(x, data, 'o')
plt.show()


# finally, built `data` and `function` **`items`** `dict` for `LmfitGlobal` ...

# In[3]:


init_gauss = {
    'amplitude': {'value':20.0, 'vary':True, 'min':-np.inf, 'max':+np.inf},
    'center': {'value':5, },
    'sigma': {'value':1, },
}

init_loren = {
        'amplitude': {'value':8.0, 'vary':True, 'min':-np.inf, 'max':+np.inf},
        'center': {'value':8, },
        'sigma': {'value':1, },
}


# In[4]:


# data dict
data_dict = {
    'xy': xy,         # data_xy, i.e numpy.column_stack([x, y_0, y_1, ..., y_n])
    'xrange': None    # x range in (min, max) of the data range to fit, default is None
    }

func_lst = [
    {
        'func_name': gaussian,
        'init_params' : init_gauss,
        'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
    },
    {
        'func_name': lorentzian,
        'init_params' : init_loren,
        'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
    },
]

# function dict
function_dict = {
    'theory': func_lst,
    'theory_connectors': ['+'],
}


# items 
items = {
    'data': data_dict,              # 1. data (see above)
    'functions': function_dict,     # 2. thoery (see above)
}


# In[5]:


builder = (
    GlobalFitBuilder()
    .set_data(x, data)                      # x and all y datasets
    .add_model(gaussian,   init_gauss, func_kws={})
    .add_model(lorentzian, init_loren, func_kws={})       
    .connect("+")                             # how to combine the 2 functions
)

items = builder.build()


# fit with iter callback...

# In[6]:


def make_iter_cb(every: int = 1):
    def iter_cb(params, iteration, resid, *args, **kws):
        if iteration % every != 0:
            return

        print(
            f"iter={iteration}",
            ", ".join(
                f"{p.name}={p.value:.5f}"
                for p in params.values()
                if p.vary
            )
        )

    return iter_cb

per_iteration = make_iter_cb(every=5)   # every 5 iterations


# call `LmfitGlobal` class ...

# In[7]:


lg = LmfitGlobal(items, log_level='info')
lg.fit(iter_cb=per_iteration)


# In[8]:


lg.report()
lg.plot()
plt.show()


# #### Applying an Inequality Constraint
# 
# In this step, we extend the global fitting model by introducing a new parameter and enforcing a physically motivated constraint between two peak-center parameters.
# 
# ---
# 
# ##### **1. Adding the Peak-Splitting Parameter `alpha`**
# 
# We introduce a new global fitting parameter called **`alpha`**, representing the peak splitting between two components.  
# This parameter is defined with the following properties:
# 
# - It is **global**, i.e., not dataset‑dependent.
# - It is allowed to vary between **`'min' = 0`** and **`'max' = 5`**.
# - It is intentionally named **`alpha`** (without the underscore `_0` suffix .i.e. **`alpha_0`**) to ensure that it is *not* interpreted as a dataset‑indexed parameter by the `LmfitGlobal()`.
# 
# We can add non existing parameter as:
# ```python
# LmfitGlobal.add_par(
#         self,
#         *parlist: Iterable
#     )
# ```
# 
# where  `parlist` is of type: 
# ```python
# *parlist: Union[str, list[str], lmfit.Parameter, lmfit.Parameters, Iterable, Dict]
# ```
# 
# and 
# ```python
# parlist = {
#   "alpha": {"value": 2.5,  "min":0, "max": 5, "vary": True}
#   }
# ```
# 
# This parameter will later be used to impose a constraint on the second peak center.
# 
# ---
# 
# ##### **2. Constraining `c1_center_0` Using an Expression**
# 
# After defining `alpha`, we enforce a relationship between the two peak centers: `c1_center_0` $=$ `c0_center_0` $ +\alpha$
# 
# This ensures that:
# 
# - The second peak center (`c1_center_0`) is always shifted relative to the first (`c0_center_0`).
# - The shift is controlled by the global parameter `alpha`.
# - The inequality constraint  `c1_center_0` $\ge$ `c0_center_0`
#   is automatically satisfied because `alpha ≥ 0`.
# 
# To apply this parameter constraint, we use:
# 
# ```python
# LmfitGlobal.set_expr(
#         self,
#         mapping: dict[str, str | None],
#         *,
#         overwrite_expr: bool = False
#     )
# ```
# 
# where
# ```python
# mapping   = {
#                 "c1_center_0": "alpha + c0_center_0", 
#                 "c1_sigma_0": "c0_sigma_0"
#             }
# ```
# 
# Note: the case for   `"c1_sigma_0": "c0_sigma_0"` is similar to use to the use of `LmfitGlobal.set_global_params()` in previous examples

# In[9]:


lg.rebuild()  # rebuild LmfitGlobal

parlist = {
  "alpha": {"value": 2.5,  "min":0, "max": 5, "vary": True}
  }
lg.add_par(parlist)  # add new non existing parameters from fit functions

# --- set "inequality" contrain below
mapping   = {
                "c1_center_0": "alpha + c0_center_0", 
                "c1_sigma_0": "c0_sigma_0"
            }
lg.set_expr(mapping, overwrite_expr=True)

lg.init_params.pretty_print()  # pretty print initial parameters 


# In[10]:


lg.fit(verbose=True, iter_cb=per_iteration)  # verbose, if True will show fit parameters


# In[11]:


def _data_label(i: int, is_multidataset: bool) -> str | None:
    if is_multidataset:
        return f"data {i+1}"
    return "data" if i == 0 else None

def _fit_label(i: int, is_multidataset: bool, same_fit_color: bool) -> str | None:
    if not is_multidataset:
        return "fit" if i == 0 else None
    if same_fit_color:
        return "fit" if i == 0 else None
    return f"fit {i+1}"


fd = lg.get_fitdata(numpoints=1024) # int or None
fd.x_data
fd.resid_fit

fig, ax = plt.subplots()


for i in range(fd.ny):
    # --- data ---
    ax.plot(
        fd.x_data,
        fd.y_data[:, i],
        'o',
        zorder=1,
        label=_data_label(i, fd.is_multidataset),
    )

    # --- fit ---
    ax.plot(
        fd.x_model,
        fd.y_fit[:, i],
        '-',
        lw=2,
        color='k',
        zorder=2,
        label=_fit_label(i, fd.is_multidataset, same_fit_color=True)
        # label="fit" if i == fd.ny-1 else None, # ONE legend entry only
    )

    # --- components ---
    comps = None
    comps_names = None
    if fd.is_multicomponent:
        comps = fd.components
        comps_names = fd.component_names
        if fd.is_multidataset:
            comp = comps[i]
        else:
            comp = comps
        for name in comps_names:
            d_dict = comp[name]
            ax.plot(fd.x_data,  d_dict['data'],  '--', zorder=3, label=name)
            # ax.plot(fd.x_model, d_dict['model'], '--', label=name)

fontsize=16
ax.minorticks_on()
ax.set_xlabel('x', fontsize=fontsize+4)
ax.set_ylabel('y', fontsize=fontsize+4)
ax.tick_params(axis='x', labelsize=fontsize, labelcolor='k')  # Increase x-axis tick font size
ax.tick_params(axis='y', labelsize=fontsize, labelcolor='k')  # Increase y-axis tick font size
ax.tick_params(direction="in", which="both", top=True, right=True, labelsize=fontsize, labelcolor="k")
ax.tick_params(axis="both", which="major", length=10, width=1.0)
ax.tick_params(axis="both", which="minor", length=5,  width=1.0)
for spine in ax.spines.values():
    spine.set_linewidth(2)

# ax.legend()
ax.legend(
    prop={
        "size": fontsize,
        "family": "sans-serif",
        "weight": "normal",
    },
    frameon=False,
)
plt.show()

# see next examples on how we can handle multi-component and multi-dataset fit

