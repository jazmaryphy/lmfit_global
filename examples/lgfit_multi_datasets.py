#!/usr/bin/env python
# coding: utf-8

# # Fit Multiple Data Sets
# 
# This example shows how to use `lmfit_global.LmfitGlobal` class to fit multiple (simulated) Gaussian data sets simultaneously. 
# 
# This example is similar to `example_fit_multi_datasets.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# In[1]:


try:
    from lmfit_global import LmfitGlobal
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..") # parent folder of examples
    sys.path.insert(0, ROOT)
    from lmfit_global import LmfitGlobal

import matplotlib.pyplot as plt
# sys.path


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


# --- create raw data with above function
ndata = 5        # number of datasets
numpoints = 151  # number of points per dataset

xy = np.zeros([numpoints, ndata+1], dtype=float) # xy will hold x plus ndata columns of y

np.random.seed(2021) # reproducibility

x = np.linspace(-1, 2, numpoints) # x grid
xy[:, 0] = x

plt.figure()  # plot raw data
for i in range(ndata):
    amplitude = +0.60 + 9.50*np.random.rand()
    center    = -0.20 + 1.20*np.random.rand()
    sigma     = +0.25 + 0.03*np.random.rand()

    dat = gaussian(x, amplitude, center, sigma) \
          + np.random.normal(scale=0.1, size=x.size)    # add noise

    xy[:, i+1] = dat

    plt.plot(x, dat, 'o')
plt.show()


# finally, built `data` and `function` **`items`** `dict` for `LmfitGlobal` ...

# In[3]:


# data dict
data_dict = {
    'xy': xy,         # data_xy, i.e numpy.column_stack([x, y_0, y_1, ..., y_n])
    'xrange': None    # x range in (min, max) of the data range to fit, default is None
    }

# --- NOTE ---
# (1) init_params items must match "gaussian" function arguments defined above
# (2) init_params argument you can set, 'value', 'vary':True/False, bounds, 'min'/'max' below
# (3) you can set mimimal either, 'value', or 'vary' or 'min'/'max', else defualt parameters will be used
# (4) defualt parameters are: 'value':-inf, 'vary':True, 'min':-inf & 'max':+inf
func_lst = [
    {
        'func_name': gaussian,
        'init_params' : {
            'amplitude': {'value':1.0, 'vary':True, 'min':0.0, 'max':200},
            'center': {'value':0.4, 'vary':True, 'min':-2.0, 'max':2.0},
            'sigma': {'value':0.3, 'vary':True, 'min':0.01, 'max':3.0},
        },
        'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
    },
]

# function dict
function_dict = {
    'theory': func_lst,
    'theory_connectors': None,
}

# items 
items = {
    'data': data_dict,              # 1. data (see above)
    'functions': function_dict,     # 2. theory (see above)
}


# call `LmfitGlobal` class ...

# In[4]:


lg = LmfitGlobal(items, log_level='info')


# (Optional) print & plot initial parameter definitions...

# In[ ]:


lg.init_params.pretty_print()  # pretty print initial parameters 
# The parameternames_{index} is generic how the cord works to indicate data index
# for multidatasets parameternames_0, parametername_1, ..., will be displayed

# --- fancy plots --- (USE IN NEXT EXAMPLES)
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_init(numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot init parameters
plt.show()


# Now, do fit...

# In[ ]:


lg.fit(verbose=True)  # verbose, if True will show fit parameters


# or better use: `.report()` ...

# In[ ]:


lg.report()


# plot fit...(Fancy LmfitGlobal matplotlib)

# In[ ]:


# --- fancy plots ---
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_fit(plot_residual=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot fit parameters
# --- OR ---
# lg.plot(plot_residual=True, show=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)
plt.show()


# From fit parameters shown above, its clear that `sigma_0, sigma_1, ..., sigma_4` 
# 
# can be set as global parameters (nearly constant width `sigma`$\sim$ 0.3)
# 
# this can be achieved with
# 
# ```python
# LmfitGlobal.set_global(
#         self, 
#         *parlist: Iterable, 
#         reference: Optional[str] = None, 
#         overwrite_expr: bool = False            
#     )
# ```
# 
# or with:
# ```python
# LmfitGlobal.set_global_params(
#         self, 
#         *parlist: Iterable, 
#         reference: Optional[str] = None, 
#         overwrite_expr: bool = False            
#     )
# ```
# 
# where  `parlist` is of type: 
# ```python
# *parlist: Union[str, list[str], lmfit.Parameter, lmfit.Parameters, Iterable, Dict]
# ```

# In[ ]:


lg.rebuild()  # rebuild LmfitGlobal

parlist = [f'sigma_{i}' for i in range(5)]  # list of str
lg.set_global(parlist, reference='sigma_0', overwrite_expr=True) # if reference=None default reference=parlist[0]


# (re-fit): Now, do fit...

# In[ ]:


lg.fit(verbose=True)  # verbose, if True will show fit parameters


# (re-fit): or better use: `.report()` ...

# In[ ]:


lg.report()


# (re-fit): plot fit...(Fancy LmfitGlobal matplotlib)

# In[ ]:


# --- fancy plots ---
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_fit(plot_residual=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot fit parameters
# --- OR ---
# lg.plot(plot_residual=True, show=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)
plt.show()


# get FitData object: (for more user friendly data handling & plotting...)

# In[ ]:


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
        # label=f"data {i+1}",
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
        label="fit" if i == fd.ny-1 else None, # ONE legend entry only
        # label=_fit_label(i, fd.is_multidataset, same_fit_color=True)
    )

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

# see next examples on how we can handle multi-component fit
# and using util.utils.GlobalFitBuilder() class to create LmfitGlobal items

