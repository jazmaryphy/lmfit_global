# %% [markdown]
# # Model - Gaussian
# 
# This example shows how to use `lmfit_global.LmfitGlobal` class to fit data with simple gaussian. 
# 
# This example is similar to `model_gaussian.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# %%
try:
    from lmfit_global import LmfitGlobal
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..") # parent folder of examples
    sys.path.insert(0, ROOT)
    from lmfit_global import LmfitGlobal

import matplotlib.pyplot as plt
# sys.path

# %% [markdown]
# # Define `LmfitGlobal` class `items` data
# 
# First, load raw data...

# %%
import os
import numpy as np
dpath = './data'  # data path

# --- Load data (skip header) ---
file = 'model1d_gauss.dat' # data
file = os.path.join(
    dpath,
    file
)

data = np.loadtxt(file)
x = data[:, 0]  # first  columm as x
y = data[:, 1]  # second column as y

#  --- make column data as [x, y]  ---
#  --- lets use numpy.column_stack ---
xy = np.column_stack([x, y])

# plot raw data
plt.figure()
plt.plot(x, y, 'o')
plt.show()

# %% [markdown]
# second, model function to fit data...

# %%
# import numpy as np

def gaussian(x, amp, cen, wid):
    """1-d gaussian: gaussian(x, amp, cen, wid)"""
    return (amp / (np.sqrt(2*np.pi) * wid)) * np.exp(-(x-cen)**2 / (2*wid**2))

# %% [markdown]
# finally, built `data` and `function` **`items`** `dict` for `LmfitGlobal` ...

# %%
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
            'amp': {'value':5, },
            'cen': {'value':5, },
            'wid': {'value':1, },
        },
        'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
    },
]

# function dict
function_dict = {
    'theory': func_lst,
    'theory_connectors': None,
}


# """
# # --- The `theory_connectors` (list of str): 
#     A list of binary operators (e.g., '+', '-', '*', '/') that define how to combine 
#     the theory functions listed in 'theory'. Each operator connects one function to the next 
#     in left-to-right order. 
    
#     For example: 
#     - ['+', '+'] means: theory[0] + theory[1] + theory[2].

#     - ['+', '*'] means: theory[0] + theory[1] * theory[2].

#     The number of connectors must be exactly one less than the number of theory functions.
#     The ONLY (so-far) supported operators are: '+', '-', '*', '/'.

# NOTE: Here in this  case is None or []
# """


# items 
items = {
    'data': data_dict,              # 1. data (see above)
    'functions': function_dict,     # 2. theory (see above)
}

# %% [markdown]
# call `LmfitGlobal` class ...

# %%
lg = LmfitGlobal(items, log_level='info')
lg.theory_expr # or lg.pretty_expr
# --- The box below show how the model y(x) (CompositeModel) is define using `theory_connectors` define above ---
# --- it is upto the USER to define how to define the CompositeModel ---
# --- if you are HAPPY proceed NEXT

# %% [markdown]
# (Optional) print & plot initial parameter definitions...

# %%
lg.init_params.pretty_print()  # pretty print initial parameters 
# The parameternames_{index} is generic how the cord works to indicate data index
# for multidatasets parameternames_0, parametername_1, ..., will be displayed

# --- fancy plots --- (USE IN NEXT EXAMPLES)
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_init(numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot init parameters
plt.show()

# %% [markdown]
# Now, do fit...

# %%
lg.fit(verbose=True)  # verbose, if True will show fit parameters

# %% [markdown]
# or better use: `.report()` ...

# %%
lg.report()

# %% [markdown]
# plot fit...(matplotlib)

# %%
plt.plot(xy[:, 0], xy[:, 1], 'o')
plt.plot(xy[:, 0], lg.init_fit, '--', label='initial fit')
plt.plot(xy[:, 0], lg.best_fit, '-', label='best fit')
plt.legend()
plt.show()

# %% [markdown]
# plot fit...(Fancy LmfitGlobal matplotlib)

# %%
# --- fancy plots ---
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_fit(plot_residual=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot fit parameters
# --- OR ---
# lg.plot(plot_residual=True, show=True, numpoints=None, xlabel='x', ylabel='y', pretty_kw=pretty_kw)
plt.show()

# %% [markdown]
# plot fit...(User plotting choice)
# 
# first, extract the fit data using:  `.get_fitdata(numpoints=None)`
# 
# where numpoints (integer), to get more dense point
# 
# ```python
# fd = lg.get_fitdata(numpoints=None) # int or None
# ```
# 
# returns a FitData object that provides structured access to the results.
# 
# The returned FitData object contains:
# 
# Raw data:
# ```python
# fd.x_data    # ndarray (N,)
# fd.y_data    # ndarray (N, ny)
# ```
# 
# Initial model (before fitting):
# ```python
# fd.x_model   # ndarray (N or numpoints)
# fd.y_init    # ndarray (N or numpoints, ny)
# fd.resid_init  # y_data - initial_model (on data grid)
# ```
# 
# Best-fit model (after fitting):
# ```python
# fd.y_fit       # ndarray (N or numpoints, ny) or None if not fitted
# fd.resid_fit   # ndarray (N, ny) or None if not fitted
# ```
# 
# Multi-component models (if applicable):
# ```python
# fd.components  # dict or None
# ```

# %% [markdown]
# NOW, A USER CAN USE ANY CHOICE OF PLOTTING PROGRAM/CODE
# 
# lets do matplotlib again...

# %%
fd = lg.get_fitdata(numpoints=1024) # int or None
fd.x_data


for i in range(lg.ny):
    plt.plot(fd.x_data,  fd.y_data[:, i], 'o', zorder=1)
    plt.plot(fd.x_model, fd.y_fit[:, i],  '-', zorder=2)

plt.show()

# where ny is number of datasets, luckily here we dealing with single-dataset
# see next examples on how we can handle multi-datasets