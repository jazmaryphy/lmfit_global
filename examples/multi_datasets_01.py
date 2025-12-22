# %% [markdown]
# # Gaussian Multi-datasets
# 
# This example shows how to use `lmfit_global.py` to fit multi-dataset with gaussian function. 
# 
# This example is similar to `example_fit_multi_datasets.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# %%
try:
    from lmfit_global import LmfitGlobal
except (ImportError, ModuleNotFoundError):
    import sys
    sys.path.append('./lmfit_global')
    from lmfit_global import LmfitGlobal
    
import matplotlib.pyplot as plt

# %% [markdown]
# # Create raw data
# 
# Create five simulated Gaussian data sets

# %%
import os
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

    Parameters
    ----------
    value : scalar
        Value to be ensured not to be zero.

    Returns
    -------
    scalar
        Value ensured not to be zero.

    """
    return float(np.copysign(max(tiny, abs(value)), value))

def gaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """Return a 1-dimensional Gaussian function.

    gaussian(x, amplitude, center, sigma) =
        (amplitude/(s2pi*sigma)) * exp(-(1.0*x-center)**2 / (2*sigma**2))

    """
    return ((amplitude/(max(tiny, s2pi*sigma)))
            * np.exp(-(1.0*x-center)**2 / max(tiny, (2*sigma**2))))


# --- create raw data
# number of points per dataset
numpoints = 151
# number of datasets
ndata = 5

# xy will hold x plus ndata columns of y
xy = np.zeros((numpoints, ndata+1))

# reproducibility
np.random.seed(2021)

# x grid
x = np.linspace(-1, 2, numpoints)
xy[:, 0] = x

# plot raw data
plt.figure()
for i in range(ndata):
    amplitude = 0.60 + 9.50*np.random.rand()
    center = -0.20 + 1.20*np.random.rand()
    sigma = 0.25 + 0.03*np.random.rand()

    dat = gaussian(x, amplitude, center, sigma) \
          + np.random.normal(scale=0.1, size=x.size)

    xy[:, i+1] = dat

    plt.plot(x, dat, 'o')


# %% [markdown]
# <!-- # Define the model to fit the `n` multi-dataset
# 
# #### 1. Data Format
# The `lmfit_global.py` `LmfitGlobal` module accepts raw input data as a NumPy array `numpy.ndarray` with shape: (nsample, ndata)
# 
# This means:
# - The first column is the shared x-axis: `x`
# - Each subsequent column is a separate dataset: `y_0`, `y_1`, ..., `y_n`
# 
# ##### Example
# ```python
# import numpy as np
# 
# # x values
# x = np.linspace(0, 10, 100)
# 
# # multiple y datasets
# y0 = np.exp(-(x - 3)**2)
# y1 = np.exp(-(x - 6)**2)
# 
# # stack into shape [100, 3]
# data_xy = np.column_stack([x, y0, y1])
# # -- similar how we define the simulated data called `xy` above 
# ```
# 
# Therefore, 
# - `lmfit_global.py` read the data in `dict` format
# - Let define it the data block as:
# 
# ```python
# data_block = {
#     'xy': xy,         # data_xy, i.e numpy.column_stack([x, y_0, y_1, ..., y_n])
#     'xrange': None    # x range in (min, max) of the data range to fit, default is None
#     }
# ```
# 
# 
# #### 2. Theory: Gaussian
# This model assumes each dataset follows a Gaussian profile named `gaussian` define above:
# 
# The model has three parameters: *amplitude*, *center*, and *sigma*.  
# In addition, parameters `fwhm` and `height` are included as constraints to report the full width at half maximum and the maximum peak height.
# 
# $$
# f(x; A, \mu, \sigma)
# = \frac{A}{\sigma \sqrt{2\pi}} \exp\!\left[-\frac{(x - \mu)^2}{2\sigma^2}\right]
# $$
# 
# Here, the parameter *amplitude* corresponds to $A$,  *center* corresponds to $\mu$, and   *sigma* corresponds to $\sigma$.   
# The full width at half maximum (FWHM) is $\text{FWHM}= 2 \sigma \sqrt{2 \ln 2}\approx 2.3548\, \sigma .$
# 
# The model function are define as `list of dict` define below:
# 
# ```python
# theory_func_lst = [
#     {
#         'func_name': gaussian,
#         'init_params' : {
#             'amplitude': {'value':0.5, 'vary':True, 'min':0.0, 'max':200},
#             'center': {'value':0.4, 'vary':True, 'min':-2.0, 'max':2.0},
#             'sigma': {'value':0.3, 'vary':True, 'min':0.01, 'max':3.0},
#         },
#         'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
#     },
#     {
#         SAME AS ABOVE IF MULTIPLE FUNCTION/COMPONENT REQUIRE
#     },
# ]
# ``` 
# where,
# - 'func_name': is callable function named `gaussian`
# - 'init_params': is the initial parameter attributes of the function parameters named *amplitude*, *center*, and *sigma*.  
#   - 'value': `float`, initial value
#   - 'vary': `bool`, `True`/`False`. If True to minimize the parameter else fixed to initial value
#   - 'min' & 'max': `float`, are the lower or upper bounds
# - 'func_kws': keyword arguments to pass to the function (IF ANY)
# 
# NOTE: Repeat above if you are dealing with multiple function  
# see example `multi_components_01.ipynb`, how to deal with this part:
# 
# 
# Now, 
# - `lmfit_global.py` read the `theory` in `dict` format define below:
# 
# ```python
# function_block = {
#     'theory': theory_func_lst,
#     'theory_connectors': [],       # see example `multi_components_01.ipynb` how to use the parameter
# }
# ```
# 
# #### 3. Setting: *class* `LmfitGlobal`
# 
# 
# ```python
# class LmfitGlobal(items, independent_vars=None, nan_policy='raise', method='leastsq', **fit_kws)
# ```
# 
# **Parameters**
# 
# **items** (dict): A dictionary containing the raw data and the model function
# 
# **independent_vars** (list of str, optional):
# Arguments to the model function that are independent variables. Default is ['x'].
# 
# **nan_policy** ({'raise', 'propagate', 'omit'}, optional):
# How to handle NaN or missing values in the data.
# See Notes below.
# 
# **method** (str, optional):
# Fitting method available in lmfit (default is 'leastsq')
# 
# **fit_kws** (dict, optional)
# Keyword arguments to pass to the fitting/minimizer.
# 
# 
# Notes
# 
# *nan_policy* determines what to do when a NaN or missing value is seen in the data.
# It must be one of:
# 
#  - 'raise' : raise a ValueError (default)
#  - 'propagate' : silently propagate NaNs
#  - 'omit' : drop missing data
# ---
# 
# 
# Finally, **`items`**   is define as 
# ```python
# items = {
#     'data': data_block,              # 1. data (see above)
#     'functions': function_block,     # 2. thoery (see above)
# }
# ``` -->
# 

# %% [markdown]
# # Define the model to fit the `n` multi-dataset

# %% [markdown]
# #### 1. Data Format
# The `lmfit_global.py` `LmfitGlobal` module accepts raw input data as a NumPy array `numpy.ndarray` with shape: (nsample, ndata)
# 
# This means:
# - The first column is the shared x-axis: `x`
# - Each subsequent column is a separate dataset: `y_0`, `y_1`, ..., `y_n`
# 
# ##### Example
# ```python
# import numpy as np
# 
# # x values
# x = np.linspace(0, 10, 100)
# 
# # multiple y datasets
# y0 = np.exp(-(x - 3)**2)
# y1 = np.exp(-(x - 6)**2)
# 
# # stack into shape [100, 3]
# data_xy = np.column_stack([x, y0, y1])
# # -- similar how we define the simulated data called `xy` above 
# ```
# 
# Therefore, 
# - `lmfit_global.py` read the data in `dict` format
# - Let define it the data block as:
# 
# ```python
# data_block = {
#     'xy': xy,         # data_xy, i.e numpy.column_stack([x, y_0, y_1, ..., y_n])
#     'xrange': None    # x range in (min, max) of the data range to fit, default is None
#     }
# ```

# %%
data_block = {
    'xy': xy,         # data_xy, i.e numpy.column_stack([x, y_0, y_1, ..., y_n])
    'xrange': None    # x range in (min, max) of the data range to fit, default is None
    }

# %% [markdown]
# #### 2. Theory: Gaussian
# This model assumes each dataset follows a Gaussian profile named `gaussian` define above:
# 
# The model has three parameters: *amplitude*, *center*, and *sigma*.  
# 
# $$
# f(x; A, \mu, \sigma)
# = \frac{A}{\sigma \sqrt{2\pi}} \exp\!\left[-\frac{(x - \mu)^2}{2\sigma^2}\right]
# $$
# 
# Here, the parameter *amplitude* corresponds to $A$,  *center* corresponds to $\mu$, and   *sigma* corresponds to $\sigma$.   
# The full width at half maximum (FWHM) is $\text{FWHM}= 2 \sigma \sqrt{2 \ln 2}\approx 2.3548\, \sigma .$
# 
# The model function are define as `list of dict` define below:
# 
# ```python
# theory_func_lst = [
#     {
#         'func_name': gaussian,
#         'init_params' : {
#             'amplitude': {'value':0.5, 'vary':True, 'min':0.0, 'max':200},
#             'center': {'value':0.4, 'vary':True, 'min':-2.0, 'max':2.0},
#             'sigma': {'value':0.3, 'vary':True, 'min':0.01, 'max':3.0},
#         },
#         'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
#     },
#     # {SAME AS ABOVE IF MULTIPLE FUNCTION/COMPONENT REQUIRE},
# ]
# ``` 
# where,
# - 'func_name': is callable function named `gaussian`
# - 'init_params': is the initial parameter attributes of the function parameters named *amplitude*, *center*, and *sigma*.  
#   - 'value': `float`, initial value
#   - 'vary': `bool`, `True`/`False`. If True to minimize the parameter else fixed to initial value
#   - 'min' & 'max': `float`, are the lower or upper bounds
# - 'func_kws': keyword arguments to pass to the function (IF ANY)
# 
# NOTE: 
# - Repeat above if you are dealing with multiple function  
# - see example `multi_components_01.ipynb`, how to deal with this part:
# - the `init_params` and attributes is set for all the `n` datasets
# 
# 
# Now, 
# - `lmfit_global.py` read the `theory` in `dict` format define below:
# 
# ```python
# function_block = {
#     'theory': theory_func_lst,
#     'theory_connectors': None,       # Default None, see example `multi_components_01.ipynb` how to use the parameter
# }
# ```

# %%
theory_func_lst = [
    {
        'func_name': gaussian,
        'init_params' : {
            'amplitude': {'value':0.5, 'vary':True, 'min':0.0, 'max':200},
            'center': {'value':0.4, 'vary':True, 'min':-2.0, 'max':2.0},
            'sigma': {'value':0.3, 'vary':True, 'min':0.01, 'max':3.0},
        },
        'func_kws': {}   # <-- Additional keyword arguments to pass to model function `'func_name'`.
    },
    # {SAME AS ABOVE IF MULTIPLE FUNCTION/COMPONENT REQUIRE },
]

function_block = {
    'theory': theory_func_lst,
    'theory_connectors': None,       # Default None, see example `multi_components_01.ipynb` how to use the parameter
}

# %% [markdown]
# #### 3. Setting: *class* `LmfitGlobal`
# 
# 
# ```python
# class LmfitGlobal(items, independent_vars=None, nan_policy='raise', method='leastsq', **fit_kws)
# ```
# 
# **Parameters**
# 
# **items** (dict): A dictionary containing the raw data and the model function
# 
# **independent_vars** (list of str, optional):
# Arguments to the model function that are independent variables. Default is ['x'].
# 
# **nan_policy** ({'raise', 'propagate', 'omit'}, optional):
# How to handle NaN or missing values in the data.
# See Notes below.
# 
# **method** (str, optional):
# Fitting method available in lmfit (default is 'leastsq')
# 
# **fit_kws** (dict, optional)
# Keyword arguments to pass to the fitting/minimizer.
# 
# 
# Notes
# 
# *nan_policy* determines what to do when a NaN or missing value is seen in the data.
# It must be one of:
# 
#  - 'raise' : raise a ValueError (default)
#  - 'propagate' : silently propagate NaNs
#  - 'omit' : drop missing data
# ---
# 
# 
# Finally, **`items`**   is define as 
# ```python
# items = {
#     'data': data_block,              # 1. data (see above)
#     'functions': function_block,     # 2. thoery (see above)
# }
# ```

# %%
items = {
    'data': data_block,              # 1. data (see above)
    'functions': function_block,     # 2. thoery (see above)
}

# %%


# %% [markdown]
# ### call `LmfitGlobal`

# %%
LFG = LmfitGlobal(items)

# %%
LFG.initial_params.pretty_print()  # pretty print initial parameters

LFG.plot_init()  # plot init parameters

# """NOTE:  Index at the end of each parameters "_0, _1, ..., _n" are index of the dataset
# """

# %% [markdown]
# #### Fitting....
# 
# to fit call `.fit()` function as:
# 
# `LFG.fit()`

# %%
LFG.fit(verbose=True)  # verbose, if True will show fit parameters

# """Note:
# - The width 'sigma_*' of the of each dataset are almost constant ~0.26
# We can set them as global/share and fit

# -  go back above and set global parameters
# """ 

# %%
axes = LFG.plot(show_init=False)   # plot fit data & residuals and show initial plot (if True)

# %%
LFG.report()  # report fit parameters

# %%
LFG.set_global(['sigma_0'])   #set global parametes

LFG.initial_params.pretty_print()  # pretty print initial parameters

# %%
# repeat

LFG.fit(verbose=True)  # verbose, if True will show fit parameters

# """all sigma share same values
# """

axes = LFG.plot(show_init=False)   # plot fit data and show initial plot 

# %%
LFG.report()  # report fit parameters

# %%



