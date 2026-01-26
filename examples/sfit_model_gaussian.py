#!/usr/bin/env python
# coding: utf-8

# # Model - Gaussian
# 
# This example shows how to use `lmfit_global.simplefit` code to fit data with simple gaussian. 
# 
# This example is similar to `model_gaussian.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# In[1]:


try:
    from lmfit_global import simplefit
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..") # parent folder of examples
    sys.path.insert(0, ROOT)
    from lmfit_global import simplefit

import matplotlib.pyplot as plt
# sys.path


# In[2]:


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


# In[3]:


from lmfit_global.utils import lineshapes

p0 = [5.0, 5.0, 1.0]   # init params
### --- or in form of --- ###
# p0 = {
#     'amplitude': {'value':5.0, 'vary': True},
#     'center':    {'value':5.0, 'vary': True},
#     'sigma':     {'value':1.0, 'vary': True},
# }                     # init params

result = None
return_result = True
if return_result:
    popt, perr, result = simplefit(
        lineshapes.gaussian,
        p0=p0,
        x=x,
        y=y,
        err=1.0,
        fit_method="leastsq",
        return_result=return_result
    )
else:
    popt, perr = simplefit(
        lineshapes.gaussian,
        p0=p0,
        x=x,
        y=y,
        err=1.0,
        fit_method="leastsq",
        return_result=False
    )

print("\nFit status smplefit:")
print("====================")
print("Best-fit parameters:        ", popt)
print("Asymptotic error:           ", perr)
print()

if result:
    import lmfit
    print(lmfit.fit_report(result))

yfit  = lineshapes.gaussian(x, *popt)
plt.plot(x, y, 'o')
plt.plot(x, yfit, '-', label='best fit')
plt.legend()
plt.show()

