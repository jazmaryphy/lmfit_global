# LmfitGlobal

`LmfitGlobal` is the core class of **lmfit-global**.
It provides a unified interface for **global (single-run) curve fitting**
across multiple datasets and multiple model components using
[`lmfit`](https://lmfit.github.io/lmfit-py/).

It manages:

- model construction
- parameter bookkeeping
- residual evaluation
- fitting and diagnostics
- uncertainty propagation

---

## When should you use `LmfitGlobal`?

Use `LmfitGlobal` if:

- You need to fit **multiple datasets simultaneously**
- Your model has **shared or constrained parameters**
- Your fit consists of **multiple functional components**
- You want **consistent uncertainty estimates** across datasets

---

## Basic usage

```python
from lmfit_global import LmfitGlobal

lg = LmfitGlobal(
    items=items,
    log_level='info'
)

lg.init_params.pretty_print()         # print initial parameters
lg.pretty_expr  # or lg.theory_expr   # to print model expressions

lg.set_data(x, ylist)      # set data if not define in items above

lg.set_nan_policy("omit")  # define NaN policy, "raise", "omit" or "propagate"

lg.plot_init()         # plot init fit data
# or fancy plot with
pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
lg.plot_init(numpoints=600, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # numpoints=None, default, plot init parameters
plt.show()

parlist = {"alpha": {"value": 2.5,  "min":0, "max": 5, "vary": True}}
lg.add_par(parlist)      # add new non existing parameters from fit functions
# or with 
lg.add_params(parlist)   # add new non existing parameters from fit functions

# --- set "inequality" contrain below
mapping   = {
                "c1_center_0": "alpha + c0_center_0", 
                "c1_sigma_0": "c0_sigma_0"
            }
lg.set_expr(mapping, overwrite_expr=True)

lg.init_params.pretty_print()  # pretty print initial parameters 

parlist = [f'sigma_{i}' for i in range(5)]                              # list of str
lg.set_global(parlist, reference='sigma_0', overwrite_expr=True)        # if reference=None default reference=parlist[0]
# or with 
lg.set_global_params(parlist, reference='sigma_0', overwrite_expr=True) # if reference=None default reference=parlist[0]

lg.rebuild()          # rebuild LmfitGlobal to default

lg.fit(verbose=True)  # fit
result = lg.result    # lmfit.MinimizerResult

lg.report()           # report fit optimized parameters values, uncertainties & correlations, fit statistics, etc

lg.plot()             # plot fit data, including residuals
# UMCOMMENT BELOW FOR MORE FANCY PLOT
# pretty_kw={'width': 6, 'height':6, 'dpi':100} # width and height and dpi of figure, or None to use default settings
# lg.plot(numpoints=600, xlabel='x', ylabel='y', pretty_kw=pretty_kw)  # plot init parameters
plt.show()

fd = lg.get_fitdata(numpoints=1024)   # structured fit results for plotting and analysis

dely = lg.eval_uncertainty(sigma=3)   # confidence interval
dely_pred = lg.dely_predicted         # prediction interval
dely_comps = lg.dely_comps            # per-component uncertainty (if multicomponent)

# AND MANY MORE...
```
