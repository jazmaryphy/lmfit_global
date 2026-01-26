# %% [markdown]
# # Model - multi-components & uncertainty
# 
# This example shows how to use `lmfit_global.LmfitGlobal` class to fit data with multi-components and evaluate uncertainty. 
# 
# This example is similar to `model_uncertainty2.py` of [lmfit examples](https://lmfit.github.io/lmfit-py/examples/index.html) or [github link](https://github.com/lmfit/lmfit-py/tree/master/examples)
# 

# %%
try:
    from lmfit_global import LmfitGlobal
    from lmfit_global.utils import lineshapes
    from lmfit_global.utils.builders import GlobalFitBuilder
except (ImportError, ModuleNotFoundError):
    import os, sys
    ROOT = os.path.abspath("..")  # parent of examples
    sys.path.insert(0, ROOT)

    from lmfit_global import LmfitGlobal
    from lmfit_global.utils import lineshapes
    from lmfit_global.utils.builders import GlobalFitBuilder

import matplotlib
import matplotlib.pyplot as plt

# %%
import os
import numpy as np
dpath = './data'  # data path

# --- Load data (skip header) ---
file = 'NIST_Gauss2.dat' # data
file = os.path.join(
    dpath,
    file
)

data = np.loadtxt(file)
x = data[:, 1]  # second  columm as x
y = data[:, 0]  # first   column as y

xy = np.column_stack([x, y])

plt.plot(x, y)
plt.show()

# %%
init_expone = {
        'amplitude': {'value':100, 'vary':True, 'min':-np.inf, 'max':+np.inf},
        'decay': {'value':80, },
}

init_gauss1 = {
    'amplitude': {'value':3000, 'vary':True, 'min':-np.inf, 'max':+np.inf},
    'center': {'value':100, },
    'sigma': {'value':10, },
}

init_gauss2 = {
    'amplitude': {'value':3000, 'vary':True, 'min':-np.inf, 'max':+np.inf},
    'center': {'value':150, },
    'sigma': {'value':10, },
}


# %%
builder = (
    GlobalFitBuilder()
    .set_data(x, y)                      # x and all y datasets
    .add_model(lineshapes.exponential,   init_expone, func_kws={})
    .add_model(lineshapes.gaussian,      init_gauss1, func_kws={})      
    .add_model(lineshapes.gaussian,      init_gauss2, func_kws={})     
    .connect("+", "+")                             # how to combine the 2 functions
)

items = builder.build()

# %%
lg = LmfitGlobal(items, log_level='info')
lg.fit(iter_cb=None)

# %%
lg.report()
lg.plot()
plt.show()

# %%
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

# %%
fd = lg.get_fitdata(numpoints=1024) # int or None
# fd.x_data
# fd.resid_fit

# %%
fig, ax = plt.subplots()


for i in range(fd.ny):
    # --- data ---
    ax.plot(
        fd.x_data,
        fd.y_data[:, i],
        'o',
        color='#99002299',
        zorder=1,
        label=_data_label(i, fd.is_multidataset),
    )

    # --- fit ---
    ax.plot(
        fd.x_model,
        fd.y_fit[:, i],
        '-',
        lw=2,
        color='b',
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
            ax.plot(fd.x_data,  d_dict['data'],  '--', zorder=3, lw=1.5, label=name)
            # ax.plot(fd.x_model, d_dict['model'], '--', label=name)

# Styling
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

# %%
fig, ax = plt.subplots()

dely = lg.eval_uncertainty(
    x=fd.x_model,
    sigma=3
)

for i in range(fd.ny):

    # --- data ---
    ax.plot(
        fd.x_data,
        fd.y_data[:, i],
        'o',
        color='#99002299',
        zorder=2,
        label=_data_label(i, fd.is_multidataset),
    )

    # --- fit ---
    ax.plot(
        fd.x_model,
        fd.y_fit[:, i],
        '-',
        lw=1,
        color='b',
        zorder=3,
        label=_fit_label(i, fd.is_multidataset, same_fit_color=True),
    )

    # --- uncertainty band ---
    ax.fill_between(
        fd.x_model,
        fd.y_fit[:, i] - dely[:, i],
        fd.y_fit[:, i] + dely[:, i],
        color="#8A8A8A",
        label=r'3-$\sigma$ band' if i == 0 else None,
    )

# Styling
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
        "size": fontsize-2,
        "family": "sans-serif",
        "weight": "normal",
    },
    frameon=False,
)
plt.show()


# %%
COLORS = {
    "data": "#1F77B4",        # muted blue
    "fit": "#000000",         # black
    "conf": "#9E9E9E",        # medium gray
    "pred": "#D0D0D0",        # light gray
    "components": [
        "#D55E00",            # vermillion
        "#009E73",            # green
        "#CC79A7",            # purple
        "#E69F00",            # orange
    ],
}

PLOT_COMPONENTS = True
PLOT_TOTAL_UNCERTAINTY = True
PLOT_COMPONENT_UNCERTAINTY = True
PLOT_PREDICTION_UNCERTAINTY = True
SIGMA = 3

NEED_UNCERTAINTY = (
    PLOT_TOTAL_UNCERTAINTY
    or PLOT_COMPONENT_UNCERTAINTY
    or PLOT_PREDICTION_UNCERTAINTY
)

dely = None
dely_pred = None
dely_comps = None

if NEED_UNCERTAINTY:
    lg.eval_uncertainty(x=fd.x_model, sigma=SIGMA)
    dely = lg.dely
    dely_pred = getattr(lg, "dely_predicted", None)
    dely_comps = getattr(lg, "dely_comps", None)


dely_comps = (
    lg.dely_comps
    if (PLOT_COMPONENT_UNCERTAINTY and fd.is_multicomponent and hasattr(lg, "dely_comps"))
    else {}
)

# %%
fig, ax = plt.subplots(figsize=(7.5, 5.5))

for i in range(fd.ny):

    # Data
    ax.plot(
        fd.x_data,
        fd.y_data[:, i],
        "o",
        ms=5,
        color=COLORS["data"],
        zorder=4,
        label=_data_label(i, fd.is_multidataset),
    )

    # Total fit
    ax.plot(
        fd.x_model,
        fd.y_fit[:, i],
        "-",
        lw=2.2,
        color=COLORS["fit"],
        zorder=5,
        label=_fit_label(i, fd.is_multidataset, same_fit_color=True),
    )

    # Prediction interval (widest, behind)
    if PLOT_PREDICTION_UNCERTAINTY and dely_pred is not None:
        label = rf"{SIGMA}$\sigma$ prediction" if i == 0 else None
        ax.fill_between(
            fd.x_model,
            fd.y_fit[:, i] - dely_pred[:, i],
            fd.y_fit[:, i] + dely_pred[:, i],
            color=COLORS["pred"],
            alpha=0.45,
            zorder=1,
            label=label,
        )

    # Confidence interval (narrower, on top)
    if PLOT_TOTAL_UNCERTAINTY and dely is not None:
        label = rf"{SIGMA}$\sigma$ confidence" if i == 0 else None
        ax.fill_between(
            fd.x_model,
            fd.y_fit[:, i] - dely[:, i],
            fd.y_fit[:, i] + dely[:, i],
            color=COLORS["conf"],
            alpha=0.6,
            zorder=2,
            label=label,
        )

    # Components
    if PLOT_COMPONENTS and fd.is_multicomponent:

        comps = fd.components[i] if fd.is_multidataset else fd.components
        comp_names = fd.component_names

        for j, cname in enumerate(comp_names):
            ccol = COLORS["components"][j % len(COLORS["components"])]

            ax.plot(
                fd.x_model,
                comps[cname]["model"],
                "--",
                lw=1.8,
                color=ccol,
                zorder=3,
                label=cname if i == 0 else None,
            )

            # Component uncertainty
            if (
                PLOT_COMPONENT_UNCERTAINTY
                and dely_comps is not None
                and cname in dely_comps
            ):
                ax.fill_between(
                    fd.x_model,
                    comps[cname]["model"] - dely_comps[cname][:, i],
                    comps[cname]["model"] + dely_comps[cname][:, i],
                    color=ccol,
                    alpha=0.25,
                    zorder=2,
                )

# Styling
fontsize = 14

ax.set_xlabel("x", fontsize=fontsize + 2)
ax.set_ylabel("y", fontsize=fontsize + 2)

ax.minorticks_on()
ax.tick_params(
    direction="in",
    which="both",
    top=True,
    right=True,
    labelsize=fontsize,
)
ax.tick_params(which="major", length=8, width=1.2)
ax.tick_params(which="minor", length=4, width=1.0)

for spine in ax.spines.values():
    spine.set_linewidth(1.4)

# prop={
#     "size": fontsize-2,
#     "family": "sans-serif",
#     "weight": "normal",
# }
ax.legend(
    frameon=False,
    # prop=prop,
    fontsize=fontsize,
    handlelength=2.5,
    ncol=1,
)

plt.tight_layout()
plt.show()