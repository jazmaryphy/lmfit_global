# %%
import inspect
import numpy as np

# %%
# --- The package lmfit is a MUST
try:
    import lmfit
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("lmfit is required. Install with `pip install lmfit`") from exc

# %%
def simplefit(
    model,
    p0,
    x,
    y,
    err=1.0,
    backend="model",
    fit_method="leastsq",
    return_result=False,
    **fit_kws,
):
    """
    Fit a model function to data using lmfit.

    This function provides a unified interface to both ``lmfit.Model`` and
    ``lmfit.Minimizer`` backends. The model function must follow the
    conventional signature ``f(x, p1, p2, ...)``.

    Args:
        model (callable):
            Model function with signature ``f(x, p1, p2, ...) -> array_like``.
        p0 (list or dict):
            Initial parameter values. If a list, it must match the order of
            parameters in the model signature. If a dict, keys must match
            parameter names. Dictionary values may be scalars or keyword
            dictionaries accepted by ``lmfit.Parameters.add``.
        x (array_like):
            Independent variable values.
        y (array_like):
            Dependent variable values.
        err (float or array_like, optional):
            Measurement errors. If scalar, a constant error is assumed for
            all data points. Defaults to 1.0.
        backend (str, optional):
            Fitting backend to use. Must be either ``"model"`` or
            ``"minimizer"``. Defaults to ``"model"``.
        fit_method (str, optional):
            Minimization method passed to lmfit (e.g. ``"leastsq"``,
            ``"nelder"``, ``"powell"``). Defaults to ``"leastsq"``.
        return_result (bool, optional):
            If True, include the lmfit result object in the return values.
        **fit_kws:
            Additional keyword arguments passed to
            ``lmfit.Model.fit`` or ``lmfit.Minimizer.minimize``.

    Returns:
        lmfit result or tuple:
            - lmfit result (default)
            - or ``(popt, perr)``
            - or ``(popt, perr, result)``
    """

    x = np.asarray(x)
    y = np.asarray(y)

    if np.isscalar(err):
        err = np.full_like(y, err, dtype=float)
    else:
        err = np.asarray(err, dtype=float)

    if np.any(err <= 0):
        raise ValueError("All error values must be positive.")

    # --- Inspect model signature ---
    sig = inspect.signature(model)
    param_names = list(sig.parameters.keys())[1:]  # skip x

    if not param_names:
        raise ValueError("Model must have at least one parameter after x.")

    # --- Build lmfit.Parameters ---
    params = lmfit.Parameters()

    if isinstance(p0, dict):
        for name in param_names:
            val = p0[name]
            if isinstance(val, dict):
                params.add(name, **val)
            else:
                params.add(name, value=val)
    else:
        if len(p0) != len(param_names):
            raise ValueError(
                "Length of p0 does not match number of model parameters."
            )
        for name, val in zip(param_names, p0):
            params.add(name, value=val)

    # ------------------------------
    # lmfit.Model backend
    # ------------------------------
    if backend == "model":
        lm_model = lmfit.Model(model, independent_vars=["x"])
        result = lm_model.fit(
            y,
            params,
            x=x,
            weights=1.0 / err,
            method=fit_method,
            **fit_kws,
        )

    # ------------------------------
    # lmfit.Minimizer backend
    # ------------------------------
    elif backend == "minimizer":

        def residual(params):
            pvals = [params[name].value for name in param_names]
            return (y - model(x, *pvals)) / err

        minner = lmfit.Minimizer(residual, params)
        result = minner.minimize(method=fit_method, **fit_kws)

    else:
        raise ValueError("backend must be either 'model' or 'minimizer'")

    # # ------------------------------
    # # SciPy-style outputs
    # # ------------------------------
    # if return_scipy:
    #     popt = np.array([result.params[n].value for n in param_names])

    #     perr = np.array([
    #         result.params[n].stderr
    #         if result.params[n].stderr is not None else np.nan
    #         for n in param_names
    #     ])

    #     pcov = getattr(result, "covar", None)

    #     # if return_result:
    #     #     return popt, perr, pcov, result
    #     return popt, perr, pcov

    # return result


    # --- Extract SciPy-style outputs ---
    popt = np.array([result.params[name].value for name in param_names])
    perr = np.array([
        result.params[name].stderr
        if result.params[name].stderr is not None else np.nan
        for name in param_names
    ])

    if return_result:
        return popt, perr, result

    return popt, perr