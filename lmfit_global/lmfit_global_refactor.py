"""
Refactored and cleaned version of LmfitGlobal.
- Removes duplicated methods
- Caches function signatures
- Unified parameter update/tie API
- Improved NaN handling and safer residual computation
- Clearer logging and warnings

This single-file refactor keeps the original external API names where possible
but simplifies internals and removes duplicated blocks.

Note: This is a refactor for readability and maintainability. Behavior should be
carefully tested against your existing unit tests and datasets.
"""

from __future__ import annotations

import copy
import functools as ft
import inspect
import itertools
import logging
import operator
import textwrap
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import lmfit
except Exception as exc:  # pragma: no cover - runtime dependency
    raise ImportError("lmfit is required. Install with `pip install lmfit`") from exc

# Optional imports
try:
    from sklearn.metrics import r2_score as sk_r2
except Exception:
    sk_r2 = None

# local util fallback (if available in your environment)
try:
    from globutils import r2_score_util, alphanumeric_sort, getfloat_attr, gformat
except Exception:
    # fallback minimal implementations if globutils isn't installed
    def r2_score_util(y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        ss_res = np.nansum((y_true - y_pred) ** 2)
        ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan

    def alphanumeric_sort(x):
        return x

    def getfloat_attr(obj, attr):
        return getattr(obj, attr, None)

    def gformat(x):
        try:
            return f"{x:.7g}"
        except Exception:
            return str(x)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
if not logger.handlers:
    logger.addHandler(handler)

_VALID_CONNECTORS = {
    '+': operator.add,
    '-': operator.sub,
    '*': operator.mul,
    '/': operator.truediv,
}


@dataclass
class ModelSpec:
    func: Callable
    init_params: Dict[str, dict]
    func_kws: Dict[str, object] = field(default_factory=dict)


class LmfitGlobal:
    """Refactored class providing a simplified and safer interface for multi-component
    fits across multiple datasets (columns).

    Key improvements over the legacy file:
      - single definitions for expr/param updating
      - signature caching
      - unified parameter tie/update API
      - clear NaN handling policies
    """

    def __init__(self, items: dict, independent_vars: Optional[List[str]] = None,
                 nan_policy: str = 'raise', method: str = 'leastsq', **fit_kws):
        self.items = copy.deepcopy(items)
        self.method = method
        self.fit_kws = fit_kws
        self.nan_policy = nan_policy
        self.independent_vars = independent_vars

        # internal state
        self.models: List[lmfit.Model] = []
        self.model_specs: List[ModelSpec] = []
        self.prefixes: List[str] = []
        self.func_signatures: Dict[Callable, inspect.Signature] = {}

        self._parse_data()
        self._parse_functions()

        # build parameters and composite model
        self.initial_params = lmfit.Parameters()
        self._create_models()
        self._init_parameters()

        # cached evaluation
        self.y_sim: Optional[np.ndarray] = None
        self.result: Optional[lmfit.MinimizerResult] = None

        logger.info("LmfitGlobal initialized.")

    # -----------------------------
    # Parsing and validation
    # -----------------------------
    def _parse_data(self):
        data = self.items.get('data') or {}
        xy = data.get('xy')
        if xy is None:
            raise ValueError('`data.xy` missing')
        arr = np.asarray(xy)
        if arr.ndim != 2:
            raise ValueError('`data.xy` must be 2D array-like with columns [x, y1, y2, ...]')

        self.data_x = arr[:, 0]
        self.data_y = arr[:, 1:]
        self.N, self.ny = self.data_y.shape
        self.data_xrange = data.get('xrange')

        if self.data_xrange is not None:
            if (not isinstance(self.data_xrange, (tuple, list)) or len(self.data_xrange) != 2):
                raise ValueError('`data.xrange` must be a (min, max) pair')
            idx = np.where((self.data_x >= self.data_xrange[0]) & (self.data_x <= self.data_xrange[1]))[0]
            if idx.size == 0:
                raise ValueError('No data in provided xrange')
            self.xdat = self.data_x[idx]
            self.ydat = np.take(self.data_y, idx, axis=0)
        else:
            self.xdat = self.data_x
            self.ydat = self.data_y

        self.has_nan = np.isnan(self.ydat).any()
        if self.has_nan and self.nan_policy == 'raise':
            raise ValueError('NaNs present in data but nan_policy="raise"')

    def _parse_functions(self):
        funcs = self.items.get('functions') or {}
        theory = funcs.get('theory')
        if not theory or not isinstance(theory, list):
            raise ValueError('`functions.theory` must be a non-empty list')

        self.model_specs = []
        for entry in theory:
            func = entry.get('func_name')
            if not callable(func):
                raise ValueError('each theory entry must provide a callable `func_name`')
            init_params = entry.get('init_params')
            if not isinstance(init_params, dict):
                raise ValueError('`init_params` must be dict of parameter hints')
            func_kws = entry.get('func_kws', {}) or {}
            # cache signature
            self.func_signatures[func] = inspect.signature(func)
            self.model_specs.append(ModelSpec(func=func, init_params=init_params, func_kws=func_kws))

        self.theory_connectors: List[str] = funcs.get('theory_connectors') or []
        if len(self.model_specs) > 1:
            if not isinstance(self.theory_connectors, list) or len(self.theory_connectors) != len(self.model_specs) - 1:
                raise ValueError('`theory_connectors` must be list of length n_models-1')
            for op in self.theory_connectors:
                if op not in _VALID_CONNECTORS:
                    raise ValueError(f'Unsupported connector: {op}')
        else:
            self.theory_connectors = []

    # -----------------------------
    # Model construction and params
    # -----------------------------
    def _create_models(self):
        self.models = []
        self.prefixes = []
        for i, spec in enumerate(self.model_specs):
            prefix = f'c{i}_' if len(self.model_specs) > 1 else ''
            indep_vars = [list(self.func_signatures[spec.func].parameters.keys())[0]] + list(spec.func_kws.keys())
            model = lmfit.Model(spec.func, prefix=prefix, independent_vars=indep_vars)
            model.func_kws = spec.func_kws
            self.models.append(model)
            self.prefixes.append(prefix)

    def _init_parameters(self):
        self.initial_params = lmfit.Parameters()
        for iy in range(self.ny):
            for model in self.models:
                pset = model.make_params()
                for pname, p in pset.items():
                    name = f'{pname}_{iy}'
                    # copy relevant attributes
                    self.initial_params.add(name,
                                            value=p.value,
                                            vary=p.vary,
                                            min=p.min if p.min is not None else -np.inf,
                                            max=p.max if p.max is not None else np.inf)
        logger.info('Initialized parameters')

    # -----------------------------
    # Parameter utilities (unified)
    # -----------------------------
    def _normalize_names(self, inputs: Iterable) -> List[str]:
        names = []
        for it in inputs:
            if isinstance(it, str):
                names.append(it)
            elif isinstance(it, lmfit.Parameter):
                names.append(it.name)
            elif isinstance(it, lmfit.Parameters):
                names.extend(list(it.keys()))
            elif isinstance(it, (list, tuple)):
                names.extend(self._normalize_names(it))
            else:
                raise TypeError(f'Unsupported param spec: {type(it)}')
        # deduplicate keeping order
        seen = set()
        out = []
        for n in names:
            if n not in seen:
                seen.add(n)
                out.append(n)
        return out

    def tie(self, names: Iterable, master: Optional[str] = None):
        """Make parameters listed in `names` share a single master parameter.

        If `master` is None the first name in `names` is used as master. Names
        should be full parameter names like 'sigma_0'.
        """
        names_list = self._normalize_names(names)
        if not names_list:
            raise ValueError('No parameter names supplied')
        if master is None:
            master = names_list[0]
        if master not in self.initial_params:
            raise ValueError(f'Master parameter {master} not found')
        # clear master expr
        self.initial_params[master].expr = None
        flag = self.initial_params[master].vary
        for pname in names_list:
            if pname == master:
                continue
            if pname not in self.initial_params:
                warnings.warn(f'Parameter {pname} not found; skipping')
                continue
            self.initial_params[pname].expr = master
        self.initial_params[master].set(vary=flag)
        logger.info(f'Tied {names_list} to master {master}')

    def set_expr(self, mapping: Dict[str, str]):
        """Set expression for given parameter names.
        mapping: {param_name: expr_string}
        """
        for name, expr in mapping.items():
            if name not in self.initial_params:
                raise ValueError(f'Parameter {name} not found')
            self.initial_params[name].expr = expr
            logger.info(f'Set expr for {name}: {expr}')

    def update_params(self, updates: Dict[str, dict]):
        """Update parameter attributes using a mapping {name: {k:v}}.
        Allowed keys: value, vary, min, max, expr, brute_step
        """
        allowed = {'value', 'vary', 'min', 'max', 'expr', 'brute_step'}
        for name, attrs in updates.items():
            if name not in self.initial_params:
                logger.warning(f'Adding new parameter {name} from update')
                # ensure we provide at least a value
                self.initial_params.add(name, **{k: v for k, v in attrs.items() if k in allowed})
                continue
            safe = {k: v for k, v in attrs.items() if k in allowed}
            if safe:
                self.initial_params[name].set(**safe)
                logger.info(f'Updated {name} with {safe}')

    # -----------------------------
    # Evaluation and residuals
    # -----------------------------
    def _evaluate_one_model(self, model: lmfit.Model, x: np.ndarray, params: lmfit.Parameters, i: int) -> np.ndarray:
        # collect parameter values from params using model.prefix
        prefix = model.prefix
        sig = self.func_signatures[model.func]
        # build kwargs for function from params and model.func_kws
        kw = dict(model.func_kws)  # constant kws
        for name in list(sig.parameters.keys())[1:]:
            param_key = f'{prefix}{name}_{i}' if prefix else f'{name}_{i}'
            if param_key in params:
                kw[name] = params[param_key].value
        return model.func(x, **kw)

    def _evaluate_all(self, params: lmfit.Parameters) -> np.ndarray:
        y_sim = np.zeros_like(self.ydat, dtype=float)
        for i in range(self.ny):
            # evaluate each model component
            comps = [self._evaluate_one_model(m, self.xdat, params, i) for m in self.models]
            # reduce with connectors
            if not comps:
                raise RuntimeError('No model components')
            out = comps[0]
            for op, comp in zip(self.theory_connectors, comps[1:]):
                out = _VALID_CONNECTORS[op](out, comp)
            y_sim[:, i] = out
        return y_sim

    def _residual(self, params: lmfit.Parameters) -> np.ndarray:
        y_sim = self._evaluate_all(params)
        diff = y_sim - self.ydat
        # Handle NaN policy
        if self.nan_policy == 'omit':
            mask = ~np.isnan(diff)
            return diff[mask]
        if self.nan_policy == 'propagate':
            return np.nan_to_num(diff, nan=0.0)
        # default: raise was checked earlier
        return diff

    # -----------------------------
    # fitting API
    # -----------------------------
    def fit(self, method: Optional[str] = None, params: Optional[lmfit.Parameters] = None, **kws):
        method = method or self.method
        params = params or self.initial_params
        minimizer = lmfit.Minimizer(lambda p: self._residual(p), params)
        out = minimizer.minimize(method=method, **kws)
        self.result = out
        # compute y_sim for result params
        try:
            self.y_sim = self._evaluate_all(out.params)
        except Exception:
            self.y_sim = None
        return out

    # -----------------------------
    # reporting helpers
    # -----------------------------
    def lmfit_report(self, result: Optional[lmfit.MinimizerResult] = None, show_correl: bool = True):
        r = result or self.result
        if r is None:
            raise ValueError('No fit result available')
        params = r.params
        parnames = [k for k in params.keys()]
        buff = [f'[[Fit method]] {r.method}']
        buff.append(f'chi-square = {getfloat_attr(r, "chisqr")}')
        # parameters
        buff.append('[[Variables]]')
        for name in parnames:
            p = params[name]
            sval = gformat(p.value) if p.value is not None else 'None'
            if p.stderr is not None:
                s = f'{sval} +/- {gformat(p.stderr)}'
            else:
                s = sval
            buff.append(f'  {name}: {s}  (vary={p.vary})')
        # simple R2 if possible
        try:
            if self.y_sim is None:
                self.y_sim = self._evaluate_all(params)
            r2 = self._r2_safe(self.ydat.flatten(), self.y_sim.flatten())
            buff.append(f'R2 (global) = {r2:.6g}')
        except Exception:
            pass
        return '\n'.join(buff)

    def _r2_safe(self, y_true, y_pred, **kwargs):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if not np.isnan(y_true).any() and sk_r2 is not None:
            try:
                return sk_r2(y_true, y_pred, **kwargs)
            except Exception:
                pass
        return r2_score_util(y_true, y_pred, **kwargs)

    def __repr__(self):
        return f"LmfitGlobal(n_models={len(self.models)}, ny={self.ny}, nan_policy='{self.nan_policy}')"
