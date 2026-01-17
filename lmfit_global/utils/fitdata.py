import numpy as np
from ._typing import LmfitGlobalLike
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Any

# %%
@dataclass(slots=True)
class FitData:
    x_data: np.ndarray                                # raw data
    y_data: np.ndarray                                # raw data
    x_model: np.ndarray                               # model grid (same as data or dense)
    y_init: np.ndarray                                # initial model
    y_fit: np.ndarray | None                          # best-fit model (optional)
    resid_init: np.ndarray                            # initial residual
    resid_fit: np.ndarray | None                      # best-fit model residual (optional)
    components: Optional[
        Dict[int, Dict[str, Dict[str, np.ndarray]]]  
    ] = None                                         # multicomponent results (optional)
    rsquared: Optional[float] = None                 # diagnostics

    """
    # --- components ---
    # single-dataset:
    #   {component_name: {"data": ndarray, "model": ndarray}}
    #
    # multi-dataset:
    #   {dataset_index: {component_name: {"data": ndarray, "model": ndarray}}}
    """

    # ---- derived properties ----
    @property
    def has_fit(self) -> bool:
        return self.y_fit is not None

    @property
    def ny(self) -> int:
        return self.y_data.shape[1]

    @property
    def n_datasets(self) -> int:
        return self.ny
    
    @property
    def is_multidataset(self) -> bool:
        return self.ny > 1

    @property
    def has_components(self) -> bool:
        return self.components is not None

    @property
    def component_names(self) -> list[str]:
        if not self.components:
            return []

        first_val = next(iter(self.components.values()))

        # Multi-dataset case: {dataset_idx: {component_name: {...}}}
        if isinstance(first_val, dict) and "data" not in first_val:
            names = set()
            for comps in self.components.values():
                names.update(comps.keys())
            return sorted(names)

        # Single-dataset case
        return sorted(self.components.keys())

    @property
    def is_multicomponent(self) -> bool:
        return len(self.component_names) > 1

    @property
    def nc(self) -> int:
        return len(self.component_names)

    @property
    def n_components(self) -> int:
        return self.nc

    # -----------------
    # factory
    # -----------------
    @classmethod
    def from_lmfitglobal(
        cls,
        lg: LmfitGlobalLike,
        *,
        numpoints: int | None = None,
    ) -> "FitData":
        """
        Build FitData from an LmfitGlobal instance.

        Notes
        -----
        - Residuals are always computed on the data grid
        - Dense grid is used only for visualization
        """
        # --- raw data ---
        x_data = np.asarray(lg.x_data, dtype=float)
        y_data = np.asarray(lg.y_data, dtype=float)

        if y_data.ndim != 2:
            raise ValueError("y_data must be 2D (N, ny)")

        if len(x_data) != y_data.shape[0]:
            raise ValueError(
                f"x_data length ({len(x_data)}) != y_data rows ({y_data.shape[0]})"
            )

        # --- model grid ---
        use_dense = numpoints is not None and x_data.size < numpoints
        x_model = (
            np.linspace(x_data.min(), x_data.max(), numpoints)
            if use_dense else x_data
        )

        # --- initial model ---
        y_init_data = lg.eval(x=x_data, params=lg.init_params)
        y_init_model = (
            y_init_data if not use_dense
            else lg.eval(x=x_model, params=lg.init_params)
        )
        resid_init = y_data - y_init_data

        # --- best fit (optional) ---
        y_fit_model = None
        resid_fit = None
        components = None
        rsquared = None

        if getattr(lg, "fit_success", False):
            y_fit_data = lg.eval(x=x_data, params=lg.result.params)
            resid_fit = y_data - y_fit_data
            y_fit_model = (
                y_fit_data if not use_dense
                else lg.eval(x=x_model, params=lg.result.params)
            )

            rsquared = lg.rsquared

            if lg.is_multicomponent:
                components = lg.eval_components(
                    x_data=x_data,
                    x_model=x_model,
                    params=lg.result.params,
                )

        # print(f'numpoints = {numpoints} and len =  {len(x_model)}')

        return cls(
            x_data=x_data,
            y_data=y_data,
            x_model=x_model,
            y_init=y_init_model,
            resid_init=resid_init,
            y_fit=y_fit_model,
            resid_fit=resid_fit,
            components=components,
            rsquared=rsquared,
        )