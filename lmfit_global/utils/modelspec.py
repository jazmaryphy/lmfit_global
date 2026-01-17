# %%
from typing import Callable, Dict, Any
from dataclasses import dataclass, field

# %%
@dataclass(frozen=True)
class ModelSpec:
    """
    Declarative specification of a model used in global fitting.

    Attributes:
        func:
            Callable model function with signature ``f(x, *params)``.
        init_params:
            Dictionary defining initial parameters and lmfit options.
        func_kws:
            Optional keyword arguments passed to the model function.
    """
    func: Callable
    init_params: Dict[str, dict]
    func_kws: Dict[str, Any] = field(default_factory=dict)