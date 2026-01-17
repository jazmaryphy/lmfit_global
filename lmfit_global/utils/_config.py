# %%
"""Configurations."""

from __future__ import annotations

import sys
import logging
from typing import Optional, Protocol, TypeAlias, TYPE_CHECKING

class LoggerLike(Protocol):
    """Minimal protocol for logger-like objects."""

    def debug(self, msg: str, *args, **kwargs) -> None: ...
    def info(self, msg: str, *args, **kwargs) -> None: ...
    def warning(self, msg: str, *args, **kwargs) -> None: ...
    def error(self, msg: str, *args, **kwargs) -> None: ...
    def exception(self, msg: str, *args, **kwargs) -> None: ...


Logger: TypeAlias = LoggerLike
# Logger: TypeAlias = logging.Logger

# %%
# def get_default_logger(
#     name: str,
#     *,
#     log_level: Optional[int] = None,
#     # log_level: Optional[int] = logging.INFO,
#     propagate: bool = False,
# ) -> logging.Logger:
#     """Create or return a safe default logger.

#     This function:
#     - Does not configure global logging
#     - Avoids duplicate handlers
#     - Adds a fallback StreamHandler only if logging is unconfigured
#     - Plays well with application-level logging configs

#     Args:
#         name: Logger name (typically ``__name__``).
#         log_level (int): Logging level (default: INFO). 
#             If None, inherit from parent.
#         propagate: Whether to propagate records to the root logger.

#     Returns:
#         A configured ``logging.Logger`` instance.
#     """
#     logger = logging.getLogger(name)

#     if log_level is not None:
#         try:
#             logger.setLevel(getattr(logging, log_level.upper()))
#         except:
#             logger.warning(f"Invalid logging level log_level='{log_level}' ...")
#     # logger.setLevel(level)

#     logger.propagate = propagate

#     # Fallback handler only if *no* handlers exist anywhere
#     if not logging.getLogger().handlers:
#         handler = logging.StreamHandler(sys.stderr)
#         formatter = logging.Formatter(
#             "%(levelname)s: %(name)s: %(message)s"
#             # "%(levelname)s: %(message)s"
#         )
#         handler.setFormatter(formatter)
#         logger.addHandler(handler)

#     return logger

# %%
def get_default_logger(
    name: str,
    *,
    log_level: str = "",
    propagate: bool = False,
) -> logging.Logger:
    """
    Create or return a safe default logger.

    This function is intended for use inside libraries (not applications).
    It does not configure global logging and avoids adding duplicate handlers.
    A fallback StreamHandler is attached only if no handlers exist anywhere
    in the logging hierarchy.

    Args:
        name:
            Logger name (typically ``__name__`` or a class name).
        log_level:
            Optional logging level specified as a string
            (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).

            - If an empty string (default), the logger inherits its level
              from its parent and no warnings are emitted.
            - If a valid level name is provided, the logger level is set.
            - If an invalid non-empty value is provided, a warning is logged.
        propagate:
            Whether to propagate log records to the parent logger.

    Returns:
        logging.Logger:
            A configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.propagate = propagate

    # Handle log level safely
    if isinstance(log_level, str):
        level_str = log_level.strip()

        if level_str:
            level_name = level_str.upper()
            level = logging._nameToLevel.get(level_name)

            if level is not None:
                logger.setLevel(level)
            else:
                logger.warning(
                    "Invalid log_level='%s'. Valid values are: %s",
                    log_level,
                    ", ".join(logging._nameToLevel.keys()),
                )

    # Add fallback handler only if logging is entirely unconfigured
    root_logger = logging.getLogger()
    if not root_logger.handlers and not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            # logging.Formatter("%(levelname)s: %(message)s")
            logging.Formatter("%(levelname)s: %(name)s: %(message)s")
        )
        logger.addHandler(handler)

    return logger