"""Compatibility wrapper for MAT trainer.

Historically, the project imported ``TrainerRPC`` from
``train_rpc_mat``.  The actual implementation now lives in
``train_rpc``.  To avoid import-time circular dependencies we expose
``TrainerRPC`` lazily via ``__getattr__``.
"""

from typing import Any


def __getattr__(name: str) -> Any:
    """Dynamically fetch attributes from :mod:`train_rpc`.

    The function imports :mod:`train_rpc` only when an attribute is
    requested.  This mirrors the old interface while preventing
    circular imports during module initialisation.
    """

    if name == "TrainerRPC":
        from .train_rpc import TrainerRPC

        return TrainerRPC
    raise AttributeError(name)


__all__ = ["TrainerRPC"]
