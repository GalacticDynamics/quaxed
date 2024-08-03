# ruff:noqa: F822

"""Quaxed :mod:`jax.scipy.linalg`.

This module wraps the functions in :external:`jax.scipy.linalg` with
:external:`quax.quaxify`. The wrapping happens dynamically through a
module-level ``__dir__`` and ``__getattr__``. The list of available functions is
in ``__all__`` and documented in the built-in :external:`jax.scipy.linalg`
library.

"""

__all__ = [
    "svd",
]

import sys
from collections.abc import Callable
from typing import Any

import jax.scipy.linalg
from quax import quaxify


def __dir__() -> list[str]:
    return sorted(__all__)


# TODO: better return type annotation
def __getattr__(name: str) -> Callable[..., Any]:
    if name not in __all__:
        msg = f"Cannot get {name} from quaxed.scipy.linalg."
        raise AttributeError(msg)

    # Quaxify the func
    func = quaxify(getattr(jax.scipy.linalg, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, func)

    return func
