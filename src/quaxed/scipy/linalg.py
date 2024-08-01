# ruff:noqa: F822

"""Quaxed :mod:`jax.scipy.linalg`."""

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
    # Quaxify the func
    func = quaxify(getattr(jax.scipy.linalg, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, func)

    return func
