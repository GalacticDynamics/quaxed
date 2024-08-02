"""Quaxed :mod:`jax.lax`."""
# pylint: disable=undefined-all-variable

__all__ = [  # noqa: F822
    "cholesky",
    "eig",
    "eigh",
    "hessenberg",
    "lu",
    "householder_product",
    "qdwh",
    "qr",
    "shur",
    "svd",
    "triangular_solve",
    "tridiagonal",
    "tridiagonal_solve",
]


import sys
from collections.abc import Callable
from typing import Any

from jax import lax
from quax import quaxify


def __dir__() -> list[str]:
    """List the operators."""
    return sorted(__all__)


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the operator."""
    # Quaxify the operator
    out = quaxify(getattr(lax.linalg, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
