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
    "schur",
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
    """List the module contents."""
    return sorted(__all__)


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the :external:`quax.quaxify`'ed function."""
    if name not in __all__:
        msg = f"Cannot get {name} from quaxed.lax.linalg"
        raise AttributeError(msg)

    # Quaxify the operator
    out = quaxify(getattr(lax.linalg, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
