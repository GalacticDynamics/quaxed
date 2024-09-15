"""Quaxed :mod:`jax.numpy.linalg`."""

__all__ = [  # noqa: F822
    "cholesky",
    "cross",
    "det",
    "diagonal",
    "eig",
    "eigh",
    "eigvalsh",
    "inv",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
    "norm",
]

import sys
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from quax import quaxify


def __dir__() -> list[str]:
    return sorted(__all__)


def __getattr__(name: str) -> Callable[..., Any]:  # TODO: better type hint
    """Get the object from the `jax.numpy` module."""
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    # Get the object
    jnp_obj = getattr(jnp.linalg, name)

    # Quaxify?
    out = quaxify(jnp_obj)

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
