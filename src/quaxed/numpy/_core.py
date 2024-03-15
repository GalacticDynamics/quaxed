"""Quaxed :mod:`jax.numpy`."""

__all__ = [
    "allclose",
    "array_equal",
    "cbrt",
    "copy",
    "equal",
    "exp2",
    "greater",
    "hypot",
    "matmul",
    "moveaxis",
    "trace",
    "vectorize",
]

from collections.abc import Callable, Iterable
from typing import Any, TypeVar

import jax.numpy as jnp
from quax import quaxify

T = TypeVar("T")


def _doc(jax_func: Callable[..., Any]) -> Callable[[T], T]:
    """Copy docstrings from JAX functions."""

    def transfer_doc(func: T) -> T:
        """Copy docstrings from JAX functions."""
        func.__doc__ = jax_func.__doc__
        return func

    return transfer_doc


##############################################################################


allclose = quaxify(jnp.allclose)
array_equal = quaxify(jnp.array_equal)
cbrt = quaxify(jnp.cbrt)
copy = quaxify(jnp.copy)
equal = quaxify(jnp.equal)
exp2 = quaxify(jnp.exp2)
greater = quaxify(jnp.greater)
hypot = quaxify(jnp.hypot)
matmul = quaxify(jnp.matmul)
moveaxis = quaxify(jnp.moveaxis)
trace = quaxify(jnp.trace)


@_doc(jnp.vectorize)
def vectorize(
    pyfunc: Callable[..., Any],
    *,
    excluded: Iterable[int] = frozenset(),
    signature: str | None = None,
) -> Callable[..., Any]:
    return quaxify(jnp.vectorize(pyfunc, excluded=excluded, signature=signature))
