__all__ = ["argsort", "sort"]


import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def argsort(
    x: Value,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return jnp.argsort(x, axis=axis, descending=descending, stable=stable)


@quaxify
def sort(
    x: Value,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return jnp.sort(x, axis=axis, descending=descending, stable=stable)
