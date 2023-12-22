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
    descending: bool = False,  # TODO: support  # pylint: disable=unused-argument
    stable: bool = True,
) -> Value:
    return jnp.argsort(x, axis=axis, kind="stable" if stable else "quicksort")


@quaxify
def sort(
    x: Value,
    /,
    *,
    axis: int = -1,
    descending: bool = False,  # TODO: support  # pylint: disable=unused-argument
    stable: bool = True,
) -> Value:
    return jnp.sort(x, axis=axis, kind="stable" if stable else "quicksort")
