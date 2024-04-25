"""Sorting functions."""

__all__ = ["argsort", "sort"]


from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def argsort(
    x: ArrayLike,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return array_api.argsort(x, axis=axis, descending=descending, stable=stable)


@quaxify
def sort(
    x: ArrayLike,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return array_api.sort(x, axis=axis, descending=descending, stable=stable)
