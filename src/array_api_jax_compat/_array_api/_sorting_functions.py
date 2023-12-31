"""Sorting functions."""

__all__ = ["argsort", "sort"]


from jax.experimental import array_api
from quax import Value

from array_api_jax_compat._utils import quaxify


@quaxify
def argsort(
    x: Value,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return array_api.argsort(x, axis=axis, descending=descending, stable=stable)


@quaxify
def sort(
    x: Value,
    /,
    *,
    axis: int = -1,
    descending: bool = False,
    stable: bool = True,
) -> Value:
    return array_api.sort(x, axis=axis, descending=descending, stable=stable)
