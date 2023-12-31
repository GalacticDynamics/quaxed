"""Utility functions."""

__all__ = ["all", "any"]

from jax.experimental import array_api
from quax import Value

from ._utils import quaxify


@quaxify
def all(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.all(x, axis=axis, keepdims=keepdims)


@quaxify
def any(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.any(x, axis=axis, keepdims=keepdims)
