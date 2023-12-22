"""Utility functions."""

__all__ = ["all", "any"]

import jax.numpy as jnp
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
    return jnp.all(x, axis=axis, keepdims=keepdims)


@quaxify
def any(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.any(x, axis=axis, keepdims=keepdims)
