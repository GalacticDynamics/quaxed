__all__ = ["argmax", "argmin", "nonzero", "where"]


import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def argmax(x: Value, /, *, axis: int | None = None, keepdims: bool = False) -> Value:
    return jnp.argmax(x, axis=axis, keepdims=keepdims)


@quaxify
def argmin(x: Value, /, *, axis: int | None = None, keepdims: bool = False) -> Value:
    return jnp.argmin(x, axis=axis, keepdims=keepdims)


@quaxify
def nonzero(x: Value, /) -> tuple[Value, ...]:
    return jnp.nonzero(x)


@quaxify
def where(condition: Value, x1: Value, x2: Value, /) -> Value:
    return jnp.where(condition, x1, x2)
