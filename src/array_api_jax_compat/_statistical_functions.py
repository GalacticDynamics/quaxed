__all__ = ["cumulative_sum", "max", "mean", "min", "prod", "std", "sum", "var"]


import jax.numpy as jnp
from quax import Value

from ._types import DType
from ._utils import quaxify


@quaxify
def cumulative_sum(
    x: Value,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Value:
    return jnp.cumsum(x, axis=axis, dtype=dtype)


@quaxify
def max(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.max(x, axis=axis, keepdims=keepdims)


@quaxify
def mean(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.mean(x, axis=axis, keepdims=keepdims)


@quaxify
def min(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.min(x, axis=axis, keepdims=keepdims)


@quaxify
def prod(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def std(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return jnp.std(x, axis=axis, correction=correction, keepdims=keepdims)


@quaxify
def sum(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return jnp.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def var(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return jnp.var(x, axis=axis, correction=correction, keepdims=keepdims)
