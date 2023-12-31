__all__ = ["max", "mean", "min", "prod", "std", "sum", "var"]


from jax.experimental import array_api
from quax import Value

from array_api_jax_compat._types import DType
from array_api_jax_compat._utils import quaxify


@quaxify
def max(  # pylint: disable=redefined-builtin
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.max(x, axis=axis, keepdims=keepdims)


@quaxify
def mean(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.mean(x, axis=axis, keepdims=keepdims)


@quaxify
def min(  # pylint: disable=redefined-builtin
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.min(x, axis=axis, keepdims=keepdims)


@quaxify
def prod(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def std(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return array_api.std(x, axis=axis, correction=correction, keepdims=keepdims)


@quaxify
def sum(  # pylint: disable=redefined-builtin
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def var(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return array_api.var(x, axis=axis, correction=correction, keepdims=keepdims)
