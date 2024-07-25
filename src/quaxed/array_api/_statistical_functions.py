__all__ = ["cumulative_sum", "max", "mean", "min", "prod", "std", "sum", "var"]


from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._types import DType
from quaxed._utils import quaxify


@quaxify
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    dtype: DType | None = None,
    include_initial: bool = False,
) -> Value:
    return array_api.cumulative_sum(
        x, axis=axis, dtype=dtype, include_initial=include_initial
    )


@quaxify
def max(  # pylint: disable=redefined-builtin
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.max(x, axis=axis, keepdims=keepdims)


@quaxify
def mean(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.mean(x, axis=axis, keepdims=keepdims)


@quaxify
def min(  # pylint: disable=redefined-builtin
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.min(x, axis=axis, keepdims=keepdims)


@quaxify
def prod(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.prod(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def std(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return array_api.std(x, axis=axis, correction=correction, keepdims=keepdims)


@quaxify
def sum(  # pylint: disable=redefined-builtin
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    dtype: DType | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.sum(x, axis=axis, dtype=dtype, keepdims=keepdims)


@quaxify
def var(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    correction: int | float = 0.0,
    keepdims: bool = False,
) -> Value:
    return array_api.var(x, axis=axis, correction=correction, keepdims=keepdims)
