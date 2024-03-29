__all__ = ["argmax", "argmin", "nonzero", "where"]


from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def argmax(
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.argmax(x, axis=axis, keepdims=keepdims)


@quaxify
def argmin(
    x: ArrayLike,
    /,
    *,
    axis: int | None = None,
    keepdims: bool = False,
) -> Value:
    return array_api.argmin(x, axis=axis, keepdims=keepdims)


@quaxify
def nonzero(x: ArrayLike, /) -> tuple[Value, ...]:
    return array_api.nonzero(x)


@quaxify
def where(condition: ArrayLike, x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.where(condition, x1, x2)
