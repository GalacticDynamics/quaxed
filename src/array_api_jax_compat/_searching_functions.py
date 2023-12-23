__all__ = ["argmax", "argmin", "nonzero", "where"]


from jax.experimental import array_api
from quax import Value

from ._utils import quaxify


@quaxify
def argmax(x: Value, /, *, axis: int | None = None, keepdims: bool = False) -> Value:
    return array_api.argmax(x, axis=axis, keepdims=keepdims)


@quaxify
def argmin(x: Value, /, *, axis: int | None = None, keepdims: bool = False) -> Value:
    return array_api.argmin(x, axis=axis, keepdims=keepdims)


@quaxify
def nonzero(x: Value, /) -> tuple[Value, ...]:
    return array_api.nonzero(x)


@quaxify
def where(condition: Value, x1: Value, x2: Value, /) -> Value:
    return array_api.where(condition, x1, x2)
