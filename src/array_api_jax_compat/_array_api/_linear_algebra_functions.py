__all__ = ["matmul", "matrix_transpose", "tensordot", "vecdot"]


from collections.abc import Sequence

from jax.experimental import array_api
from quax import Value

from array_api_jax_compat._utils import quaxify


@quaxify
def matmul(x1: Value, x2: Value, /) -> Value:
    return array_api.matmul(x1, x2)


@quaxify
def matrix_transpose(x: Value, /) -> Value:
    return array_api.matrix_transpose(x)


@quaxify
def tensordot(
    x1: Value,
    x2: Value,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Value:
    return array_api.tensordot(x1, x2, axes=axes)


@quaxify
def vecdot(x1: Value, x2: Value, /, *, axis: int = -1) -> Value:
    return array_api.vecdot(x1, x2, axis=axis)
