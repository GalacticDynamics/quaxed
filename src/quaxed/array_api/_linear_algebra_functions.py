__all__ = ["matmul", "matrix_transpose", "tensordot", "vecdot"]


from collections.abc import Sequence

from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def matmul(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.matmul(x1, x2)


@quaxify
def matrix_transpose(x: ArrayLike, /) -> Value:
    return array_api.matrix_transpose(x)


@quaxify
def tensordot(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Value:
    return array_api.tensordot(x1, x2, axes=axes)


@quaxify
def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1) -> Value:
    return array_api.vecdot(x1, x2, axis=axis)
