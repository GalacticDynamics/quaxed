__all__ = ["matmul", "matrix_transpose", "tensordot", "vecdot"]


from collections.abc import Sequence

import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def matmul(x1: Value, x2: Value, /) -> Value:
    return jnp.matmul(x1, x2)


@quaxify
def matrix_transpose(x: Value, /) -> Value:
    return jnp.transpose(x)


@quaxify
def tensordot(
    x1: Value,
    x2: Value,
    /,
    *,
    axes: int | tuple[Sequence[int], Sequence[int]] = 2,
) -> Value:
    return jnp.tensordot(x1, x2, axes=axes)


@quaxify
def vecdot(x1: Value, x2: Value, /, *, axis: int = -1) -> Value:
    return jnp.dot(x1, x2, axis=axis)
