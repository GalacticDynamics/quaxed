"""Linear algebra functions."""

__all__ = [
    "cholesky",
    "cross",
    "det",
    "diagonal",
    "eigh",
    "eigvalsh",
    "inv",
    "matmul",
    "matrix_norm",
    "matrix_power",
    "matrix_rank",
    "matrix_transpose",
    "outer",
    "pinv",
    "qr",
    "slogdet",
    "solve",
    "svd",
    "svdvals",
    "tensordot",
    "trace",
    "vecdot",
    "vector_norm",
]


from collections.abc import Sequence
from typing import Literal

import jax.numpy as jnp
from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._types import DType
from quaxed._utils import quaxify


@quaxify
def cholesky(x: ArrayLike, /, *, upper: bool = False) -> Value:
    return array_api.linalg.cholesky(x, upper=upper)


@quaxify
def cross(x1: ArrayLike, x2: ArrayLike, /, *, axis: int = -1) -> Value:
    return array_api.linalg.cross(x1, x2, axis=axis)


@quaxify
def det(x: ArrayLike, /) -> Value:
    return array_api.linalg.det(x)


@quaxify
def diagonal(x: ArrayLike, /, *, offset: int = 0) -> Value:
    return array_api.linalg.diagonal(x, offset=offset)


@quaxify
def eigh(x: ArrayLike, /) -> tuple[ArrayLike]:
    return array_api.linalg.eigh(x)


@quaxify
def eigvalsh(x: ArrayLike, /) -> Value:
    return array_api.linalg.eigvalsh(x)


@quaxify
def inv(x: ArrayLike, /) -> Value:
    return array_api.linalg.inv(x)


@quaxify
def matmul(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.matmul(x1, x2)


@quaxify
def matrix_norm(
    x: ArrayLike,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | Literal["fro", "nuc"] | None = "fro",
) -> Value:
    return array_api.linalg.matrix_norm(x, keepdims=keepdims, ord=ord)


@quaxify
def matrix_power(x: ArrayLike, n: int, /) -> Value:
    return array_api.linalg.matrix_power(x, n)


@quaxify
def matrix_rank(x: ArrayLike, /, *, rtol: ArrayLike | None = None) -> Value:
    return array_api.linalg.matrix_rank(x, rtol=rtol)


@quaxify
def matrix_transpose(x: ArrayLike, /) -> Value:
    return array_api.linalg.matrix_transpose(x)


@quaxify
def outer(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.linalg.outer(x1, x2)


@quaxify
def pinv(x: ArrayLike, /, *, rtol: ArrayLike | None = None) -> Value:
    return array_api.linalg.pinv(x, rtol=rtol)


@quaxify
def qr(
    x: ArrayLike,
    /,
    *,
    mode: Literal["reduced", "complete"] = "reduced",
) -> tuple[ArrayLike, ArrayLike]:
    return array_api.linalg.qr(x, mode=mode)


@quaxify
def slogdet(x: ArrayLike, /) -> tuple[Value, Value]:
    return array_api.linalg.slogdet(x)


@quaxify
def solve(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.linalg.solve(x1, x2)


@quaxify
def svd(x: ArrayLike, /, *, full_matrices: bool = True) -> tuple[Value, Value, Value]:
    return array_api.linalg.svd(x, full_matrices=full_matrices)


@quaxify
def svdvals(x: ArrayLike, /) -> Value:
    return array_api.linalg.svdvals(x)


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
def trace(x: ArrayLike, /, *, offset: int = 0, dtype: DType | None = None) -> Value:
    return jnp.trace(x, offset=offset, dtype=dtype)


@quaxify
def vecdot(x1: ArrayLike, x2: ArrayLike, /, *, axis: int | None = None) -> Value:
    return array_api.vecdot(x1, x2, axis=axis)


@quaxify
def vector_norm(
    x: ArrayLike,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,  # pylint: disable=redefined-builtin
) -> Value:
    return array_api.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
