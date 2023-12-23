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
from quax import Value

from ._types import DType
from ._utils import quaxify


@quaxify
def cholesky(x: Value, /, *, upper: bool = False) -> Value:
    return array_api.linalg.cholesky(x, upper=upper)


@quaxify
def cross(x1: Value, x2: Value, /, *, axis: int = -1) -> Value:
    return array_api.linalg.cross(x1, x2, axis=axis)


@quaxify
def det(x: Value, /) -> Value:
    return array_api.linalg.det(x)


@quaxify
def diagonal(x: Value, /, *, offset: int = 0) -> Value:
    return array_api.linalg.diagonal(x, offset=offset)


@quaxify
def eigh(x: Value, /) -> tuple[Value]:
    return array_api.linalg.eigh(x)


@quaxify
def eigvalsh(x: Value, /) -> Value:
    return array_api.linalg.eigvalsh(x)


@quaxify
def inv(x: Value, /) -> Value:
    return array_api.linalg.inv(x)


@quaxify
def matmul(x1: Value, x2: Value, /) -> Value:
    return array_api.matmul(x1, x2)


@quaxify
def matrix_norm(
    x: Value,
    /,
    *,
    keepdims: bool = False,
    ord: int | float | Literal["fro", "nuc"] | None = "fro",
) -> Value:
    return array_api.linalg.matrix_norm(x, keepdims=keepdims, ord=ord)


@quaxify
def matrix_power(x: Value, n: int, /) -> Value:
    return array_api.linalg.matrix_power(x, n)


@quaxify
def matrix_rank(x: Value, /, *, rtol: float | Value | None = None) -> Value:
    return array_api.linalg.matrix_rank(x, rtol=rtol)


@quaxify
def matrix_transpose(x: Value, /) -> Value:
    return array_api.linalg.matrix_transpose(x)


@quaxify
def outer(x1: Value, x2: Value, /) -> Value:
    return array_api.linalg.outer(x1, x2)


@quaxify
def pinv(x: Value, /, *, rtol: float | Value | None = None) -> Value:
    return array_api.linalg.pinv(x, rtol=rtol)


@quaxify
def qr(
    x: Value,
    /,
    *,
    mode: Literal["reduced", "complete"] = "reduced",
) -> tuple[Value, Value]:
    return array_api.linalg.qr(x, mode=mode)


@quaxify
def slogdet(x: Value, /) -> tuple[Value, Value]:
    return array_api.linalg.slogdet(x)


@quaxify
def solve(x1: Value, x2: Value, /) -> Value:
    return array_api.linalg.solve(x1, x2)


@quaxify
def svd(x: Value, /, *, full_matrices: bool = True) -> tuple[Value, Value, Value]:
    return array_api.linalg.svd(x, full_matrices=full_matrices)


@quaxify
def svdvals(x: Value, /) -> Value:
    return array_api.linalg.svdvals(x)


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
def trace(x: Value, /, *, offset: int = 0, dtype: DType | None = None) -> Value:
    return jnp.trace(x, offset=offset, dtype=dtype)


@quaxify
def vecdot(x1: Value, x2: Value, /, *, axis: int | None = None) -> Value:
    return array_api.vecdot(x1, x2, axis=axis)


@quaxify
def vector_norm(
    x: Value,
    /,
    *,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    ord: int | float = 2,  # pylint: disable=redefined-builtin
) -> Value:
    return array_api.linalg.vector_norm(x, axis=axis, keepdims=keepdims, ord=ord)
