"""Array API creation functions."""

__all__ = [
    "arange",
    "asarray",
    "empty_like",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones_like",
    "tril",
    "triu",
    "zeros_like",
]


from typing import Literal, TypeVar

import jax
import jax.numpy as jnp
from jaxtyping import ArrayLike
from plum import dispatch
from quax import Value

from quaxed._types import DType
from quaxed._utils import quaxify

T = TypeVar("T")

# =============================================================================


@dispatch
def arange(
    start: ArrayLike,
    stop: ArrayLike | None,
    step: ArrayLike | None,
    /,
    *,
    dtype: DType | None = None,
) -> ArrayLike:
    return jnp.arange(start, stop, step, dtype=dtype)


@dispatch  # type: ignore[no-redef]
def arange(
    start: ArrayLike,
    stop: ArrayLike | None,
    /,
    *,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
) -> ArrayLike:
    # dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype)


@dispatch  # type: ignore[no-redef]
def arange(
    start: ArrayLike,
    /,
    *,
    stop: ArrayLike | None = None,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
) -> ArrayLike:
    # dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype)


@dispatch  # type: ignore[no-redef]
def arange(
    *,
    start: ArrayLike,
    stop: ArrayLike | None = None,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
) -> ArrayLike | Value:
    # dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype)


# =============================================================================


@quaxify
def asarray(
    obj: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
    order: Literal["C", "F", "A", "K"] | None = None,
) -> Value:
    return jnp.asarray(obj, dtype=dtype, order=order)


# =============================================================================


@dispatch  # type: ignore[misc]
def empty_like(
    prototype: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
    shape: tuple[int, ...] | None = None,
) -> ArrayLike:
    return jnp.empty_like(prototype, dtype=dtype, shape=shape)


# =============================================================================


@dispatch
def full(
    shape: tuple[int, ...] | int,
    fill_value: ArrayLike,
    *,
    dtype: DType | None = None,
) -> ArrayLike:
    return jnp.full(shape, fill_value, dtype=dtype)


@dispatch  # type: ignore[no-redef]
def full(
    shape: tuple[int, ...] | int,
    *,
    fill_value: ArrayLike,
    dtype: DType | None = None,
) -> ArrayLike:
    return full(shape, fill_value, dtype=dtype)


# =============================================================================


@dispatch
def full_like(
    x: ArrayLike,
    /,
    fill_value: ArrayLike,
    *,
    dtype: DType | None = None,
    shape: tuple[int, ...] | None = None,
) -> ArrayLike:
    return jnp.full_like(x, fill_value, dtype=dtype, shape=shape)


@dispatch  # type: ignore[no-redef]
def full_like(
    x: ArrayLike,
    *,
    fill_value: ArrayLike,
    dtype: DType | None = None,
    shape: tuple[int, ...] | None = None,
) -> ArrayLike:
    # dispatch on both `x` and `fill_value`
    return full_like.invoke(type(x), type(fill_value))(
        x, fill_value, dtype=dtype, shape=shape
    )


# =============================================================================


@dispatch
def linspace(  # noqa: PLR0913
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
    /,
    *,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DType | None = None,
    axis: int = 0,
) -> jax.Array | jax.core.Tracer | Value:
    return jnp.linspace(
        start, stop, num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis
    )


@dispatch  # type: ignore[no-redef]
def linspace(  # noqa: PLR0913
    start: ArrayLike,
    stop: ArrayLike,
    /,
    *,
    num: int,
    endpoint: bool = True,
    retstep: bool = False,
    dtype: DType | None = None,
    axis: int = 0,
) -> jax.Array | jax.core.Tracer | Value:
    # dispatch on `start`, `stop`, and `num`
    return linspace(
        start, stop, num, endpoint=endpoint, retstep=retstep, dtype=dtype, axis=axis
    )


# =============================================================================


@quaxify
def meshgrid(
    *arrays: ArrayLike, copy: bool = True, sparse: bool = False, indexing: str = "xy"
) -> list[ArrayLike]:
    return jnp.meshgrid(*arrays, copy=copy, sparse=sparse, indexing=indexing)


# =============================================================================


@dispatch  # type: ignore[misc]
def ones_like(
    x: ArrayLike, /, *, dtype: DType | None = None, shape: tuple[int, ...] | None = None
) -> ArrayLike:
    return jnp.ones_like(x, dtype=dtype, shape=shape)


# =============================================================================


@quaxify
def tril(x: ArrayLike, /, *, k: int = 0) -> ArrayLike:
    return jnp.tril(x, k=k)


# =============================================================================


@quaxify
def triu(x: ArrayLike, /, *, k: int = 0) -> ArrayLike:
    return jnp.triu(x, k=k)


# =============================================================================


@dispatch  # type: ignore[misc]
def zeros_like(
    x: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
) -> ArrayLike | jax.Array:
    return jnp.zeros_like(x, dtype=dtype)
