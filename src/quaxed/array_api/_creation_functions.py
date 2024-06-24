"""Array API creation functions."""

__all__ = [
    "arange",
    "asarray",
    "empty",
    "empty_like",
    "eye",
    "from_dlpack",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "ones",
    "ones_like",
    "tril",
    "triu",
    "zeros",
    "zeros_like",
]


from functools import partial
from typing import TypeVar

import jax
import jax.numpy as jnp
from jax import Device
from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._types import DType
from quaxed._utils import quaxify

from ._dispatch import dispatcher

T = TypeVar("T")

# =============================================================================


@dispatcher
def arange(
    start: ArrayLike,
    stop: ArrayLike | None,
    step: ArrayLike | None,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    return array_api.arange(start, stop, step, dtype=dtype, device=device)


@dispatcher  # type: ignore[no-redef]
def arange(
    start: ArrayLike,
    stop: ArrayLike | None,
    *,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    # re-dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype, device=device)


@dispatcher  # type: ignore[no-redef]
def arange(
    start: ArrayLike,
    *,
    stop: ArrayLike | None = None,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    # re- dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype, device=device)


@dispatcher  # type: ignore[no-redef]
def arange(
    *,
    start: ArrayLike,
    stop: ArrayLike | None = None,
    step: ArrayLike | None = None,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike | Value:
    # re-dispatch on `start`, `stop`, and `step`
    return arange(start, stop, step, dtype=dtype, device=device)


# =============================================================================


@partial(jax.jit, static_argnames=("dtype", "device", "copy"))
@quaxify
def asarray(
    obj: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Value:
    return array_api.asarray(obj, dtype=dtype, device=device, copy=copy)


# =============================================================================


def empty(
    shape: tuple[int, ...], *, dtype: DType | None = None, device: Device | None = None
) -> jax.Array:
    return array_api.empty(shape, dtype=dtype, device=device)


# =============================================================================


@dispatcher  # type: ignore[misc]
def empty_like(
    x: ArrayLike, /, *, dtype: DType | None = None, device: Device | None = None
) -> ArrayLike:
    return array_api.empty_like(x, dtype=dtype, device=device)


# =============================================================================


def eye(
    n_rows: int,
    n_cols: int | None = None,
    /,
    *,
    k: int = 0,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array:
    return array_api.eye(n_rows, n_cols, k=k, dtype=dtype, device=device)


# =============================================================================


def from_dlpack(x: object, /) -> jax.Array:
    return array_api.from_dlpack(x)


# =============================================================================


@dispatcher
def full(
    shape: tuple[int, ...] | int,
    fill_value: ArrayLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    return array_api.full(shape, fill_value, dtype=dtype, device=device)


@dispatcher  # type: ignore[no-redef]
def full(
    shape: tuple[int, ...] | int,
    *,
    fill_value: ArrayLike,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    return full(shape, fill_value, dtype=dtype, device=device)


# =============================================================================


@dispatcher
def full_like(
    x: ArrayLike,
    /,
    fill_value: ArrayLike,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    return array_api.full_like(x, fill_value, dtype=dtype, device=device)


@dispatcher  # type: ignore[no-redef]
def full_like(
    x: ArrayLike,
    *,
    fill_value: ArrayLike,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike:
    # dispatch on both `x` and `fill_value`
    return full_like.invoke(type(x), type(fill_value))(
        x, fill_value, dtype=dtype, device=device
    )


# =============================================================================


@dispatcher
def linspace(  # noqa: PLR0913
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> jax.Array | jax.core.Tracer | Value:
    return array_api.linspace(
        start,
        stop,
        num,
        dtype=dtype,
        device=device,
        endpoint=endpoint,
    )


@dispatcher  # type: ignore[no-redef]
def linspace(  # noqa: PLR0913
    start: ArrayLike,
    stop: ArrayLike,
    /,
    *,
    num: int,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> ArrayLike:
    # dispatch on `start`, `stop`, and `num`
    return linspace(start, stop, num, dtype=dtype, device=device, endpoint=endpoint)


# =============================================================================


@quaxify
def meshgrid(*arrays: ArrayLike, indexing: str = "xy") -> list[ArrayLike]:
    return jnp.meshgrid(*arrays, indexing=indexing)


# =============================================================================


def ones(
    shape: tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array:
    return array_api.ones(shape, dtype=dtype, device=device)


# =============================================================================


@dispatcher  # type: ignore[misc]
def ones_like(
    x: ArrayLike, /, *, dtype: DType | None = None, device: Device | None = None
) -> ArrayLike:
    return array_api.ones_like(x, dtype=dtype, device=device)


# =============================================================================


@quaxify
def tril(x: ArrayLike, /, *, k: int = 0) -> ArrayLike:
    return array_api.tril(x, k=k)


# =============================================================================


# @partial(jax.jit, static_argnames=("k",))
@quaxify
def triu(x: ArrayLike, /, *, k: int = 0) -> ArrayLike:
    return array_api.triu(x, k=k)


# =============================================================================


def zeros(
    shape: tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array:
    return array_api.zeros(shape, dtype=dtype, device=device)


# =============================================================================


# @partial(jax.jit, static_argnames=("dtype", "device"))
# @quaxify
@dispatcher  # type: ignore[misc]
def zeros_like(
    x: ArrayLike,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> ArrayLike | jax.Array:
    return array_api.zeros_like(x, dtype=dtype, device=device)
