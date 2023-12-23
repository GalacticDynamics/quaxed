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
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import Device
from jax.experimental import array_api
from quax import Value

from ._dispatch import dispatcher
from ._types import DType, NestedSequence, SupportsBufferProtocol
from ._utils import quaxify

T = TypeVar("T")

# =============================================================================


@quaxify  # TODO: probably need to dispatch this instead
def arange(
    start: Value,
    /,
    stop: Value | None = None,
    step: Value = 1,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Value:
    return array_api.arange(start, stop, step, dtype=dtype, device=device)


# =============================================================================


@partial(jax.jit, static_argnames=("dtype", "device", "copy"))
@quaxify
def asarray(
    obj: Value
    | bool
    | int
    | float
    | complex
    | NestedSequence[Any]
    | SupportsBufferProtocol,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    copy: bool | None = None,
) -> Value:
    return array_api.asarray(obj, dtype=dtype, device=device, copy=copy)


# =============================================================================


def empty(
    shape: tuple[int, ...],
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array:
    return array_api.empty(shape, dtype=dtype, device=device)


# =============================================================================


# @partial(jax.jit, static_argnames=("dtype", "device"))
# @quaxify  # TODO: quaxify won't work here because of how the function is defined.
@dispatcher  # type: ignore[misc]
def empty_like(
    x: jax.Array | jax.core.Tracer | Value,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array | jax.core.Tracer | Value:
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


@dispatcher  # type: ignore[misc]
def full(
    shape: tuple[int, ...],
    fill_value: int | float | complex | bool | jax.Array,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array | Value:
    return array_api.full(shape, fill_value, dtype=dtype, device=device)


# =============================================================================


# @partial(jax.jit, static_argnames=("dtype", "device"))
# @quaxify  # TODO: quaxify won't work here because of how the function is defined.
@dispatcher  # type: ignore[misc]
def full_like(
    x: jax.Array | jax.core.Tracer | Value,
    /,
    fill_value: bool | int | float | complex,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array | jax.core.Tracer | Value:
    return array_api.full_like(x, fill_value, dtype=dtype, device=device)


# =============================================================================


@dispatcher  # type: ignore[misc]
def linspace(  # noqa: PLR0913
    start: int | float | complex | jax.Array,
    stop: int | float | complex | jax.Array,
    /,
    num: int,
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


# =============================================================================


@quaxify
def meshgrid(*arrays: Value, indexing: str = "xy") -> list[Value]:
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


# @partial(jax.jit, static_argnames=("dtype", "device"))
# @quaxify  # TODO: quaxify won't work here because of how the function is defined.
@dispatcher  # type: ignore[misc]
def ones_like(
    x: jax.Array | jax.core.Tracer | Value,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> jax.Array | jax.core.Tracer | Value:
    return array_api.ones_like(x, dtype=dtype, device=device)


# =============================================================================


# @partial(jax.jit, static_argnames=("k",))
@quaxify
def tril(x: Value, /, *, k: int = 0) -> Value:
    return jnp.tril(x, k=k)


# =============================================================================


# @partial(jax.jit, static_argnames=("k",))
@quaxify
def triu(x: Value, /, *, k: int = 0) -> Value:
    return jnp.triu(x, k=k)


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
    x: jax.Array | jax.core.Tracer | Value,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> Value | jax.core.Tracer | jax.Array:
    return array_api.zeros_like(x, dtype=dtype, device=device)
