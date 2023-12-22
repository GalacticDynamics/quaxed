"""Array API creation functions."""

__all__ = [
    # "arange",
    "asarray",
    # "empty",
    "empty_like",
    # "eye",
    # "from_dlpack",
    # "full",
    "full_like",
    # "linspace",
    "meshgrid",
    # "ones",
    "ones_like",
    "tril",
    "triu",
    # "zeros",
    "zeros_like",
]


from functools import partial
from typing import Any, TypeVar

import jax
import jax.numpy as jnp
from jax import Device
from quax import Value

from ._dispatch import dispatcher
from ._types import DType, NestedSequence, SupportsBufferProtocol
from ._utils import quaxify

T = TypeVar("T")

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
    copy: bool | None = None,  # TODO: support  # pylint: disable=unused-argument
) -> Value:
    out = jnp.asarray(obj, dtype=dtype)
    return jax.device_put(out, device=device)
    # TODO: jax.lax.cond is not yet supported by Quax.
    # out = jax.lax.cond(bool(copy), lambda x: jax.lax.copy_p.bind(x), lambda x: x, out)


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
    out = jnp.empty_like(x, dtype=dtype)
    return jax.device_put(out, device=device)


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
    out = jnp.full_like(x, fill_value, dtype=dtype)
    return jax.device_put(out, device=device)


# =============================================================================


@quaxify
def meshgrid(*arrays: Value, indexing: str = "xy") -> list[Value]:
    return jnp.meshgrid(*arrays, indexing=indexing)


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
    out = jnp.ones_like(x, dtype=dtype)
    return jax.device_put(out, device=device)


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
    out = jnp.zeros_like(x, dtype=dtype)
    return jax.device_put(out, device=device)


# @dispatcher
# def zeros_like(
#     x: quax.zero.Zero, /, *, dtype: DType | None = None, device: Device | None = None
# ) -> jnp.ndarray:
#     out = jnp.zeros_like(x, dtype=dtype)
#     out = jax.device_put(out, device=device)
#     return out
