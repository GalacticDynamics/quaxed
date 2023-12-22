__all__ = [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "moveaxis",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
    "tile",
    "unstack",
]

import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def broadcast_arrays(*arrays: Value) -> list[Value]:
    return jnp.broadcast_arrays(*arrays)


@quaxify
def broadcast_to(x: Value, /, shape: tuple[int, ...]) -> Value:
    return jnp.broadcast_to(x, shape)


@quaxify
def concat(
    arrays: tuple[Value, ...] | list[Value],
    /,
    *,
    axis: int | None = 0,
) -> Value:
    return jnp.concatenate(arrays, axis=axis)


@quaxify
def expand_dims(x: Value, /, *, axis: int = 0) -> Value:
    return jnp.expand_dims(x, axis=axis)


@quaxify
def flip(x: Value, /, *, axis: int | tuple[int, ...] | None = None) -> Value:
    return jnp.flip(x, axis=axis)


@quaxify
def moveaxis(
    x: Value,
    source: int | tuple[int, ...],
    destination: int | tuple[int, ...],
    /,
) -> Value:
    return jnp.moveaxis(x, source, destination)


@quaxify
def permute_dims(x: Value, /, axes: tuple[int, ...]) -> Value:
    return jnp.transpose(x, axes)


@quaxify
def reshape(x: Value, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Value:
    return jnp.reshape(x, shape, order="C" if copy else "K")


@quaxify
def roll(
    x: Value,
    /,
    shift: int | tuple[int, ...],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Value:
    return jnp.roll(x, shift, axis=axis)


@quaxify
def squeeze(x: Value, /, axis: int | tuple[int, ...]) -> Value:
    return jnp.squeeze(x, axis=axis)


@quaxify
def stack(arrays: tuple[Value, ...] | list[Value], /, *, axis: int = 0) -> Value:
    return jnp.stack(arrays, axis=axis)


@quaxify
def tile(x: Value, repetitions: tuple[int, ...], /) -> Value:
    return jnp.tile(x, repetitions)


@quaxify
def unstack(
    x: Value,  # TODO: support  # pylint: disable=unused-argument
    /,
    *,
    axis: int = 0,  # TODO: support  # pylint: disable=unused-argument
) -> tuple[Value, ...]:
    msg = "not yet supported."
    raise NotImplementedError(msg)
    # return jnp.split(x, axis=axis)
