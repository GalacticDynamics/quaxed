__all__ = [
    "broadcast_arrays",
    "broadcast_to",
    "concat",
    "expand_dims",
    "flip",
    "permute_dims",
    "reshape",
    "roll",
    "squeeze",
    "stack",
]

from jax.experimental import array_api
from quax import Value

from ._utils import quaxify


@quaxify
def broadcast_arrays(*arrays: Value) -> list[Value]:
    return array_api.broadcast_arrays(*arrays)


@quaxify
def broadcast_to(x: Value, /, shape: tuple[int, ...]) -> Value:
    return array_api.broadcast_to(x, shape)


@quaxify
def concat(
    arrays: tuple[Value, ...] | list[Value],
    /,
    *,
    axis: int | None = 0,
) -> Value:
    return array_api.concat(arrays, axis=axis)


@quaxify
def expand_dims(x: Value, /, *, axis: int = 0) -> Value:
    return array_api.expand_dims(x, axis=axis)


@quaxify
def flip(x: Value, /, *, axis: int | tuple[int, ...] | None = None) -> Value:
    return array_api.flip(x, axis=axis)


@quaxify
def permute_dims(x: Value, /, axes: tuple[int, ...]) -> Value:
    return array_api.permute_dims(x, axes=axes)


@quaxify
def reshape(x: Value, /, shape: tuple[int, ...], *, copy: bool | None = None) -> Value:
    return array_api.reshape(x, shape, copy=copy)


@quaxify
def roll(
    x: Value,
    /,
    shift: int | tuple[int],
    *,
    axis: int | tuple[int, ...] | None = None,
) -> Value:
    return array_api.roll(x, shift=shift, axis=axis)


@quaxify
def squeeze(x: Value, /, axis: int | tuple[int, ...]) -> Value:
    return array_api.squeeze(x, axis=axis)


@quaxify
def stack(arrays: tuple[Value, ...] | list[Value], /, *, axis: int = 0) -> Value:
    return array_api.stack(arrays, axis=axis)
