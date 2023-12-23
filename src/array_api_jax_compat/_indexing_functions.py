__all__ = ["take"]

from jax.experimental import array_api
from quax import Value


def take(x: Value, indices: Value, /, *, axis: int | None = None) -> Value:
    return array_api.take(x, indices, axis=axis)
