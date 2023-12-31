__all__ = ["take"]

from jax.experimental import array_api
from quax import Value

from ._utils import quaxify


@quaxify
def take(x: Value, indices: Value, /, *, axis: int | None = None) -> Value:
    return array_api.take(x, indices, axis=axis)
