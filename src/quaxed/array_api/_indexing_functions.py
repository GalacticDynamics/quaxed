__all__ = ["take"]

from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def take(x: ArrayLike, indices: ArrayLike, /, *, axis: int | None = None) -> Value:
    return array_api.take(x, indices, axis=axis)
