__all__ = ["take"]

import jax.numpy as jnp
from quax import Value


def take(x: Value, indices: Value, /, *, axis: int | None = None) -> Value:
    return jnp.take(x, indices, axis=axis)
