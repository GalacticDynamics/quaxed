__all__ = ["unique_all", "unique_counts", "unique_inverse", "unique_values"]


import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def unique_all(x: Value, /) -> tuple[Value, Value, Value, Value]:
    return jnp.unique(x, return_counts=True, return_index=True, return_inverse=True)


@quaxify
def unique_counts(x: Value, /) -> tuple[Value, Value]:
    return jnp.unique(x, return_counts=True)


@quaxify
def unique_inverse(x: Value, /) -> tuple[Value, Value]:
    return jnp.unique(x, return_inverse=True)


@quaxify
def unique_values(x: Value, /) -> Value:
    return jnp.unique(x)
