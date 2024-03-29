__all__ = ["unique_all", "unique_counts", "unique_inverse", "unique_values"]


from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def unique_all(x: ArrayLike, /) -> tuple[Value, Value, Value, Value]:
    return array_api.unique_all(x)


@quaxify
def unique_counts(x: ArrayLike, /) -> tuple[Value, Value]:
    return array_api.unique_counts(x)


@quaxify
def unique_inverse(x: ArrayLike, /) -> tuple[Value, Value]:
    return array_api.unique_inverse(x)


@quaxify
def unique_values(x: ArrayLike, /) -> Value:
    return array_api.unique_values(x)
