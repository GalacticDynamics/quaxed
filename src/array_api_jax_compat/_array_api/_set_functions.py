__all__ = ["unique_all", "unique_counts", "unique_inverse", "unique_values"]


from jax.experimental import array_api
from quax import Value

from array_api_jax_compat._utils import quaxify


@quaxify
def unique_all(x: Value, /) -> tuple[Value, Value, Value, Value]:
    return array_api.unique_all(x)


@quaxify
def unique_counts(x: Value, /) -> tuple[Value, Value]:
    return array_api.unique_counts(x)


@quaxify
def unique_inverse(x: Value, /) -> tuple[Value, Value]:
    return array_api.unique_inverse(x)


@quaxify
def unique_values(x: Value, /) -> Value:
    return array_api.unique_values(x)
