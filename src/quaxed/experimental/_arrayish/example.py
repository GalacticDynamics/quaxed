"""Arrayish."""

__all__: list[str] = ["AbstractVal"]


import equinox as eqx
import jax
from jaxtyping import Array
from quax import ArrayValue


class AbstractVal(ArrayValue):  # type: ignore[misc]
    """ABC for example arrayish object."""

    #: The array.
    v: eqx.AbstractVar[Array]

    def aval(self) -> jax.core.ShapedArray:
        return jax.core.ShapedArray(self.v.shape, self.v.dtype)

    def materialise(self) -> Array:
        return self.v
