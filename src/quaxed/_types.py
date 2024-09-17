"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved."""

__all__: list[str] = []

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp
import plum
from jaxtyping import ArrayLike

# Simplify the display of ArrayLike
plum.activate_union_aliases()
plum.set_union_alias(ArrayLike, "ArrayLike")


@runtime_checkable
class DType(Protocol):
    """The dtype of an array."""

    dtype: jnp.dtype[Any]
