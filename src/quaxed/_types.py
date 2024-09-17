"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Pre-quaxed libraries for multiple dispatch over abstract array types in JAX
"""

__all__: list[str] = []

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class DType(Protocol):
    """The dtype of an array."""

    dtype: jnp.dtype[Any]
