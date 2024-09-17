"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Array-API JAX compatibility
"""

__all__: list[str] = []

from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class DType(Protocol):
    """The dtype of an array."""

    dtype: jnp.dtype[Any]
