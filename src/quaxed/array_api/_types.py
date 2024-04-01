"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Array-API JAX compatibility
"""

__all__: list[str] = []

from typing import Any, Protocol, TypeVar, runtime_checkable

import jax.numpy as jnp


@runtime_checkable
class DType(Protocol):
    """The dtype of an array."""

    dtype: jnp.dtype[Any]


@runtime_checkable  # TODO: need actual implementation
class SupportsBufferProtocol(Protocol):
    """Supports the buffer protocol."""


_T_co = TypeVar("_T_co", covariant=True)


@runtime_checkable
class NestedSequence(Protocol[_T_co]):
    """A nested sequence."""

    def __getitem__(self, key: int, /) -> "_T_co | NestedSequence[_T_co]": ...

    def __len__(self, /) -> int: ...
