"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

array-api-jax-compat: Array-API JAX compatibility
"""

from __future__ import annotations

__all__ = ["grad", "value_and_grad"]

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax._src.api import AxisName


def grad(
    fun: Callable[..., Any],
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Any]:
    """`grad`."""
    raise NotImplementedError("TODO")  # noqa: EM101


def value_and_grad(
    fun: Callable[..., Any],
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., tuple[Any, Any]]:
    """`value_and_grad`."""
    raise NotImplementedError("TODO")  # noqa: EM101
