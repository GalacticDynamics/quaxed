"""Quaxed :mod:`jax`."""

__all__ = [
    "device_put",
    "grad",
    "jacfwd",
]

from collections.abc import Callable, Hashable, Sequence
from typing import Any, TypeAlias

import jax
from quax import quaxify

AxisName: TypeAlias = Hashable


# =============================================================================

device_put = quaxify(jax.device_put)


def grad(  # noqa: PLR0913
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.grad`."""
    return quaxify(
        jax.grad(
            fun,
            argnums,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            reduce_axes=reduce_axes,
        )
    )


def hessian(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.hessian`."""
    return quaxify(jax.hessian(fun, argnums, holomorphic=holomorphic, has_aux=has_aux))


def jacfwd(
    fun: Callable[..., Any],
    argnums: int | Sequence[int] = 0,
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.jacfwd`."""
    return quaxify(jax.jacfwd(fun, argnums, holomorphic=holomorphic, has_aux=has_aux))
