"""Quaxed :mod:`jax`."""

__all__ = [
    "device_put",
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "value_and_grad",
]

from collections.abc import Callable, Hashable
from typing import Any, TypeAlias

import jax
from quax import quaxify

AxisName: TypeAlias = Hashable


# =============================================================================

device_put = quaxify(jax.device_put)


# -----------------------------------------------------------------------------


def grad(
    fun: Callable[..., Any],
    *,
    filter_spec: Any = True,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.grad`.

    This adds the additional parameter ``filter_spec``, which is passed to
    :func:`quax.quaxify`, to support passing integers when ``allow_int`` is
    `True`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from quaxed import grad

    >>> grad(jnp.square)(jnp.array(3.))
    Array(6., dtype=float32, weak_type=True)

    """
    return quaxify(jax.grad(fun, **kwargs), filter_spec=filter_spec)


# -----------------------------------------------------------------------------


def hessian(
    fun: Callable[..., Any],
    *,
    filter_spec: Any = True,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.hessian`.

    This adds the additional parameter ``filter_spec``, which is passed to
    :func:`quax.quaxify`, to support passing integers when ``allow_int`` is
    `True`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from quaxed import hessian

    >>> hessian(jnp.square)(jnp.array(3.))
    Array(2., dtype=float32, weak_type=True)

    """
    return quaxify(jax.hessian(fun, **kwargs), filter_spec=filter_spec)


# -----------------------------------------------------------------------------


def jacfwd(
    fun: Callable[..., Any],
    *,
    filter_spec: Any = True,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.jacfwd`.

    This adds the additional parameter ``filter_spec``, which is passed to
    :func:`quax.quaxify`, to support passing integers when ``allow_int`` is
    `True`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from quaxed import jacfwd

    >>> jacfwd(jnp.square)(jnp.array(3.))
    Array(6., dtype=float32, weak_type=True)

    """
    return quaxify(jax.jacfwd(fun, **kwargs), filter_spec=filter_spec)


def jacrev(
    fun: Callable[..., Any],
    *,
    filter_spec: Any = True,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.jacfwd`.

    This adds the additional parameter ``filter_spec``, which is passed to
    :func:`quax.quaxify`, to support passing integers when ``allow_int`` is
    `True`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from quaxed import jacrev

    >>> jacrev(jnp.square)(jnp.array(3.))
    Array(6., dtype=float32, weak_type=True)

    """
    return quaxify(jax.jacrev(fun, **kwargs), filter_spec=filter_spec)


# -----------------------------------------------------------------------------


def value_and_grad(
    fun: Callable[..., Any],
    *,
    filter_spec: Any = True,
    **kwargs: Any,
) -> Callable[..., Any]:
    """Quaxed version of :func:`jax.value_and_grad`.

    This adds the additional parameter ``filter_spec``, which is passed to
    :func:`quax.quaxify`, to support passing integers when ``allow_int`` is
    `True`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from quaxed import value_and_grad

    >>> value_and_grad(jnp.square)(jnp.array(3.))
    (Array(9., dtype=float32, weak_type=True),
     Array(6., dtype=float32, weak_type=True))

    """
    return quaxify(jax.value_and_grad(fun, **kwargs), filter_spec=filter_spec)
