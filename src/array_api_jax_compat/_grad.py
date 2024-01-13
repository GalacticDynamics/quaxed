"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

array-api-jax-compat: Array-API JAX compatibility
"""

from __future__ import annotations

__all__ = ["grad"]

from typing import TYPE_CHECKING, Any, Protocol, TypeVar

import jax
import jax.numpy as jnp
from quax import quaxify
from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from jax._src.api import AxisName


class SupportsGetItem(Protocol):
    def __getitem__(self, key: Any) -> Self:
        ...


T = TypeVar("T")
IT = TypeVar("IT", bound=SupportsGetItem)


def grad(
    fun: Callable[..., Any],
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    reduce_axes: Sequence[AxisName] = (),
    vmap_kw: dict[str, Any] | None = None,
    vmap_batch: tuple[int, ...] | None = None,
) -> Callable[..., Any]:
    """Quaxified :func:`jax.grad`.

    Creates a function that evaluates the gradient of ``fun``.

    Parameters
    ----------
    fun : callable
        Function to be differentiated. Its arguments at positions specified by
        `argnums` should be arrays, scalars, or standard Python containers.
        Argument arrays in the positions specified by `argnums` must be of
        inexact (i.e., floating-point or complex) type. It should return a
        scalar (which includes arrays with shape `()` but not arrays with shape
        `(1,)` etc.)
    argnums : int or sequence of ints, optional
        Specifies which positional argument(s) to differentiate with respect to
        (default 0).
    has_aux : bool, optional
        Indicates whether `fun` returns a pair where the first element is
        considered the output of the mathematical function to be differentiated
        and the second element is auxiliary data. Default False.
    holomorphic : bool, optional
        Indicates whether `fun` is promised to be holomorphic. If True, inputs
        and outputs must be complex. Default False.
    allow_int : bool, optional
        Whether to allow differentiating with respect to integer valued inputs.
        The gradient of an integer input will have a trivial vector-space dtype
        (float0).  Default False.
    reduce_axes : tuple of axis names, optional
        If an axis is listed here, and `fun` implicitly broadcasts a value over
        that axis, the backward pass will perform a `psum` of the corresponding
        gradient.  Otherwise, the gradient will be per-example over named axes.
        For example, if `'batch'` is a named batch axis, `grad(f,
        reduce_axes=('batch',))` will create a function that computes the total
        gradient while `grad(f)` will create one that computes the per-example
        gradient.

    Returns
    -------
    callable
        A function with the same arguments as `fun`, that evaluates the gradient
        of `fun`.  If `argnums` is an integer then the gradient has the same
        shape and type as the positional argument indicated by that integer. If
        `argnums` is a tuple of integers, the gradient is a tuple of values with
        the same shapes and types as the corresponding arguments. If `has_aux`
        is True then a pair of (gradient, auxiliary_data) is returned.

    Examples
    --------
    >>> import jax
    >>>
    >>> grad_tanh = jax.grad(jax.numpy.tanh)
    >>> print(grad_tanh(0.2))
    0.961043
    """
    # TODO: get this working using the actual `grad` function.
    # There are some interesting issues to resolve. See
    # https://github.com/patrick-kidger/quax/issues/5.
    # In the meantime, we workaround this by using `jacfwd` instead.
    if allow_int:
        msg = "allow_int is not yet supported"
        raise NotImplementedError(msg)
    if reduce_axes:
        msg = "reduce_axes is not yet supported"
        raise NotImplementedError(msg)

    grad_substitute = quaxify(
        jax.jacfwd(fun, argnums=argnums, has_aux=has_aux, holomorphic=holomorphic),
    )

    def grad_func(*args: Any) -> Any:
        for i, arg in enumerate(args):
            assert (  # noqa: S101
                len(jnp.shape(arg)) < 2
            ), f"arg {i} has shape {arg.shape}"

        return grad_substitute(*args)

    return grad_func
