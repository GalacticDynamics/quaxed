"""Quaxed :mod:`jax.numpy`."""

__all__ = ["vectorize"]

import functools
from collections.abc import Callable, Collection
from typing import Any, TypeVar

import jax
from jax._src.numpy.vectorize import (
    _apply_excluded,
    _check_output_dims,
    _parse_gufunc_signature,
    _parse_input_dimensions,
)

from ._core import asarray, expand_dims, squeeze

T = TypeVar("T")


def vectorize(  # noqa: C901
    pyfunc: Callable[..., Any],
    *,
    excluded: Collection[int | str] = frozenset(),
    signature: str | None = None,
) -> Callable[..., Any]:
    """Define a vectorized function with broadcasting.

    This is a copy-paste from :func:`jax.numpy.vectorize`, but the internals are
    all replaced with their :mod:`quaxed` counterparts to allow quax-friendly
    objects to pass through. The only thing that isn't quaxed is `jax.vmap`,
    which allows any array-like object to pass through without converting it.
    Note that this behaviour is DIFFERENT than doing ``quaxify(jnp.vectorize)``
    since `quaxify` makes objects look like arrays, not their actual type,
    which can be problematic. This function passes through the objects
    unchanged (so long as they are amenable to the reshapes and ``vamap``).

    Arguments:
    ---------
    pyfunc: callable
        function to vectorize.
    excluded: Collection[int | str], optional
        optional set of integers representing positional arguments for which the
        function will not be vectorized. These will be passed directly to
        ``pyfunc`` unmodified.
    signature: str | None
        optional generalized universal function signature, e.g.,
        ``(m,n),(n)->(m)`` for vectorized matrix-vector multiplication. If
        provided, ``pyfunc`` will be called with (and expected to return) arrays
        with shapes given by the size of corresponding core dimensions. By
        default, pyfunc is assumed to take scalars arrays as input and output.

    Returns
    -------
    callable
        Vectorized version of the given function.

    Here are a few examples of how one could write vectorized linear algebra
    routines using :func:`vectorize`:

    >>> from functools import partial

    >>> @partial(jnp.vectorize, signature='(k),(k)->(k)')
    ... def cross_product(a, b):
    ...   assert a.shape == b.shape and a.ndim == b.ndim == 1
    ...   return jnp.array([a[1] * b[2] - a[2] * b[1],
    ...                     a[2] * b[0] - a[0] * b[2],
    ...                     a[0] * b[1] - a[1] * b[0]])

    >>> @partial(jnp.vectorize, signature='(n,m),(m)->(n)')
    ... def matrix_vector_product(matrix, vector):
    ...   assert matrix.ndim == 2 and matrix.shape[1:] == vector.shape
    ...   return matrix @ vector

    These functions are only written to handle 1D or 2D arrays (the ``assert``
    statements will never be violated), but with vectorize they support
    arbitrary dimensional inputs with NumPy style broadcasting, e.g.,

    >>> cross_product(jnp.ones(3), jnp.ones(3)).shape
    (3,)
    >>> cross_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2, 3)
    >>> cross_product(jnp.ones((1, 2, 3)), jnp.ones((2, 1, 3))).shape
    (2, 2, 3)
    >>> matrix_vector_product(jnp.ones(3), jnp.ones(3))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    ValueError: input with shape (3,) does not have enough dimensions for all
    core dimensions ('n', 'k') on vectorized function with excluded=frozenset()
    and signature='(n,k),(k)->(k)'
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones(3)).shape
    (2,)
    >>> matrix_vector_product(jnp.ones((2, 3)), jnp.ones((4, 3))).shape
    (4, 2)

    Note that this has different semantics than `jnp.matmul`:

    >>> jnp.matmul(jnp.ones((2, 3)), jnp.ones((4, 3)))  # doctest: +IGNORE_EXCEPTION_DETAIL
    Traceback (most recent call last):
    TypeError: dot_general requires contracting dimensions to have the same shape, got [3] and [4].
    """  # noqa: E501
    if any(not isinstance(exclude, str | int) for exclude in excluded):
        msg = (
            "jax.numpy.vectorize can only exclude integer or string arguments, "
            f"but excluded={excluded!r}"
        )
        raise TypeError(msg)

    if any(isinstance(e, int) and e < 0 for e in excluded):
        msg = f"excluded={excluded!r} contains negative numbers"
        raise ValueError(msg)

    @functools.wraps(pyfunc)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        error_context = (
            f"on vectorized function with excluded={excluded!r} and "
            f"signature={signature!r}"
        )
        excluded_func, args, kwargs = _apply_excluded(pyfunc, excluded, args, kwargs)

        if signature is not None:
            input_core_dims, output_core_dims = _parse_gufunc_signature(signature)
        else:
            input_core_dims = [()] * len(args)
            output_core_dims = None

        none_args = {i for i, arg in enumerate(args) if arg is None}
        if any(none_args):
            if any(input_core_dims[i] != () for i in none_args):
                msg = f"Cannot pass None at locations {none_args} with {signature=}"
                raise ValueError(msg)
            excluded_func, args, _ = _apply_excluded(excluded_func, none_args, args, {})
            input_core_dims = [
                dim for i, dim in enumerate(input_core_dims) if i not in none_args
            ]

        args = tuple(map(asarray, args))

        broadcast_shape, dim_sizes = _parse_input_dimensions(
            args, input_core_dims, error_context
        )

        checked_func = _check_output_dims(
            excluded_func, dim_sizes, output_core_dims, error_context
        )

        # Rather than broadcasting all arguments to full broadcast shapes, prefer
        # expanding dimensions using vmap. By pushing broadcasting
        # into vmap, we can make use of more efficient batching rules for
        # primitives where only some arguments are batched (e.g., for
        # lax_linalg.triangular_solve), and avoid instantiating large broadcasted
        # arrays.

        squeezed_args = []
        rev_filled_shapes = []

        for arg, core_dims in zip(args, input_core_dims, strict=True):
            noncore_shape = arg.shape[: arg.ndim - len(core_dims)]

            pad_ndim = len(broadcast_shape) - len(noncore_shape)
            filled_shape = pad_ndim * (1,) + noncore_shape
            rev_filled_shapes.append(filled_shape[::-1])

            squeeze_indices = tuple(
                i for i, size in enumerate(noncore_shape) if size == 1
            )
            squeezed_arg = squeeze(arg, axis=squeeze_indices)
            squeezed_args.append(squeezed_arg)

        vectorized_func = checked_func
        dims_to_expand = []
        for negdim, axis_sizes in enumerate(zip(*rev_filled_shapes, strict=True)):
            in_axes = tuple(None if size == 1 else 0 for size in axis_sizes)
            if all(axis is None for axis in in_axes):
                dims_to_expand.append(len(broadcast_shape) - 1 - negdim)
            else:
                vectorized_func = jax.vmap(vectorized_func, in_axes)
        result = vectorized_func(*squeezed_args)

        if not dims_to_expand:
            out = result
        elif isinstance(result, tuple):
            dims_to_expand_ = tuple(dims_to_expand)
            out = tuple(expand_dims(r, axis=dims_to_expand_) for r in result)
        else:
            out = expand_dims(result, axis=tuple(dims_to_expand))

        return out

    return wrapped
