"""Unary operations for array-ish objects."""

# fmt: off
__all__ = [
    "LaxUnaryMixin", "NumpyUnaryMixin",
    # ----------
    "LaxPosMixin", "NumpyPosMixin",  # __pos__
    "LaxNegMixin", "NumpyNegMixin",  # __neg__
    "NumpyInvertMixin",              # __invert__
    "LaxAbsMixin", "NumpyAbsMixin",  # __abs__
]
# fmt: on

from typing import Generic
from typing_extensions import TypeVar

import optype as opt

import quaxed.lax as qlax
import quaxed.numpy as qnp

T = TypeVar("T")
R = TypeVar("R", default=bool)

# -----------------------------------------------
# `__pos__`


class LaxPosMixin:
    """Mixin for ``__pos__`` method using quaxified `jax.lax.pos`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxPosMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> +x
    MyArray(value=i32[3])

    """  # noqa: E501

    def __pos__(self: opt.CanPosSelf) -> opt.CanPosSelf:
        return self  # TODO: more robust implementation


class NumpyPosMixin:
    """Mixin for ``__pos__`` method using quaxified `jax.numpy.pos`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyPosMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> +x
    MyArray(value=i32[3])

    """  # noqa: E501

    def __pos__(self) -> opt.CanPosSelf:
        return qnp.positive(self)


# -----------------------------------------------
# `__neg__`


class LaxNegMixin(Generic[R]):
    """Mixin for ``__neg__`` method using quaxified `jax.lax.neg`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxNegMixin[Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> -x
    Array([-1, -2, -3], dtype=int32)

    """  # noqa: E501

    def __neg__(self) -> R:
        return qlax.neg(self)


class NumpyNegMixin(Generic[R]):
    """Mixin for ``__neg__`` method using quaxified `jax.numpy.neg`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyNegMixin[Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> -x
    Array([-1, -2, -3], dtype=int32)

    """  # noqa: E501

    def __neg__(self) -> R:
        return qnp.negative(self)


# -----------------------------------------------
# `__invert__`


# TODO: LaxInvertMixin for jax.lax.invert does not exist yet


class NumpyInvertMixin(Generic[R]):
    """Mixin for ``__invert__`` method using quaxified `jax.numpy.invert`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyInvertMixin[Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> ~x
    Array([-2, -3, -4], dtype=int32)

    """  # noqa: E501

    def __invert__(self) -> R:
        return qnp.invert(self)


# -----------------------------------------------
# `__abs__`


class LaxAbsMixin(Generic[R]):
    """Mixin for ``__abs__`` method using quaxified `jax.lax.abs`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxAbsMixin[Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([-1, -2, -3]))
    >>> abs(x)
    Array([1, 2, 3], dtype=int32)

    """  # noqa: E501

    def __abs__(self) -> R:
        return qlax.abs(self)


class NumpyAbsMixin(Generic[R]):
    """Mixin for ``__abs__`` method using quaxified `jax.numpy.abs`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyAbsMixin[Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([-1, -2, -3]))
    >>> abs(x)
    Array([1, 2, 3], dtype=int32)

    """  # noqa: E501

    def __abs__(self) -> R:
        return qnp.abs(self)


# ===============================================
# Combined Mixins


class LaxUnaryMixin(LaxPosMixin, LaxNegMixin, LaxAbsMixin):
    """Combined mixin for unary operations using quaxified `jax.lax`."""


class NumpyUnaryMixin(NumpyPosMixin, NumpyNegMixin, NumpyAbsMixin, NumpyInvertMixin):
    """Combined mixin for unary operations using quaxified `jax.numpy`."""
