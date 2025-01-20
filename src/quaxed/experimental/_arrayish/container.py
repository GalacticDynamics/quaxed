"""Container operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxLenMixin", "NumpyLenMixin",  # __len__
    "LaxLengthHintMixin", "NumpyLengthHintMixin",  # __length_hint__
]
# fmt: on

from typing import Protocol, TypeVar, runtime_checkable

import quaxed.numpy as qnp

T = TypeVar("T")


@runtime_checkable
class HasShape(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the shape of the object."""
        ...


# -----------------------------------------------
# `__len__`


class LaxLenMixin:
    """Mixin for ``__len__`` method using quaxified `jax.lax.len`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxLenMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    """  # noqa: E501

    def __len__(self: HasShape) -> int:
        return self.shape[0]


class NumpyLenMixin:
    """Mixin for ``__len__`` method using quaxified `jax.numpy.len`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyLenMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    """  # noqa: E501

    def __len__(self) -> int:
        return qnp.shape(self)[0]


# -----------------------------------------------
# `__length_hint__`


class LaxLengthHintMixin:
    """Mixin for ``__length_hint__`` method using quaxified `jax.lax.length_hint`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxLengthHintMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    """  # noqa: E501

    def __length_hint__(self: HasShape) -> int:
        return self.shape[0]


class NumpyLengthHintMixin:
    """Mixin for ``__length_hint__`` method using quaxified `jax.numpy.length_hint`.

    Examples
    --------
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyLengthHintMixin):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    """  # noqa: E501

    def __length_hint__(self) -> int:
        return qnp.shape(self)[0]
