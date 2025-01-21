"""Container operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxLenMixin", "NumpyLenMixin",  # __len__
    "LaxLengthHintMixin", "NumpyLengthHintMixin",  # __length_hint__
]
# fmt: on

from typing import Protocol, runtime_checkable

import quaxed.numpy as qnp


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
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxLenMixin

    >>> class Val(AbstractVal, LaxLenMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    >>> x = Val(jnp.array(1))
    >>> len(x)
    0

    """

    def __len__(self: HasShape) -> int:
        return self.shape[0] if self.shape else 0


class NumpyLenMixin:
    """Mixin for ``__len__`` method using quaxified `jax.numpy.len`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyLenMixin

    >>> class Val(AbstractVal, NumpyLenMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> len(x)
    3

    >>> x = Val(jnp.array(1))
    >>> len(x)
    0

    """

    def __len__(self) -> int:
        shape = qnp.shape(self)
        return shape[0] if shape else 0


# -----------------------------------------------
# `__length_hint__`


class LaxLengthHintMixin:
    """Mixin for ``__length_hint__`` method using quaxified `jax.lax.length_hint`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxLengthHintMixin

    >>> class Val(AbstractVal, LaxLengthHintMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    >>> x = Val(jnp.array(0))
    >>> x.__length_hint__()
    0

    """

    def __length_hint__(self: HasShape) -> int:
        return self.shape[0] if self.shape else 0


class NumpyLengthHintMixin:
    """Mixin for ``__length_hint__`` method using quaxified `jax.numpy.length_hint`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyLengthHintMixin

    >>> class Val(AbstractVal, NumpyLengthHintMixin):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x.__length_hint__()
    3

    >>> x = Val(jnp.array(1))
    >>> x.__length_hint__()
    0

    """

    def __length_hint__(self) -> int:
        shape = qnp.shape(self)
        return shape[0] if shape else 0
