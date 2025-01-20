"""Copy operations for Array-ish objects."""

__all__ = [
    "NumpyCopyMixin",  # __copy__
    "NumpyDeepCopyMixin",  # __deepcopy__
]

from typing import Any, Generic
from typing_extensions import TypeVar

import optype as opt

import quaxed.numpy as qnp

RCopy = TypeVar("RCopy", default=opt.copy.CanCopySelf)
RDeepcopy = TypeVar("RDeepcopy", default=opt.copy.CanDeepcopySelf)

# -----------------------------------------------
# `__copy__`


class NumpyCopyMixin(Generic[RCopy]):
    """Mixin for ``__copy__`` method using quaxified `jax.numpy.copy`.

    Examples
    --------
    >>> import copy
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyCopyMixin[Any]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> copy.copy(x)
    Array([1, 2, 3], dtype=int32)

    """  # noqa: E501

    def __copy__(self) -> RCopy:
        return qnp.copy(self)


# -----------------------------------------------
# `__deepcopy__`


class NumpyDeepCopyMixin(Generic[RDeepcopy]):
    """Mixin for ``__deepcopy__`` method using quaxified `jax.numpy.copy`.

    Examples
    --------
    >>> import copy
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyDeepCopyMixin[Any]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> copy.deepcopy(x)
    Array([1, 2, 3], dtype=int32)

    """  # noqa: E501

    def __deepcopy__(self, memo: dict[Any, Any], /) -> RDeepcopy:
        return qnp.copy(self)
