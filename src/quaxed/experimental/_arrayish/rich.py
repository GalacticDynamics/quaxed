"""Rich operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxComparisonMixin", "NumpyComparisonMixin",  # rich comparison
    # ----------
    "LaxEqMixin", "NumpyEqMixin",  # `__eq__`
    "LaxNeMixin", "NumpyNeMixin",  # `__ne__`
    "LaxLtMixin", "NumpyLtMixin",  # `__lt__`
    "LaxLeMixin", "NumpyLeMixin",  # `__le__`
    "LaxGtMixin", "NumpyGtMixin",  # `__gt__`
    "LaxGeMixin", "NumpyGeMixin",  # `__ge__`
]
# fmt: on


from typing import Generic
from typing_extensions import TypeVar, override

import quaxed.lax as qlax
import quaxed.numpy as qnp

T = TypeVar("T")
Rbool = TypeVar("Rbool", default=bool)

# -----------------------------------------------
# `__eq__`


class LaxEqMixin(Generic[T, Rbool]):
    """Mixin for ``__eq__`` method using quaxified `jax.lax.eq`.

    !!! warning
        Equinox PyTree provides an `__eq__` method that cannot be overridden by
        subclassing in this way. To ensure correct behavior, you need to
        explicitly assign the `__eq__` method in your subclass.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxEqMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     __eq__ = LaxEqMixin.__eq__
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x == x
    Array([ True,  True,  True], dtype=bool)

    >>> x == 1
    Array([ True, False, False], dtype=bool)

    """  # noqa: E501

    @override
    def __eq__(self, other: T) -> Rbool:  # type: ignore[override]
        return qlax.eq(self, other)


class NumpyEqMixin(Generic[T, Rbool]):
    """Mixin for ``__eq__`` method using quaxified `jax.numpy.eq`.

    !!! warning
        Equinox PyTree provides an `__eq__` method that cannot be overridden by
        subclassing in this way. To ensure correct behavior, you need to
        explicitly assign the `__eq__` method in your subclass.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyEqMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     __eq__ = NumpyEqMixin.__eq__
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x == x
    Array([ True,  True,  True], dtype=bool)

    >>> x == 1
    Array([ True, False, False], dtype=bool)

    """  # noqa: E501

    @override
    def __eq__(self, other: T) -> Rbool:  # type: ignore[override]
        return qnp.equal(self, other)


# -----------------------------------------------
# `__ne__`


class LaxNeMixin(Generic[T, Rbool]):
    """Mixin for ``__ne__`` method using quaxified `jax.lax.ne`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxNeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x != x
    Array([False, False, False], dtype=bool)

    >>> x != 1
    Array([False,  True,  True], dtype=bool)

    """  # noqa: E501

    @override
    def __ne__(self, other: T) -> Rbool:  # type: ignore[override]
        return qlax.ne(self, other)


class NumpyNeMixin(Generic[T, Rbool]):
    """Mixin for ``__ne__`` method using quaxified `jax.numpy.ne`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyNeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x != x
    Array([False, False, False], dtype=bool)

    >>> x != 1
    Array([False,  True,  True], dtype=bool)

    """  # noqa: E501

    @override
    def __ne__(self, other: T) -> Rbool:  # type: ignore[override]
        return qnp.not_equal(self, other)


# -----------------------------------------------
# `__lt__`


class LaxLtMixin(Generic[T, Rbool]):
    """Mixin for ``__lt__`` method using quaxified `jax.lax.lt`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxLtMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x < 2
    Array([ True, False, False], dtype=bool)

    """  # noqa: E501

    def __lt__(self, other: T) -> Rbool:
        return qlax.lt(self, other)


class NumpyLtMixin(Generic[T, Rbool]):
    """Mixin for ``__lt__`` method using quaxified `jax.numpy.lt`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyLtMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x < 2
    Array([ True, False, False], dtype=bool)

    """  # noqa: E501

    def __lt__(self, other: T) -> Rbool:
        return qnp.less(self, other)


# -----------------------------------------------
# `__le__`


class LaxLeMixin(Generic[T, Rbool]):
    """Mixin for ``__le__`` method using quaxified `jax.lax.le`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxLeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x <= 2
    Array([ True,  True, False], dtype=bool)

    """  # noqa: E501

    def __le__(self, other: T) -> Rbool:
        return qlax.le(self, other)


class NumpyLeMixin(Generic[T, Rbool]):
    """Mixin for ``__le__`` method using quaxified `jax.numpy.le`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyLeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x <= 2
    Array([ True,  True, False], dtype=bool)

    """  # noqa: E501

    def __le__(self, other: T) -> Rbool:
        return qnp.less_equal(self, other)


# -----------------------------------------------
# `__gt__`


class LaxGtMixin(Generic[T, Rbool]):
    """Mixin for ``__gt__`` method using quaxified `jax.lax.gt`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxGtMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x > 2
    Array([False, False,  True], dtype=bool)

    """  # noqa: E501

    def __gt__(self, other: T) -> Rbool:
        return qlax.gt(self, other)


class NumpyGtMixin(Generic[T, Rbool]):
    """Mixin for ``__gt__`` method using quaxified `jax.numpy.gt`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyGtMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x > 2
    Array([False, False,  True], dtype=bool)

    """  # noqa: E501

    def __gt__(self, other: T) -> Rbool:
        return qnp.greater(self, other)


# -----------------------------------------------
# `__ge__`


class LaxGeMixin(Generic[T, Rbool]):
    """Mixin for ``__ge__`` method using quaxified `jax.lax.ge`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxGeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x >= 2
    Array([False,  True,  True], dtype=bool)

    """  # noqa: E501

    def __ge__(self, other: T) -> Rbool:
        return qlax.ge(self, other)


class NumpyGeMixin(Generic[T, Rbool]):
    """Mixin for ``__ge__`` method using quaxified `jax.numpy.ge`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyGeMixin[Any, Bool[Array, "..."]]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x >= 2
    Array([False,  True,  True], dtype=bool)

    """  # noqa: E501

    def __ge__(self, other: T) -> Rbool:
        return qnp.greater_equal(self, other)


# ================================================


class LaxComparisonMixin(
    LaxEqMixin[T, Rbool],
    LaxNeMixin[T, Rbool],
    LaxLtMixin[T, Rbool],
    LaxLeMixin[T, Rbool],
    LaxGtMixin[T, Rbool],
    LaxGeMixin[T, Rbool],
):
    pass


class NumpyComparisonMixin(
    NumpyEqMixin[T, Rbool],
    NumpyNeMixin[T, Rbool],
    NumpyLtMixin[T, Rbool],
    NumpyLeMixin[T, Rbool],
    NumpyGtMixin[T, Rbool],
    NumpyGeMixin[T, Rbool],
):
    pass
