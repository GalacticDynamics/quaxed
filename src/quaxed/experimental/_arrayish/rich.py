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
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxEqMixin

    >>> class Val(AbstractVal, LaxEqMixin[object, Bool[Array, "..."]]):
    ...     v: Array
    ...     __eq__ = LaxEqMixin.__eq__  # NOTE: this is necessary

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x == x
    Array([ True,  True,  True], dtype=bool)

    >>> x == 1
    Array([ True, False, False], dtype=bool)

    """

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
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyEqMixin

    >>> class Val(AbstractVal, NumpyEqMixin[object, Bool[Array, "..."]]):
    ...     v: Array
    ...     __eq__ = NumpyEqMixin.__eq__  # NOTE: this is necessary

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x == x
    Array([ True,  True,  True], dtype=bool)

    >>> x == 1
    Array([ True, False, False], dtype=bool)

    """

    @override
    def __eq__(self, other: T) -> Rbool:  # type: ignore[override]
        return qnp.equal(self, other)


# -----------------------------------------------
# `__ne__`


class LaxNeMixin(Generic[T, Rbool]):
    """Mixin for ``__ne__`` method using quaxified `jax.lax.ne`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxNeMixin

    >>> class Val(AbstractVal, LaxNeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x != x
    Array([False, False, False], dtype=bool)

    >>> x != 1
    Array([False,  True,  True], dtype=bool)

    """

    @override
    def __ne__(self, other: T) -> Rbool:  # type: ignore[override]
        return qlax.ne(self, other)


class NumpyNeMixin(Generic[T, Rbool]):
    """Mixin for ``__ne__`` method using quaxified `jax.numpy.ne`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyNeMixin

    >>> class Val(AbstractVal, NumpyNeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x != x
    Array([False, False, False], dtype=bool)

    >>> x != 1
    Array([False,  True,  True], dtype=bool)

    """

    @override
    def __ne__(self, other: T) -> Rbool:  # type: ignore[override]
        return qnp.not_equal(self, other)


# -----------------------------------------------
# `__lt__`


class LaxLtMixin(Generic[T, Rbool]):
    """Mixin for ``__lt__`` method using quaxified `jax.lax.lt`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxLtMixin

    >>> class Val(AbstractVal, LaxLtMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x < 2
    Array([ True, False, False], dtype=bool)

    """

    def __lt__(self, other: T) -> Rbool:
        return qlax.lt(self, other)


class NumpyLtMixin(Generic[T, Rbool]):
    """Mixin for ``__lt__`` method using quaxified `jax.numpy.lt`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyLtMixin

    >>> class Val(AbstractVal, NumpyLtMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x < 2
    Array([ True, False, False], dtype=bool)

    """

    def __lt__(self, other: T) -> Rbool:
        return qnp.less(self, other)


# -----------------------------------------------
# `__le__`


class LaxLeMixin(Generic[T, Rbool]):
    """Mixin for ``__le__`` method using quaxified `jax.lax.le`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxLeMixin

    >>> class Val(AbstractVal, LaxLeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x <= 2
    Array([ True,  True, False], dtype=bool)

    """

    def __le__(self, other: T) -> Rbool:
        return qlax.le(self, other)


class NumpyLeMixin(Generic[T, Rbool]):
    """Mixin for ``__le__`` method using quaxified `jax.numpy.le`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyLeMixin

    >>> class Val(AbstractVal, NumpyLeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x <= 2
    Array([ True,  True, False], dtype=bool)

    """

    def __le__(self, other: T) -> Rbool:
        return qnp.less_equal(self, other)


# -----------------------------------------------
# `__gt__`


class LaxGtMixin(Generic[T, Rbool]):
    """Mixin for ``__gt__`` method using quaxified `jax.lax.gt`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxGtMixin

    >>> class Val(AbstractVal, LaxGtMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x > 2
    Array([False, False,  True], dtype=bool)

    """

    def __gt__(self, other: T) -> Rbool:
        return qlax.gt(self, other)


class NumpyGtMixin(Generic[T, Rbool]):
    """Mixin for ``__gt__`` method using quaxified `jax.numpy.gt`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyGtMixin

    >>> class Val(AbstractVal, NumpyGtMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x > 2
    Array([False, False,  True], dtype=bool)

    """

    def __gt__(self, other: T) -> Rbool:
        return qnp.greater(self, other)


# -----------------------------------------------
# `__ge__`


class LaxGeMixin(Generic[T, Rbool]):
    """Mixin for ``__ge__`` method using quaxified `jax.lax.ge`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxGeMixin

    >>> class Val(AbstractVal, LaxGeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x >= 2
    Array([False,  True,  True], dtype=bool)

    """

    def __ge__(self, other: T) -> Rbool:
        return qlax.ge(self, other)


class NumpyGeMixin(Generic[T, Rbool]):
    """Mixin for ``__ge__`` method using quaxified `jax.numpy.ge`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyGeMixin

    >>> class Val(AbstractVal, NumpyGeMixin[object, Bool[Array, "..."]]):
    ...     v: Array

    >>> x = Val(jnp.array([1, 2, 3]))
    >>> x >= 2
    Array([False,  True,  True], dtype=bool)

    """

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
