"""Rounding operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxRoundMixin", "NumpyRoundMixin",  # __round__
    "LaxTruncMixin", "NumpyTruncMixin",  # __trunc__
    "LaxFloorMixin", "NumpyFloorMixin",  # __floor__
    "LaxCeilMixin", "NumpyCeilMixin",  # __ceil__
]
# fmt: on

from functools import partial
from typing import Generic, Literal
from typing_extensions import TypeVar

import jax
from jaxtyping import Array, Bool, PyTree, Shaped

import quaxed.lax as qlax
import quaxed.numpy as qnp

from .rich import LaxGeMixin

T = TypeVar("T")
R = TypeVar("R", default=bool)

# -----------------------------------------------
# `__round__`


class LaxRoundMixin(Generic[R]):
    """Mixin for ``__round__`` method using quaxified `jax.lax.round`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxRoundMixin

    >>> class Val(AbstractVal, LaxRoundMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> round(x)
    Array([1., 3., 4.], dtype=float32)

    """

    _ROUNDING_METHOD: Literal[qlax.RoundingMethod] = qlax.RoundingMethod.AWAY_FROM_ZERO  # type: ignore[valid-type]

    def __round__(self) -> R:
        return qlax.round(self, self._ROUNDING_METHOD)


class NumpyRoundMixin(Generic[R]):
    """Mixin for ``__round__`` method using quaxified `jax.numpy.round`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyRoundMixin

    >>> class Val(AbstractVal, NumpyRoundMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> round(x)
    Array([1., 2., 4.], dtype=float32)

    """

    @partial(jax.jit, static_argnums=1)
    def __round__(self, ndigits: int = 0) -> R:
        return qnp.round(self, decimals=ndigits)


# -----------------------------------------------
# `__trunc__`


class LaxTruncMixin(
    LaxGeMixin[PyTree[Shaped[Array, "..."]], Bool[Array, "..."]], Generic[R]
):
    """Mixin for ``__trunc__`` method using quaxified `jax.lax.floor`, `jax.lax.ceil`.

    Uses quaxified `jax.lax.select` to apply `qlax.floor` for positive values
    and `qlax.ceil` for negative values.

    Examples
    --------
    >>> import math
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxTruncMixin

    >>> class Val(AbstractVal, LaxTruncMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7, -1.2, -2.5, -3.7]))
    >>> math.trunc(x)
    Array([ 1.,  2.,  3., -1., -2., -3.], dtype=float32)

    """

    def __trunc__(self) -> R:
        return qlax.select(
            qlax.ge(self, qlax.full_like(self, 0)),
            qlax.floor(self),
            qlax.ceil(self),
        )


class NumpyTruncMixin(Generic[R]):
    """Mixin for ``__trunc__`` method using quaxified `jax.numpy.trunc`.

    Examples
    --------
    >>> import math
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyTruncMixin

    >>> class Val(AbstractVal, NumpyTruncMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> math.trunc(x)
    Array([1., 2., 3.], dtype=float32)

    """

    def __trunc__(self) -> R:
        return qnp.trunc(self)


# -----------------------------------------------
# `__floor__`


class LaxFloorMixin(Generic[R]):
    """Mixin for ``__floor__`` method using quaxified `jax.lax.floor`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxFloorMixin

    >>> class Val(AbstractVal, LaxFloorMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> x.__floor__()
    Array([1., 2., 3.], dtype=float32)

    """

    def __floor__(self) -> R:
        return qlax.floor(self)


class NumpyFloorMixin(Generic[R]):
    """Mixin for ``__floor__`` method using quaxified `jax.numpy.floor`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyFloorMixin

    >>> class Val(AbstractVal, NumpyFloorMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> x.__floor__()
    Array([1., 2., 3.], dtype=float32)

    """

    def __floor__(self) -> R:
        return qnp.floor(self)


# -----------------------------------------------
# `__ceil__`


class LaxCeilMixin(Generic[R]):
    """Mixin for ``__ceil__`` method using quaxified `jax.lax.ceil`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, LaxCeilMixin

    >>> class Val(AbstractVal, LaxCeilMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> x.__ceil__()
    Array([2., 3., 4.], dtype=float32)

    """

    def __ceil__(self) -> R:
        return qlax.ceil(self)


class NumpyCeilMixin(Generic[R]):
    """Mixin for ``__ceil__`` method using quaxified `jax.numpy.ceil`.

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quaxed.experimental.arrayish import AbstractVal, NumpyCeilMixin

    >>> class Val(AbstractVal, NumpyCeilMixin[Array]):
    ...     v: Array

    >>> x = Val(jnp.array([1.2, 2.5, 3.7]))
    >>> x.__ceil__()
    Array([2., 3., 4.], dtype=float32)

    """

    def __ceil__(self) -> R:
        return qnp.ceil(self)
