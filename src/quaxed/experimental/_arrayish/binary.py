"""Binary operations for Array-ish objects."""

# fmt: off
__all__ = [
    "LaxBinaryOpsMixin", "NumpyBinaryOpsMixin",
    # ========== Float Operations ==========
    "LaxMathMixin", "NumpyMathMixin",
    # ----- add -----
    "LaxBothAddMixin", "NumpyBothAddMixin",
    "LaxAddMixin", "NumpyAddMixin",  # __add__
    "LaxRAddMixin", "NumpyRAddMixin",  # __radd__
    # ----- sub -----
    "LaxBothSubMixin", "NumpyBothSubMixin",
    "LaxSubMixin", "NumpySubMixin",  # __sub__
    "LaxRSubMixin", "NumpyRSubMixin",  # __rsub__
    # ----- mul -----
    "LaxBothMulMixin", "NumpyBothMulMixin",
    "LaxMulMixin", "NumpyMulMixin",  # __mul__
    "LaxRMulMixin", "NumpyRMulMixin",  # __rmul__
    # ---- matmul -----
    "LaxMatMulMixin", "NumpyMatMulMixin",  # __matmul__
    # "LaxRMatMulMixin", "NumpyRMatMulMixin",  # __rmatmul__
    # ----- truediv -----
    "LaxBothTrueDivMixin", "NumpyBothTrueDivMixin",
    "LaxTrueDivMixin", "NumpyTrueDivMixin",  # __truediv__
    "LaxRTrueDivMixin", "NumpyRTrueDivMixin",  # __rtruediv__
    # ----- floordiv -----
    "LaxBothFloorDivMixin", "NumpyBothFloorDivMixin",
    "LaxFloorDivMixin", "NumpyFloorDivMixin",  # __floordiv__
    "LaxRFloorDivMixin", "NumpyRFloorDivMixin",  # __rfloordiv__
    # ----- mod -----
    "LaxBothModMixin", "NumpyBothModMixin",
    "LaxModMixin", "NumpyModMixin",  # __mod__
    "LaxRModMixin", "NumpyRModMixin",  # __rmod__
    # ----- divmod -----
    "NumpyBothDivModMixin",
    # "LaxDivModMixin",
    "NumpyDivModMixin",
    # "LaxRDivModMixin",
    "NumpyRDivModMixin",
    # ----- pow -----
    "LaxBothPowMixin", "NumpyBothPowMixin",
    "LaxPowMixin", "NumpyPowMixin",  # __pow__
    "LaxRPowMixin", "NumpyRPowMixin",  # __rpow__
    # ========== Bitwise Operations ==========
    "LaxBitwiseMixin", "NumpyBitwiseMixin",
    # ----- lshift -----
    "LaxBothLShiftMixin", "NumpyBothLShiftMixin",
    "LaxLShiftMixin", "NumpyLShiftMixin",  # __lshift__
    "LaxRLShiftMixin", "NumpyRLShiftMixin",  # __rlshift__
    # ----- rshift -----
    "LaxBothRShiftMixin", "NumpyBothRShiftMixin",
    "LaxRShiftMixin", "NumpyRShiftMixin",  # __rshift__
    "LaxRRShiftMixin", "NumpyRRShiftMixin",  # __rrshift__
    # ----- and -----
    "LaxBothAndMixin", "NumpyBothAndMixin",
    "LaxAndMixin", "NumpyAndMixin",  # __and__
    "LaxRAndMixin", "NumpyRAndMixin",  # __rand__
    # ----- xor -----
    "LaxBothXorMixin", "NumpyBothXorMixin",
    "LaxXorMixin", "NumpyXorMixin",  # __xor__
    "LaxRXorMixin", "NumpyRXorMixin",  # __rxor__
    # ----- or -----
    "LaxBothOrMixin", "NumpyBothOrMixin",
    "LaxOrMixin", "NumpyOrMixin",  # __or__
    "LaxROrMixin", "NumpyROrMixin", # __ror__
]
# fmt: on

from typing import Generic, Literal
from typing_extensions import TypeVar

import quaxed.lax as qlax
import quaxed.numpy as qnp

T = TypeVar("T")
R = TypeVar("R", default=bool)


# ===============================================
# Add

# -------------------------------------
# `__add__`


class LaxAddMixin(Generic[T, R]):
    """Mixin for ``__add__`` method using quaxified `jax.lax.add`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxAddMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x + x
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __add__(self, other: T) -> R:
        return qlax.add(self, other)


class NumpyAddMixin(Generic[T, R]):
    """Mixin for ``__add__`` method using quaxified `jax.numpy.add`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyAddMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x + x
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __add__(self, other: T) -> R:
        return qnp.add(self, other)


# -------------------------------------
# `__radd__`


class LaxRAddMixin(Generic[T, R]):
    """Mixin for ``__radd__`` method using quaxified `jax.lax.add`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRAddMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 + x
    Array([2, 3, 4], dtype=int32)

    """  # noqa: E501

    def __radd__(self, other: T) -> R:
        return qlax.add(other, self)


class NumpyRAddMixin(Generic[T, R]):
    """Mixin for ``__radd__`` method using quaxified `jax.numpy.add`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRAddMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 + x
    Array([2, 3, 4], dtype=int32)

    """  # noqa: E501

    def __radd__(self, other: T) -> R:
        return qnp.add(other, self)


# -------------------------------------


class LaxBothAddMixin(LaxAddMixin[T, R], LaxRAddMixin[T, R]):
    pass


class NumpyBothAddMixin(NumpyAddMixin[T, R], NumpyRAddMixin[T, R]):
    pass


# ===============================================
# Subtraction

# -------------------------------------
# `__sub__`


class LaxSubMixin(Generic[T, R]):
    """Mixin for ``__sub__`` method using quaxified `jax.lax.sub`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxSubMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x - x
    Array([0, 0, 0], dtype=int32)

    """  # noqa: E501

    def __sub__(self, other: T) -> R:
        return qlax.sub(self, other)


class NumpySubMixin(Generic[T, R]):
    """Mixin for ``__sub__`` method using quaxified `jax.numpy.sub`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpySubMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x - x
    Array([0, 0, 0], dtype=int32)

    """  # noqa: E501

    def __sub__(self, other: T) -> R:
        return qnp.subtract(self, other)


# -------------------------------------
# `__rsub__`


class LaxRSubMixin(Generic[T, R]):
    """Mixin for ``__rsub__`` method using quaxified `jax.lax.sub`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRSubMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 - x
    Array([ 0, -1, -2], dtype=int32)

    """  # noqa: E501

    def __rsub__(self, other: T) -> R:
        return qlax.sub(other, self)


class NumpyRSubMixin(Generic[T, R]):
    """Mixin for ``__rsub__`` method using quaxified `jax.numpy.sub`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array, Bool
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRSubMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 - x
    Array([ 0, -1, -2], dtype=int32)

    """  # noqa: E501

    def __rsub__(self, other: T) -> R:
        return qnp.subtract(other, self)


# -------------------------------------


class LaxBothSubMixin(LaxSubMixin[T, R], LaxRSubMixin[T, R]):
    pass


class NumpyBothSubMixin(NumpySubMixin[T, R], NumpyRSubMixin[T, R]):
    pass


# ===============================================

# -------------------------------------
# `__mul__`


class LaxMulMixin(Generic[T, R]):
    """Mixin for ``__mul__`` method using quaxified `jax.lax.mul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x * x
    Array([1, 4, 9], dtype=int32)

    """  # noqa: E501

    def __mul__(self, other: T) -> R:
        return qlax.mul(self, other)


class NumpyMulMixin(Generic[T, R]):
    """Mixin for ``__mul__`` method using quaxified `jax.numpy.mul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x * x
    Array([1, 4, 9], dtype=int32)

    """  # noqa: E501

    def __mul__(self, other: T) -> R:
        return qnp.multiply(self, other)


# -------------------------------------
# `__rmul__`


class LaxRMulMixin(Generic[T, R]):
    """Mixin for ``__rmul__`` method using quaxified `jax.lax.mul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 * x
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __rmul__(self, other: T) -> R:
        return qlax.mul(other, self)


class NumpyRMulMixin(Generic[T, R]):
    """Mixin for ``__rmul__`` method using quaxified `jax.numpy.mul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 * x
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __rmul__(self, other: T) -> R:
        return qnp.multiply(other, self)


# -------------------------------------


class LaxBothMulMixin(LaxMulMixin[T, R], LaxRMulMixin[T, R]):
    pass


class NumpyBothMulMixin(NumpyMulMixin[T, R], NumpyRMulMixin[T, R]):
    pass


# ===============================================

# -------------------------------------
# `__matmul__`


class LaxMatMulMixin(Generic[T, R]):
    """Mixin for ``__matmul__`` method using quaxified `jax.lax.matmul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxMatMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([[1, 2], [3, 4]]))
    >>> y = MyArray(jnp.array([[5, 6], [7, 8]]))
    >>> x @ y
    Array([[19, 22],
           [43, 50]], dtype=int32)

    """  # noqa: E501

    def __matmul__(self, other: T) -> R:
        return qlax.dot(self, other)  # TODO: is this the right operator?


class NumpyMatMulMixin(Generic[T, R]):
    """Mixin for ``__matmul__`` method using quaxified `jax.numpy.matmul`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyMatMulMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([[1, 2], [3, 4]]))
    >>> y = MyArray(jnp.array([[5, 6], [7, 8]]))
    >>> x @ y
    Array([[19, 22],
           [43, 50]], dtype=int32)

    """  # noqa: E501

    def __matmul__(self, other: T) -> R:
        return qnp.matmul(self, other)


# ===============================================
# Float Division

# -------------------------------------
# `__truediv__`


class LaxTrueDivMixin(Generic[T, R]):
    """Mixin for ``__truediv__`` method using quaxified `jax.lax.truediv`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxTrueDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x / 2  # Note: integer division
    Array([0, 1, 1], dtype=int32)

    """  # noqa: E501

    def __truediv__(self, other: T) -> R:
        return qlax.div(self, other)


class NumpyTrueDivMixin(Generic[T, R]):
    """Mixin for ``__truediv__`` method using quaxified `jax.numpy.true_divide`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyTrueDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x / 2
    Array([0.5, 1. , 1.5], dtype=float32)

    """  # noqa: E501

    def __truediv__(self, other: T) -> R:
        return qnp.true_divide(self, other)


# -------------------------------------
# `__rtruediv__`


class LaxRTrueDivMixin(Generic[T, R]):
    """Mixin for ``__rtruediv__`` method using quaxified `jax.lax.truediv`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRTrueDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 / x
    Array([2, 1, 0], dtype=int32)

    """  # noqa: E501

    def __rtruediv__(self, other: T) -> R:
        return qlax.div(other, self)


class NumpyRTrueDivMixin(Generic[T, R]):
    """Mixin for ``__rtruediv__`` method using quaxified `jax.numpy.true_divide`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRTrueDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 / x
    Array([2. , 1. , 0.6666667], dtype=float32)

    """  # noqa: E501

    def __rtruediv__(self, other: T) -> R:
        return qnp.true_divide(other, self)


# -------------------------------------


class LaxBothTrueDivMixin(LaxTrueDivMixin[T, R], LaxRTrueDivMixin[T, R]):
    pass


class NumpyBothTrueDivMixin(NumpyTrueDivMixin[T, R], NumpyRTrueDivMixin[T, R]):
    pass


# ===============================================
# Integer Division

# -------------------------------------
# Floor division


class LaxFloorDivMixin(Generic[T, R]):
    """Mixin for ``__floordiv__`` method using quaxified `jax.lax`.

    Note that lax does not have a floor division function, so this is
    ``lax.floor(lax.div(x, y))``.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxFloorDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1., 2, 3]))
    >>> x // 2.
    Array([0., 1., 1.], dtype=float32)

    """  # noqa: E501

    def __floordiv__(self, other: T) -> R:
        return qlax.floor(qlax.div(self, other))


class NumpyFloorDivMixin(Generic[T, R]):
    """Mixin for ``__floordiv__`` method using quaxified `jax.numpy.floor_divide`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyFloorDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x // 2
    Array([0, 1, 1], dtype=int32)

    """  # noqa: E501

    def __floordiv__(self, other: T) -> R:
        return qnp.floor_divide(self, other)


# -------------------------------------
# `__rfloordiv__`


class LaxRFloorDivMixin(Generic[T, R]):
    """Mixin for ``__rfloordiv__`` method using quaxified `jax.lax`.

    Note that lax does not have a floor division function, so this is
    ``lax.floor(lax.div(x, y))``.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRFloorDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1., 2, 3]))
    >>> 2. // x
    Array([2., 1., 0.], dtype=float32)

    """  # noqa: E501

    def __rfloordiv__(self, other: T) -> R:
        return qlax.floor(qlax.div(other, self))


class NumpyRFloorDivMixin(Generic[T, R]):
    """Mixin for ``__rfloordiv__`` method using quaxified `jax.numpy.floor_divide`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRFloorDivMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 // x
    Array([2, 1, 0], dtype=int32)

    """  # noqa: E501

    def __rfloordiv__(self, other: T) -> R:
        return qnp.floor_divide(other, self)


# -------------------------------------


class LaxBothFloorDivMixin(LaxFloorDivMixin[T, R], LaxRFloorDivMixin[T, R]):
    pass


class NumpyBothFloorDivMixin(NumpyFloorDivMixin[T, R], NumpyRFloorDivMixin[T, R]):
    pass


# ===============================================
# Modulus

# -------------------------------------
# `__mod__`


class LaxModMixin(Generic[T, R]):
    """Mixin for ``__mod__`` method using quaxified `jax.lax.rem`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x % 2
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __mod__(self, other: T) -> R:
        return qlax.rem(self, other)


class NumpyModMixin(Generic[T, R]):
    """Mixin for ``__mod__`` method using quaxified `jax.numpy.mod`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x % 2
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __mod__(self, other: T) -> R:
        return qnp.mod(self, other)


# -------------------------------------
# `__rmod__`


class LaxRModMixin(Generic[T, R]):
    """Mixin for ``__rmod__`` method using quaxified `jax.lax.rem`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 % x
    Array([0, 0, 2], dtype=int32)

    """  # noqa: E501

    def __rmod__(self, other: T) -> R:
        return qlax.rem(other, self)


class NumpyRModMixin(Generic[T, R]):
    """Mixin for ``__rmod__`` method using quaxified `jax.numpy.mod`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 % x
    Array([0, 0, 2], dtype=int32)

    """  # noqa: E501

    def __rmod__(self, other: T) -> R:
        return qnp.mod(other, self)


# -------------------------------------


class LaxBothModMixin(LaxModMixin[T, R], LaxRModMixin[T, R]):
    pass


class NumpyBothModMixin(NumpyModMixin[T, R], NumpyRModMixin[T, R]):
    pass


# ===============================================
# Divmod

# -------------------------------------
# `__divmod__`


# TODO: a jax.lax.divmod equivalent?


class NumpyDivModMixin(Generic[T, R]):
    """Mixin for ``__divmod__`` method using quaxified `jax.numpy.divmod`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyDivModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([5, 7, 9]))
    >>> divmod(x, 2)
    (Array([2, 3, 4], dtype=int32), Array([1, 1, 1], dtype=int32))

    """  # noqa: E501

    def __divmod__(self, other: T) -> R:
        return qnp.divmod(self, other)


# -------------------------------------
# `__rdivmod__`


# TODO: a jax.lax.divmod equivalent?


class NumpyRDivModMixin(Generic[T, R]):
    """Mixin for ``__rdivmod__`` method using quaxified `jax.numpy.divmod`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRDivModMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([5, 7, 9]))
    >>> divmod(20, x)
    (Array([4, 2, 2], dtype=int32), Array([0, 6, 2], dtype=int32))

    """  # noqa: E501

    def __rdivmod__(self, other: T) -> R:
        return qnp.divmod(other, self)


# -------------------------------------


class NumpyBothDivModMixin(NumpyDivModMixin[T, R], NumpyRDivModMixin[T, R]):
    pass


# ===============================================
# Power

# -------------------------------------
# `__pow__`


class LaxPowMixin(Generic[T, R]):
    """Mixin for ``__pow__`` method using quaxified `jax.lax.pow`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxPowMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1., 2, 3]))  # must be floating
    >>> x ** 2
    Array([1., 4., 9.], dtype=float32)

    """  # noqa: E501

    def __pow__(self, other: T) -> R:
        return qlax.pow(self, other)


class NumpyPowMixin(Generic[T, R]):
    """Mixin for ``__pow__`` method using quaxified `jax.numpy.pow`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyPowMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x ** 2
    Array([1, 4, 9], dtype=int32)

    """  # noqa: E501

    def __pow__(self, other: T) -> R:
        return qnp.power(self, other)


# -------------------------------------
# `__rpow__`


class LaxRPowMixin(Generic[T, R]):
    """Mixin for ``__rpow__`` method using quaxified `jax.lax.pow`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRPowMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2. ** x
    Array([2., 4., 8.], dtype=float32)

    """  # noqa: E501

    def __rpow__(self, other: T) -> R:
        return qlax.pow(other, self)


class NumpyRPowMixin(Generic[T, R]):
    """Mixin for ``__rpow__`` method using quaxified `jax.numpy.pow`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRPowMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 2 ** x
    Array([2, 4, 8], dtype=int32)

    """  # noqa: E501

    def __rpow__(self, other: T) -> R:
        return qnp.power(other, self)


# -------------------------------------


class LaxBothPowMixin(LaxPowMixin[T, R], LaxRPowMixin[T, R]):
    pass


class NumpyBothPowMixin(NumpyPowMixin[T, R], NumpyRPowMixin[T, R]):
    pass


# ===============================================
# Left Shift

# -------------------------------------
# `__lshift__`


class LaxLShiftMixin(Generic[T, R]):
    """Mixin for ``__lshift__`` method using quaxified `jax.lax.lshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxLShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x << 1
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __lshift__(self, other: T) -> R:
        return qlax.shift_left(self, other)


class NumpyLShiftMixin(Generic[T, R]):
    """Mixin for ``__lshift__`` method using quaxified `jax.numpy.lshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyLShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x << 1
    Array([2, 4, 6], dtype=int32)

    """  # noqa: E501

    def __lshift__(self, other: T) -> R:
        return qnp.left_shift(self, other)


# -------------------------------------
# `__rlshift__`


class LaxRLShiftMixin(Generic[T, R]):
    """Mixin for ``__rlshift__`` method using quaxified `jax.lax.lshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRLShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 << x
    Array([2, 4, 8], dtype=int32)

    """  # noqa: E501

    def __rlshift__(self, other: T) -> R:
        return qlax.shift_left(other, self)


class NumpyRLShiftMixin(Generic[T, R]):
    """Mixin for ``__rlshift__`` method using quaxified `jax.numpy.lshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRLShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 << x
    Array([2, 4, 8], dtype=int32)

    """  # noqa: E501

    def __rlshift__(self, other: T) -> R:
        return qnp.left_shift(other, self)


# -------------------------------------


class LaxBothLShiftMixin(LaxLShiftMixin[T, R], LaxRLShiftMixin[T, R]):
    pass


class NumpyBothLShiftMixin(NumpyLShiftMixin[T, R], NumpyRLShiftMixin[T, R]):
    pass


# ===============================================
# Right Shift

# -------------------------------------
# `__rshift__`


class LaxRShiftMixin(Generic[T, R]):
    """Mixin for ``__rshift__`` method using quaxified `jax.lax.shift_right`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([2, 4, 8]))
    >>> x >> 1
    Array([1, 2, 4], dtype=int32)

    """  # noqa: E501

    _RIGHT_SHIFT_LOGICAL: Literal[True, False] = True

    def __rshift__(self, other: T) -> R:
        return qlax.cond(
            self._RIGHT_SHIFT_LOGICAL,
            qlax.shift_right_logical,
            qlax.shift_right_arithmetic,
            self,
            other,
        )


class NumpyRShiftMixin(Generic[T, R]):
    """Mixin for ``__rshift__`` method using quaxified `jax.numpy.rshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([2, 4, 8]))
    >>> x >> 1
    Array([1, 2, 4], dtype=int32)

    """  # noqa: E501

    def __rshift__(self, other: T) -> R:
        return qnp.right_shift(self, other)


# -------------------------------------
# `__rrshift__`


class LaxRRShiftMixin(Generic[T, R]):
    """Mixin for ``__rrshift__`` method using quaxified `jax.lax.rshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRRShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([2, 4, 8]))
    >>> 16 >> x
    Array([4, 1, 0], dtype=int32)

    """  # noqa: E501

    _RIGHT_SHIFT_LOGICAL: Literal[True, False] = True

    def __rrshift__(self, other: T) -> R:
        return qlax.cond(
            self._RIGHT_SHIFT_LOGICAL,
            qlax.shift_right_logical,
            qlax.shift_right_arithmetic,
            other,
            self,
        )


class NumpyRRShiftMixin(Generic[T, R]):
    """Mixin for ``__rrshift__`` method using quaxified `jax.numpy.rshift`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRRShiftMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([2, 4, 8]))
    >>> 16 >> x
    Array([4, 1, 0], dtype=int32)

    """  # noqa: E501

    def __rrshift__(self, other: T) -> R:
        return qnp.right_shift(other, self)


# -------------------------------------


class LaxBothRShiftMixin(LaxRShiftMixin[T, R], LaxRRShiftMixin[T, R]):
    pass


class NumpyBothRShiftMixin(NumpyRShiftMixin[T, R], NumpyRRShiftMixin[T, R]):
    pass


# ===============================================
# And

# -------------------------------------
# `__and__`


class LaxAndMixin(Generic[T, R]):
    """Mixin for ``__and__`` method using quaxified `jax.lax.and_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxAndMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x & 1
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __and__(self, other: T) -> R:
        return qlax.bitwise_and(self, other)


class NumpyAndMixin(Generic[T, R]):
    """Mixin for ``__and__`` method using quaxified `jax.numpy.and_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyAndMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x & 1
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __and__(self, other: T) -> R:
        return qnp.bitwise_and(self, other)


# -------------------------------------
# `__rand__`


class LaxRAndMixin(Generic[T, R]):
    """Mixin for ``__rand__`` method using quaxified `jax.lax.and_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRAndMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 & x
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __rand__(self, other: T) -> R:
        return qlax.bitwise_and(other, self)


class NumpyRAndMixin(Generic[T, R]):
    """Mixin for ``__rand__`` method using quaxified `jax.numpy.and_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRAndMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 & x
    Array([1, 0, 1], dtype=int32)

    """  # noqa: E501

    def __rand__(self, other: T) -> R:
        return qnp.bitwise_and(other, self)


# -------------------------------------


class LaxBothAndMixin(LaxAndMixin[T, R], LaxRAndMixin[T, R]):
    pass


class NumpyBothAndMixin(NumpyAndMixin[T, R], NumpyRAndMixin[T, R]):
    pass


# ===============================================
# Xor


# -------------------------------------
# `__xor__`


class LaxXorMixin(Generic[T, R]):
    """Mixin for ``__xor__`` method using quaxified `jax.lax.xor`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxXorMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x ^ 1
    Array([0, 3, 2], dtype=int32)

    """  # noqa: E501

    def __xor__(self, other: T) -> R:
        return qlax.bitwise_xor(self, other)


class NumpyXorMixin(Generic[T, R]):
    """Mixin for ``__xor__`` method using quaxified `jax.numpy.xor`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyXorMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x ^ 1
    Array([0, 3, 2], dtype=int32)

    """  # noqa: E501

    def __xor__(self, other: T) -> R:
        return qnp.bitwise_xor(self, other)


# -------------------------------------
# `__rxor__`


class LaxRXorMixin(Generic[T, R]):
    """Mixin for ``__rxor__`` method using quaxified `jax.lax.xor`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxRXorMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 ^ x
    Array([0, 3, 2], dtype=int32)

    """  # noqa: E501

    def __rxor__(self, other: T) -> R:
        return qlax.bitwise_xor(other, self)


class NumpyRXorMixin(Generic[T, R]):
    """Mixin for ``__rxor__`` method using quaxified `jax.numpy.xor`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyRXorMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 ^ x
    Array([0, 3, 2], dtype=int32)

    """  # noqa: E501

    def __rxor__(self, other: T) -> R:
        return qnp.bitwise_xor(other, self)


# -------------------------------------


class LaxBothXorMixin(LaxXorMixin[T, R], LaxRXorMixin[T, R]):
    pass


class NumpyBothXorMixin(NumpyXorMixin[T, R], NumpyRXorMixin[T, R]):
    pass


# ================================================
# Or

# -------------------------------------
# `__or__`


class LaxOrMixin(Generic[T, R]):
    """Mixin for ``__or__`` method using quaxified `jax.lax.or_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxOrMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x | 1
    Array([1, 3, 3], dtype=int32)

    """  # noqa: E501

    def __or__(self, other: T) -> R:
        return qlax.bitwise_or(self, other)


class NumpyOrMixin(Generic[T, R]):
    """Mixin for ``__or__`` method using quaxified `jax.numpy.or_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyOrMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> x | 1
    Array([1, 3, 3], dtype=int32)

    """  # noqa: E501

    def __or__(self, other: T) -> R:
        return qnp.bitwise_or(self, other)


# ================================================
# Ror

# -------------------------------------
# `__ror__`


class LaxROrMixin(Generic[T, R]):
    """Mixin for ``__ror__`` method using quaxified `jax.lax.or_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, LaxROrMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 | x
    Array([1, 3, 3], dtype=int32)

    """  # noqa: E501

    def __ror__(self, other: T) -> R:
        return qlax.bitwise_or(other, self)


class NumpyROrMixin(Generic[T, R]):
    """Mixin for ``__ror__`` method using quaxified `jax.numpy.or_`.

    Examples
    --------
    >>> from typing import Any
    >>> import jax
    >>> import jax.numpy as jnp
    >>> from jaxtyping import Array
    >>> from quax import ArrayValue

    >>> class MyArray(ArrayValue, NumpyROrMixin[Any, Array]):
    ...     value: Array
    ...     def aval(self): return jax.core.ShapedArray(self.value.shape, self.value.dtype)
    ...     def materialise(self): return self.value

    >>> x = MyArray(jnp.array([1, 2, 3]))
    >>> 1 | x
    Array([1, 3, 3], dtype=int32)

    """  # noqa: E501

    def __ror__(self, other: T) -> R:
        return qnp.bitwise_or(other, self)


# -------------------------------------


class LaxBothOrMixin(LaxOrMixin[T, R], LaxROrMixin[T, R]):
    pass


class NumpyBothOrMixin(NumpyOrMixin[T, R], NumpyROrMixin[T, R]):
    pass


# =================================================


class LaxMathMixin(
    LaxBothAddMixin[T, R],  # __add__, __radd__
    LaxBothSubMixin[T, R],  # __sub__, __rsub__
    LaxBothMulMixin[T, R],  # __mul__, __rmul__
    LaxMatMulMixin[T, R],  # __matmul__
    LaxBothTrueDivMixin[T, R],  # __truediv__, __rtruediv__
    LaxBothFloorDivMixin[T, R],  # __floordiv__, __rfloordiv__
    LaxBothModMixin[T, R],  # __mod__, __rmod__
    # TODO: divmod
    LaxBothPowMixin[T, R],  # __pow__, __rpow__
):
    pass


class NumpyMathMixin(
    NumpyBothAddMixin[T, R],  # __add__, __radd__
    NumpyBothSubMixin[T, R],  # __sub__, __rsub__
    NumpyBothMulMixin[T, R],  # __mul__, __rmul__
    NumpyMatMulMixin[T, R],  # __matmul__
    NumpyBothTrueDivMixin[T, R],  # __truediv__, __rtruediv__
    NumpyBothFloorDivMixin[T, R],  # __floordiv__, __rfloordiv__
    NumpyBothModMixin[T, R],  # __mod__, __rmod__
    NumpyBothDivModMixin[T, R],  # __divmod__, __rdivmod__
    NumpyBothPowMixin[T, R],  # __pow__, __rpow__
):
    pass


class LaxBitwiseMixin(
    LaxBothLShiftMixin[T, R],  # __lshift__, __rlshift__
    LaxBothRShiftMixin[T, R],  # __rshift__, __rrshift__
    LaxBothAndMixin[T, R],  # __and__, __rand__
    LaxBothXorMixin[T, R],  # __xor__, __rxor__
    LaxBothOrMixin[T, R],  # __or__, __ror__
):
    pass


class NumpyBitwiseMixin(
    NumpyBothLShiftMixin[T, R],  # __lshift__, __rlshift__
    NumpyBothRShiftMixin[T, R],  # __rshift__, __rrshift__
    NumpyBothAndMixin[T, R],  # __and__, __rand__
    NumpyBothXorMixin[T, R],  # __xor__, __rxor__
    NumpyBothOrMixin[T, R],  # __or__, __ror__
):
    pass


# =================================================


class LaxBinaryOpsMixin(LaxMathMixin[T, R], LaxBitwiseMixin[T, R]):
    pass


class NumpyBinaryOpsMixin(NumpyMathMixin[T, R], NumpyBitwiseMixin[T, R]):
    pass
