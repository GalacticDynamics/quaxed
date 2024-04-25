__all__ = [
    "abs",
    "acos",
    "acosh",
    "add",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "bitwise_and",
    "bitwise_left_shift",
    "bitwise_invert",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "ceil",
    "conj",
    "cos",
    "cosh",
    "divide",
    "equal",
    "exp",
    "expm1",
    "floor",
    "floor_divide",
    "greater",
    "greater_equal",
    "imag",
    "isfinite",
    "isinf",
    "isnan",
    "less",
    "less_equal",
    "log",
    "log1p",
    "log2",
    "log10",
    "logaddexp",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]


from jax.experimental import array_api
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def abs(x: ArrayLike, /) -> Value:
    return array_api.abs(x)


@quaxify
def acos(x: ArrayLike, /) -> Value:
    return array_api.acos(x)


@quaxify
def acosh(x: ArrayLike, /) -> Value:
    return array_api.acosh(x)


@quaxify
def add(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.add(x1, x2)


@quaxify
def asin(x: ArrayLike, /) -> Value:
    return array_api.asin(x)


@quaxify
def asinh(x: ArrayLike, /) -> Value:
    return array_api.asinh(x)


@quaxify
def atan(x: ArrayLike, /) -> Value:
    return array_api.atan(x)


@quaxify
def atan2(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.atan2(x1, x2)


@quaxify
def atanh(x: ArrayLike, /) -> Value:
    return array_api.atanh(x)


@quaxify
def bitwise_and(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.bitwise_and(x1, x2)


@quaxify
def bitwise_left_shift(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.bitwise_left_shift(x1, x2)


@quaxify
def bitwise_invert(x: ArrayLike, /) -> Value:
    return array_api.bitwise_invert(x)


@quaxify
def bitwise_or(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.bitwise_or(x1, x2)


@quaxify
def bitwise_right_shift(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.bitwise_right_shift(x1, x2)


@quaxify
def bitwise_xor(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.bitwise_xor(x1, x2)


@quaxify
def ceil(x: ArrayLike, /) -> Value:
    return array_api.ceil(x)


@quaxify
def conj(x: ArrayLike, /) -> Value:
    return array_api.conj(x)


@quaxify
def cos(x: ArrayLike, /) -> Value:
    return array_api.cos(x)


@quaxify
def cosh(x: ArrayLike, /) -> Value:
    return array_api.cosh(x)


@quaxify
def divide(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.divide(x1, x2)


@quaxify
def equal(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.equal(x1, x2)


@quaxify
def exp(x: ArrayLike, /) -> Value:
    return array_api.exp(x)


@quaxify
def expm1(x: ArrayLike, /) -> Value:
    return array_api.expm1(x)


@quaxify
def floor(x: ArrayLike, /) -> Value:
    return array_api.floor(x)


@quaxify
def floor_divide(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.floor_divide(x1, x2)


@quaxify
def greater(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.greater(x1, x2)


@quaxify
def greater_equal(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.greater_equal(x1, x2)


@quaxify
def imag(x: ArrayLike, /) -> Value:
    return array_api.imag(x)


@quaxify
def isfinite(x: ArrayLike, /) -> Value:
    return array_api.isfinite(x)


@quaxify
def isinf(x: ArrayLike, /) -> Value:
    # Jax `isinf` makes a numpy array with value `inf` and then compares it with
    # the input array. If the input array cannot be compared to base numpy
    # arrays, e.g. a Quantity with units, then Jax's `isinf` will raise an
    # unwanted error. Instead, we just negate the `isfinite` function, which
    # should work for all array-like inputs.
    return ~array_api.isfinite(x)


@quaxify
def isnan(x: ArrayLike, /) -> Value:
    return array_api.isnan(x)


@quaxify
def less(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.less(x1, x2)


@quaxify
def less_equal(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.less_equal(x1, x2)


@quaxify
def log(x: ArrayLike, /) -> Value:
    return array_api.log(x)


@quaxify
def log1p(x: ArrayLike, /) -> Value:
    return array_api.log1p(x)


@quaxify
def log2(x: ArrayLike, /) -> Value:
    return array_api.log2(x)


@quaxify
def log10(x: ArrayLike, /) -> Value:
    return array_api.log10(x)


@quaxify
def logaddexp(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.logaddexp(x1, x2)


@quaxify
def logical_and(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.logical_and(x1, x2)


@quaxify
def logical_not(x: ArrayLike, /) -> Value:
    return array_api.logical_not(x)


@quaxify
def logical_or(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.logical_or(x1, x2)


@quaxify
def logical_xor(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.logical_xor(x1, x2)


@quaxify
def multiply(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.multiply(x1, x2)


@quaxify
def negative(x: ArrayLike, /) -> Value:
    return array_api.negative(x)


@quaxify
def not_equal(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.not_equal(x1, x2)


@quaxify
def positive(x: ArrayLike, /) -> Value:
    return array_api.positive(x)


@quaxify
def pow(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.pow(x1, x2)


@quaxify
def real(x: ArrayLike, /) -> Value:
    return array_api.real(x)


@quaxify
def remainder(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.remainder(x1, x2)


@quaxify
def round(x: ArrayLike, /) -> Value:
    return array_api.round(x)


@quaxify
def sign(x: ArrayLike, /) -> Value:
    return array_api.sign(x)


@quaxify
def sin(x: ArrayLike, /) -> Value:
    return array_api.sin(x)


@quaxify
def sinh(x: ArrayLike, /) -> Value:
    return array_api.sinh(x)


@quaxify
def square(x: ArrayLike, /) -> Value:
    return array_api.square(x)


@quaxify
def sqrt(x: ArrayLike, /) -> Value:
    return array_api.sqrt(x)


@quaxify
def subtract(x1: ArrayLike, x2: ArrayLike, /) -> Value:
    return array_api.subtract(x1, x2)


@quaxify
def tan(x: ArrayLike, /) -> Value:
    return array_api.tan(x)


@quaxify
def tanh(x: ArrayLike, /) -> Value:
    return array_api.tanh(x)


@quaxify
def trunc(x: ArrayLike, /) -> Value:
    return array_api.trunc(x)
