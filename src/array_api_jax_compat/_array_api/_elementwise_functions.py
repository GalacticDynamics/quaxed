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
from quax import Value

from array_api_jax_compat._utils import quaxify


@quaxify
def abs(x: Value, /) -> Value:
    return array_api.abs(x)


@quaxify
def acos(x: Value, /) -> Value:
    return array_api.acos(x)


@quaxify
def acosh(x: Value, /) -> Value:
    return array_api.acosh(x)


@quaxify
def add(x1: Value, x2: Value, /) -> Value:
    return array_api.add(x1, x2)


@quaxify
def asin(x: Value, /) -> Value:
    return array_api.asin(x)


@quaxify
def asinh(x: Value, /) -> Value:
    return array_api.asinh(x)


@quaxify
def atan(x: Value, /) -> Value:
    return array_api.atan(x)


@quaxify
def atan2(x1: Value, x2: Value, /) -> Value:
    return array_api.atan2(x1, x2)


@quaxify
def atanh(x: Value, /) -> Value:
    return array_api.atanh(x)


@quaxify
def bitwise_and(x1: Value, x2: Value, /) -> Value:
    return array_api.bitwise_and(x1, x2)


@quaxify
def bitwise_left_shift(x1: Value, x2: Value, /) -> Value:
    return array_api.bitwise_left_shift(x1, x2)


@quaxify
def bitwise_invert(x: Value, /) -> Value:
    return array_api.bitwise_invert(x)


@quaxify
def bitwise_or(x1: Value, x2: Value, /) -> Value:
    return array_api.bitwise_or(x1, x2)


@quaxify
def bitwise_right_shift(x1: Value, x2: Value, /) -> Value:
    return array_api.bitwise_right_shift(x1, x2)


@quaxify
def bitwise_xor(x1: Value, x2: Value, /) -> Value:
    return array_api.bitwise_xor(x1, x2)


@quaxify
def ceil(x: Value, /) -> Value:
    return array_api.ceil(x)


@quaxify
def conj(x: Value, /) -> Value:
    return array_api.conj(x)


@quaxify
def cos(x: Value, /) -> Value:
    return array_api.cos(x)


@quaxify
def cosh(x: Value, /) -> Value:
    return array_api.cosh(x)


@quaxify
def divide(x1: Value, x2: Value, /) -> Value:
    return array_api.divide(x1, x2)


@quaxify
def equal(x1: Value, x2: Value, /) -> Value:
    return array_api.equal(x1, x2)


@quaxify
def exp(x: Value, /) -> Value:
    return array_api.exp(x)


@quaxify
def expm1(x: Value, /) -> Value:
    return array_api.expm1(x)


@quaxify
def floor(x: Value, /) -> Value:
    return array_api.floor(x)


@quaxify
def floor_divide(x1: Value, x2: Value, /) -> Value:
    return array_api.floor_divide(x1, x2)


@quaxify
def greater(x1: Value, x2: Value, /) -> Value:
    return array_api.greater(x1, x2)


@quaxify
def greater_equal(x1: Value, x2: Value, /) -> Value:
    return array_api.greater_equal(x1, x2)


@quaxify
def imag(x: Value, /) -> Value:
    return array_api.imag(x)


@quaxify
def isfinite(x: Value, /) -> Value:
    return array_api.isfinite(x)


@quaxify
def isinf(x: Value, /) -> Value:
    return array_api.isinf(x)


@quaxify
def isnan(x: Value, /) -> Value:
    return array_api.isnan(x)


@quaxify
def less(x1: Value, x2: Value, /) -> Value:
    return array_api.less(x1, x2)


@quaxify
def less_equal(x1: Value, x2: Value, /) -> Value:
    return array_api.less_equal(x1, x2)


@quaxify
def log(x: Value, /) -> Value:
    return array_api.log(x)


@quaxify
def log1p(x: Value, /) -> Value:
    return array_api.log1p(x)


@quaxify
def log2(x: Value, /) -> Value:
    return array_api.log2(x)


@quaxify
def log10(x: Value, /) -> Value:
    return array_api.log10(x)


@quaxify
def logaddexp(x1: Value, x2: Value, /) -> Value:
    return array_api.logaddexp(x1, x2)


@quaxify
def logical_and(x1: Value, x2: Value, /) -> Value:
    return array_api.logical_and(x1, x2)


@quaxify
def logical_not(x: Value, /) -> Value:
    return array_api.logical_not(x)


@quaxify
def logical_or(x1: Value, x2: Value, /) -> Value:
    return array_api.logical_or(x1, x2)


@quaxify
def logical_xor(x1: Value, x2: Value, /) -> Value:
    return array_api.logical_xor(x1, x2)


@quaxify
def multiply(x1: Value, x2: Value, /) -> Value:
    return array_api.multiply(x1, x2)


@quaxify
def negative(x: Value, /) -> Value:
    return array_api.negative(x)


@quaxify
def not_equal(x1: Value, x2: Value, /) -> Value:
    return array_api.not_equal(x1, x2)


@quaxify
def positive(x: Value, /) -> Value:
    return array_api.positive(x)


@quaxify
def pow(x1: Value, x2: Value, /) -> Value:
    return array_api.pow(x1, x2)


@quaxify
def real(x: Value, /) -> Value:
    return array_api.real(x)


@quaxify
def remainder(x1: Value, x2: Value, /) -> Value:
    return array_api.remainder(x1, x2)


@quaxify
def round(x: Value, /) -> Value:
    return array_api.round(x)


@quaxify
def sign(x: Value, /) -> Value:
    return array_api.sign(x)


@quaxify
def sin(x: Value, /) -> Value:
    return array_api.sin(x)


@quaxify
def sinh(x: Value, /) -> Value:
    return array_api.sinh(x)


@quaxify
def square(x: Value, /) -> Value:
    return array_api.square(x)


@quaxify
def sqrt(x: Value, /) -> Value:
    return array_api.sqrt(x)


@quaxify
def subtract(x1: Value, x2: Value, /) -> Value:
    return array_api.subtract(x1, x2)


@quaxify
def tan(x: Value, /) -> Value:
    return array_api.tan(x)


@quaxify
def tanh(x: Value, /) -> Value:
    return array_api.tanh(x)


@quaxify
def trunc(x: Value, /) -> Value:
    return array_api.trunc(x)
