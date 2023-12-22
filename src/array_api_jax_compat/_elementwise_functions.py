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
    "copysign",
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
    "maximum",
    "minimum",
    "multiply",
    "negative",
    "not_equal",
    "positive",
    "pow",
    "real",
    "remainder",
    "round",
    "sign",
    "signbit",
    "sin",
    "sinh",
    "square",
    "sqrt",
    "subtract",
    "tan",
    "tanh",
    "trunc",
]


import jax.numpy as jnp
from quax import Value

from ._utils import quaxify


@quaxify
def abs(x: Value, /) -> Value:
    return jnp.abs(x)


@quaxify
def acos(x: Value, /) -> Value:
    return jnp.arccos(x)


@quaxify
def acosh(x: Value, /) -> Value:
    return jnp.arccosh(x)


@quaxify
def add(x1: Value, x2: Value, /) -> Value:
    return jnp.add(x1, x2)


@quaxify
def asin(x: Value, /) -> Value:
    return jnp.arcsin(x)


@quaxify
def asinh(x: Value, /) -> Value:
    return jnp.arcsinh(x)


@quaxify
def atan(x: Value, /) -> Value:
    return jnp.arctan(x)


@quaxify
def atan2(x1: Value, x2: Value, /) -> Value:
    return jnp.arctan2(x1, x2)


@quaxify
def atanh(x: Value, /) -> Value:
    return jnp.arctanh(x)


@quaxify
def bitwise_and(x1: Value, x2: Value, /) -> Value:
    return jnp.bitwise_and(x1, x2)


@quaxify
def bitwise_left_shift(x1: Value, x2: Value, /) -> Value:
    return jnp.left_shift(x1, x2)


@quaxify
def bitwise_invert(x: Value, /) -> Value:
    return jnp.bitwise_not(x)


@quaxify
def bitwise_or(x1: Value, x2: Value, /) -> Value:
    return jnp.bitwise_or(x1, x2)


@quaxify
def bitwise_right_shift(x1: Value, x2: Value, /) -> Value:
    return jnp.right_shift(x1, x2)


@quaxify
def bitwise_xor(x1: Value, x2: Value, /) -> Value:
    return jnp.bitwise_xor(x1, x2)


@quaxify
def ceil(x: Value, /) -> Value:
    return jnp.ceil(x)


@quaxify
def conj(x: Value, /) -> Value:
    return jnp.conj(x)


@quaxify
def copysign(x1: Value, x2: Value, /) -> Value:
    return jnp.copysign(x1, x2)


@quaxify
def cos(x: Value, /) -> Value:
    return jnp.cos(x)


@quaxify
def cosh(x: Value, /) -> Value:
    return jnp.cosh(x)


@quaxify
def divide(x1: Value, x2: Value, /) -> Value:
    return jnp.divide(x1, x2)


@quaxify
def equal(x1: Value, x2: Value, /) -> Value:
    return jnp.equal(x1, x2)


@quaxify
def exp(x: Value, /) -> Value:
    return jnp.exp(x)


@quaxify
def expm1(x: Value, /) -> Value:
    return jnp.expm1(x)


@quaxify
def floor(x: Value, /) -> Value:
    return jnp.floor(x)


@quaxify
def floor_divide(x1: Value, x2: Value, /) -> Value:
    return jnp.floor_divide(x1, x2)


@quaxify
def greater(x1: Value, x2: Value, /) -> Value:
    return jnp.greater(x1, x2)


@quaxify
def greater_equal(x1: Value, x2: Value, /) -> Value:
    return jnp.greater_equal(x1, x2)


@quaxify
def imag(x: Value, /) -> Value:
    return jnp.imag(x)


@quaxify
def isfinite(x: Value, /) -> Value:
    return jnp.isfinite(x)


@quaxify
def isinf(x: Value, /) -> Value:
    return jnp.isinf(x)


@quaxify
def isnan(x: Value, /) -> Value:
    return jnp.isnan(x)


@quaxify
def less(x1: Value, x2: Value, /) -> Value:
    return jnp.less(x1, x2)


@quaxify
def less_equal(x1: Value, x2: Value, /) -> Value:
    return jnp.less_equal(x1, x2)


@quaxify
def log(x: Value, /) -> Value:
    return jnp.log(x)


@quaxify
def log1p(x: Value, /) -> Value:
    return jnp.log1p(x)


@quaxify
def log2(x: Value, /) -> Value:
    return jnp.log2(x)


@quaxify
def log10(x: Value, /) -> Value:
    return jnp.log10(x)


@quaxify
def logaddexp(x1: Value, x2: Value, /) -> Value:
    return jnp.logaddexp(x1, x2)


@quaxify
def logical_and(x1: Value, x2: Value, /) -> Value:
    return jnp.logical_and(x1, x2)


@quaxify
def logical_not(x: Value, /) -> Value:
    return jnp.logical_not(x)


@quaxify
def logical_or(x1: Value, x2: Value, /) -> Value:
    return jnp.logical_or(x1, x2)


@quaxify
def logical_xor(x1: Value, x2: Value, /) -> Value:
    return jnp.logical_xor(x1, x2)


@quaxify
def maximum(x1: Value, x2: Value, /) -> Value:
    return jnp.maximum(x1, x2)


@quaxify
def minimum(x1: Value, x2: Value, /) -> Value:
    return jnp.minimum(x1, x2)


@quaxify
def multiply(x1: Value, x2: Value, /) -> Value:
    return jnp.multiply(x1, x2)


@quaxify
def negative(x: Value, /) -> Value:
    return jnp.negative(x)


@quaxify
def not_equal(x1: Value, x2: Value, /) -> Value:
    return jnp.not_equal(x1, x2)


@quaxify
def positive(x: Value, /) -> Value:
    return jnp.positive(x)


@quaxify
def pow(x1: Value, x2: Value, /) -> Value:
    return jnp.power(x1, x2)


@quaxify
def real(x: Value, /) -> Value:
    return jnp.real(x)


@quaxify
def remainder(x1: Value, x2: Value, /) -> Value:
    return jnp.remainder(x1, x2)


@quaxify
def round(x: Value, /) -> Value:
    return jnp.round(x)


@quaxify
def sign(x: Value, /) -> Value:
    return jnp.sign(x)


@quaxify
def signbit(x: Value, /) -> Value:
    return jnp.signbit(x)


@quaxify
def sin(x: Value, /) -> Value:
    return jnp.sin(x)


@quaxify
def sinh(x: Value, /) -> Value:
    return jnp.sinh(x)


@quaxify
def square(x: Value, /) -> Value:
    return jnp.square(x)


@quaxify
def sqrt(x: Value, /) -> Value:
    return jnp.sqrt(x)


@quaxify
def subtract(x1: Value, x2: Value, /) -> Value:
    return jnp.subtract(x1, x2)


@quaxify
def tan(x: Value, /) -> Value:
    return jnp.tan(x)


@quaxify
def tanh(x: Value, /) -> Value:
    return jnp.tanh(x)


@quaxify
def trunc(x: Value, /) -> Value:
    return jnp.trunc(x)
