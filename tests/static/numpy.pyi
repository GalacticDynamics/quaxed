# Static type tests for quaxed.numpy with MyArray.
#
# This module contains static type tests that verify MyArray works correctly
# with quaxed.numpy functions. Type checkers (mypy, pyright) should verify
# these without runtime execution.

import quaxed.numpy as qnp
from tests.myarray import MyArray

# Define test arrays
arr000 = MyArray(qnp.asarray([1.0, 2.0, 3.0]))
arr001 = MyArray(qnp.asarray([4.0, 5.0, 6.0]))
arr002 = MyArray(qnp.asarray([[1.0, 2.0], [3.0, 4.0]]))
scalar000: float = 2.0

##############################################################################
# Single-Argument Functions

# Trigonometric functions
result000: MyArray = qnp.sin(arr000)
result001: MyArray = qnp.cos(arr000)
result002: MyArray = qnp.tan(arr000)
result003: MyArray = qnp.arcsin(arr000)
result004: MyArray = qnp.arccos(arr000)
result005: MyArray = qnp.arctan(arr000)
result006: MyArray = qnp.sinh(arr000)
result007: MyArray = qnp.cosh(arr000)
result008: MyArray = qnp.tanh(arr000)
result009: MyArray = qnp.arcsinh(arr000)
result010: MyArray = qnp.arccosh(arr000)
result011: MyArray = qnp.arctanh(arr000)

# Aliases for trigonometric functions
result012: MyArray = qnp.asin(arr000)
result013: MyArray = qnp.acos(arr000)
result014: MyArray = qnp.atan(arr000)
result015: MyArray = qnp.asinh(arr000)
result016: MyArray = qnp.acosh(arr000)
result017: MyArray = qnp.atanh(arr000)

# Exponential and logarithmic functions
result020: MyArray = qnp.exp(arr000)
result021: MyArray = qnp.exp2(arr000)
result022: MyArray = qnp.expm1(arr000)
result023: MyArray = qnp.log(arr000)
result024: MyArray = qnp.log10(arr000)
result025: MyArray = qnp.log1p(arr000)
result026: MyArray = qnp.log2(arr000)

# Power and root functions
result030: MyArray = qnp.sqrt(arr000)
result031: MyArray = qnp.square(arr000)
result032: MyArray = qnp.cbrt(arr000)
result033: MyArray = qnp.reciprocal(arr000)

# Rounding and absolute value functions
result040: MyArray = qnp.abs(arr000)
result041: MyArray = qnp.absolute(arr000)
result042: MyArray = qnp.fabs(arr000)
result043: MyArray = qnp.ceil(arr000)
result044: MyArray = qnp.floor(arr000)
result045: MyArray = qnp.trunc(arr000)
result046: MyArray = qnp.rint(arr000)
result047: MyArray = qnp.round(arr000)
result048: MyArray = qnp.fix(arr000)

# Sign and angle functions
result050: MyArray = qnp.sign(arr000)
result051: MyArray = qnp.signbit(arr000)
result052: MyArray = qnp.negative(arr000)
result053: MyArray = qnp.positive(arr000)
result054: MyArray = qnp.angle(arr000)
result055: MyArray = qnp.real(arr000)
result056: MyArray = qnp.imag(arr000)
result057: MyArray = qnp.conj(arr000)
result058: MyArray = qnp.conjugate(arr000)

# Logical functions
result060: MyArray = qnp.logical_not(arr000)
result061: MyArray = qnp.bitwise_not(arr000)
result062: MyArray = qnp.bitwise_invert(arr000)
result063: MyArray = qnp.invert(arr000)

# Conversion functions
result070: MyArray = qnp.deg2rad(arr000)
result071: MyArray = qnp.rad2deg(arr000)
result072: MyArray = qnp.degrees(arr000)
result073: MyArray = qnp.radians(arr000)

# Special functions
result080: MyArray = qnp.sinc(arr000)
result081: MyArray = qnp.i0(arr000)

# Array manipulation that preserves type
result090: MyArray = qnp.copy(arr000)
result091: MyArray = qnp.ravel(arr000)
result092: MyArray = qnp.reshape(arr000, (3, 1))
result093: MyArray = qnp.squeeze(arr002)
result094: MyArray = qnp.transpose(arr002)
result095: MyArray = qnp.flip(arr000)
result096: MyArray = qnp.fliplr(arr002)
result097: MyArray = qnp.flipud(arr002)
result098: MyArray = qnp.rot90(arr002)
result099: MyArray = qnp.expand_dims(arr000, 0)
result099a: MyArray = qnp.expand_dims(arr000, (0, 2))

# Cumulative operations
result100: MyArray = qnp.cumsum(arr000)
result101: MyArray = qnp.cumprod(arr000)

# Reduction operations with MyArray
result110: MyArray = qnp.sum(arr002)
result111: MyArray = qnp.mean(arr002)
result112: MyArray = qnp.max(arr002)
result113: MyArray = qnp.min(arr002)
result114: MyArray = qnp.amax(arr002)
result115: MyArray = qnp.amin(arr002)
result116: MyArray = qnp.prod(arr002)
result117: MyArray = qnp.std(arr002)
result118: MyArray = qnp.var(arr002)
result119: MyArray = qnp.median(arr002)

# NaN-aware reductions
result120: MyArray = qnp.nansum(arr002)
result121: MyArray = qnp.nanmean(arr002)
result122: MyArray = qnp.nanmax(arr002)
result123: MyArray = qnp.nanmin(arr002)
result124: MyArray = qnp.nanprod(arr002)
result125: MyArray = qnp.nanstd(arr002)
result126: MyArray = qnp.nanvar(arr002)
result127: MyArray = qnp.nanmedian(arr002)
result128: MyArray = qnp.nancumsum(arr000)
result129: MyArray = qnp.nancumprod(arr000)

# Sorting and searching
result130: MyArray = qnp.sort(arr000)
result131: MyArray = qnp.argsort(arr000)

##############################################################################
# Two-Argument Functions

# Arithmetic operations
result132: MyArray = qnp.add(arr000, arr001)
result133: MyArray = qnp.subtract(arr000, arr001)
result134: MyArray = qnp.multiply(arr000, arr001)
result135: MyArray = qnp.divide(arr000, arr001)
result136: MyArray = qnp.power(arr000, arr001)
result137: MyArray = qnp.floor_divide(arr000, arr001)
result138: MyArray = qnp.true_divide(arr000, arr001)
result139: MyArray = qnp.mod(arr000, arr001)
result140: MyArray = qnp.remainder(arr000, arr001)
result141: MyArray = qnp.fmod(arr000, arr001)

# Comparison operations
result150: MyArray = qnp.maximum(arr000, arr001)
result151: MyArray = qnp.minimum(arr000, arr001)
result152: MyArray = qnp.fmax(arr000, arr001)
result153: MyArray = qnp.fmin(arr000, arr001)
result154: MyArray = qnp.equal(arr000, arr001)
result155: MyArray = qnp.not_equal(arr000, arr001)
result156: MyArray = qnp.greater(arr000, arr001)
result157: MyArray = qnp.greater_equal(arr000, arr001)
result158: MyArray = qnp.less(arr000, arr001)
result159: MyArray = qnp.less_equal(arr000, arr001)

# Bitwise operations
result160: MyArray = qnp.bitwise_and(arr000, arr001)
result161: MyArray = qnp.bitwise_or(arr000, arr001)
result162: MyArray = qnp.bitwise_xor(arr000, arr001)
result163: MyArray = qnp.left_shift(arr000, arr001)
result164: MyArray = qnp.right_shift(arr000, arr001)
result165: MyArray = qnp.bitwise_left_shift(arr000, arr001)
result166: MyArray = qnp.bitwise_right_shift(arr000, arr001)

# Logical operations
result170: MyArray = qnp.logical_and(arr000, arr001)
result171: MyArray = qnp.logical_or(arr000, arr001)
result172: MyArray = qnp.logical_xor(arr000, arr001)

# Trigonometric operations
result180: MyArray = qnp.arctan2(arr000, arr001)
result181: MyArray = qnp.atan2(arr000, arr001)
result182: MyArray = qnp.hypot(arr000, arr001)

# Special math operations
result190: MyArray = qnp.copysign(arr000, arr001)
result191: MyArray = qnp.nextafter(arr000, arr001)
result192: MyArray = qnp.ldexp(arr000, arr001)
result193: MyArray = qnp.gcd(arr000, arr001)
result194: MyArray = qnp.lcm(arr000, arr001)

# Logarithmic operations
result200: MyArray = qnp.logaddexp(arr000, arr001)
result201: MyArray = qnp.logaddexp2(arr000, arr001)

# Matrix/vector operations
result210: MyArray = qnp.dot(arr000, arr001)
result211: MyArray = qnp.inner(arr000, arr001)
# TODO: outer needs overload fix - second param should support _ArrayValueT
# result212: MyArray = qnp.outer(arr000, arr001)
result213: MyArray = qnp.matmul(arr002, arr002)
result214: MyArray = qnp.vdot(arr000, arr001)
result215: MyArray = qnp.kron(arr000, arr001)
result216: MyArray = qnp.cross(arr000, arr001)

# Statistical/signal operations
result220: MyArray = qnp.convolve(arr000, arr001)
result221: MyArray = qnp.correlate(arr000, arr001)

# Test that binary ufuncs with mixed types preserve MyArray
result230: MyArray = qnp.add(arr000, scalar000)
result231: MyArray = qnp.add(scalar000, arr000)
result232: MyArray = qnp.multiply(arr000, scalar000)
result233: MyArray = qnp.multiply(scalar000, arr000)
result234: MyArray = qnp.power(arr000, scalar000)
result235: MyArray = qnp.maximum(arr000, scalar000)
result236: MyArray = qnp.minimum(arr000, scalar000)

# test full_like
result240: MyArray = qnp.full_like(arr002, 7.0)
result241: MyArray = qnp.full_like(arr002, MyArray(qnp.array(7.0)))

# Test reshape preserves MyArray type
result250: MyArray = qnp.reshape(arr000, (3, 1))

##############################################################################
# Multi-Argument Functions
