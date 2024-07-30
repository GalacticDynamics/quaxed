"""Test with :class:`MyArray` inputs."""

import jax.experimental.array_api as jax_xp
import jax.numpy as jnp
import pytest
from jax import Array
from jax._src.numpy.setops import (
    _UniqueAllResult,
    _UniqueCountsResult,
    _UniqueInverseResult,
)

import quaxed.array_api as xp
from quaxed.array_api._data_type_functions import FInfo, IInfo

from ..myarray import MyArray

###############################################################################

# =============================================================================
# Constants


def test_e():
    """Test `e`."""
    assert not isinstance(xp.e, MyArray)


def test_inf():
    """Test `inf`."""
    assert not isinstance(xp.inf, MyArray)


def test_nan():
    """Test `nan`."""
    assert not isinstance(xp.nan, MyArray)


def test_newaxis():
    """Test `newaxis`."""
    assert not isinstance(xp.newaxis, MyArray)


def test_pi():
    """Test `pi`."""
    assert not isinstance(xp.pi, MyArray)


# =============================================================================
# Creation functions


def test_arange():
    """Test `arange`."""
    # TODO: test the start, stop, step, dtype, device arguments
    got = xp.arange(MyArray(3))
    expected = MyArray(jax_xp.arange(3))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_asarray():
    """Test `asarray`."""
    # TODO: test the dtype, device, copy arguments
    got = xp.asarray(MyArray([1, 2, 3]))
    expected = MyArray(jax_xp.asarray([1, 2, 3]))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


@pytest.mark.xfail(reason="returns a jax.Array")
def test_empty():
    """Test `empty`."""
    got = xp.empty((2, 3))
    assert isinstance(got, MyArray)


def test_empty_like():
    """Test `empty_like`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.empty_like(x)
    expected = MyArray(jax_xp.empty_like(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


@pytest.mark.xfail(reason="returns a jax.Array")
def test_eye():
    """Test `eye`."""
    got = xp.eye(3)

    assert isinstance(got, MyArray)


@pytest.mark.skip("TODO")
def test_from_dlpack():
    """Test `from_dlpack`."""


@pytest.mark.xfail(reason="returns a jax.Array")
def test_full():
    """Test `full`."""
    got = xp.full((2, 3), 1.0)

    assert isinstance(got, MyArray)


def test_full_like():
    """Test `full_like`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.full_like(x, 1.0)
    expected = MyArray(jax_xp.full_like(x.array, 1.0))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_linspace():
    """Test `linspace`."""
    # TODO: test the dtype, device, endpoint arguments
    got = xp.linspace(MyArray(0.0), MyArray(10.0), 11)
    expected = MyArray(jax_xp.linspace(0.0, 10.0, 11))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_meshgrid():
    """Test `meshgrid`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))

    got1, got2 = xp.meshgrid(x, y)
    exp1, exp2 = jax_xp.meshgrid(x.array, y.array)

    assert isinstance(got1, MyArray)
    assert jnp.array_equal(got1.array, exp1)

    assert isinstance(got2, MyArray)
    assert jnp.array_equal(got2.array, exp2)


@pytest.mark.xfail(reason="returns a jax.Array")
def test_ones():
    """Test `ones`."""
    assert isinstance(xp.ones((2, 3)), MyArray)


def test_ones_like():
    """Test `ones_like`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.ones_like(x)
    expected = MyArray(jax_xp.ones_like(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_tril():
    """Test `tril`."""
    x = MyArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    got = xp.tril(x)
    expected = MyArray(jax_xp.tril(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_triu():
    """Test `triu`."""
    x = MyArray([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    got = xp.triu(x)
    expected = MyArray(jax_xp.triu(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


@pytest.mark.xfail(reason="returns a jax.Array")
def test_zeros():
    """Test `zeros`."""
    assert isinstance(xp.zeros((2, 3)), MyArray)


def test_zeros_like():
    """Test `zeros_like`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.zeros_like(x)
    expected = MyArray(jax_xp.zeros_like(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Data-type functions


def test_astype():
    """Test `astype`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.astype(x, jnp.float32)
    expected = MyArray(jax_xp.asarray(x.array, dtype=jnp.float32))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


@pytest.mark.skip("TODO")
def test_can_cast():
    """Test `can_cast`."""
    # x = jax_xp.asarray([1, 2, 3], dtype=float)
    # mx = MyArray(x)

    # assert xp.can_cast(x, float)
    # assert xp.can_cast(mx, float)


def test_finfo():
    """Test `finfo`."""
    got = xp.finfo(jnp.float32)
    expected = jax_xp.finfo(jnp.float32)

    assert isinstance(got, FInfo)
    for attr in ("bits", "eps", "max", "min", "smallest_normal", "dtype"):
        assert getattr(got, attr) == getattr(expected, attr)


def test_iinfo():
    """Test `iinfo`."""
    got = xp.iinfo(jnp.int32)
    expected = jax_xp.iinfo(jnp.int32)

    assert isinstance(got, IInfo)
    for attr in ("kind", "bits", "min", "max", "dtype"):
        assert getattr(got, attr) == getattr(expected, attr)


def test_isdtype():
    """Test `isdtype`."""
    # True by definition


def test_result_type():
    """Test `result_type`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.result_type(x, y)
    expected = jax_xp.result_type(x.array, y.array)

    assert isinstance(got, jnp.dtype)
    assert got == expected


# =============================================================================
# Elementwise functions


def test_abs():
    """Test `abs`."""
    x = MyArray([-1, 2, -3])
    got = xp.abs(x)
    expected = MyArray(jax_xp.abs(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_acos():
    """Test `acos`."""
    x = MyArray(xp.asarray([-1, 0, 1], dtype=float))
    got = xp.acos(x)
    expected = MyArray(jax_xp.acos(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_acosh():
    """Test `acosh`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.acosh(x)
    expected = MyArray(jax_xp.acosh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_add():
    """Test `add`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.add(x, y)
    expected = MyArray(jax_xp.add(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_asin():
    """Test `asin`."""
    x = MyArray(xp.asarray([-1, 0, 1], dtype=float))
    got = xp.asin(x)
    expected = MyArray(jax_xp.asin(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_asinh():
    """Test `asinh`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.asinh(x)
    expected = MyArray(jax_xp.asinh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_atan():
    """Test `atan`."""
    x = MyArray(xp.asarray([-1, 0, 1], dtype=float))
    got = xp.atan(x)
    expected = MyArray(jax_xp.atan(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_atan2():
    """Test `atan2`."""
    x = MyArray(xp.asarray([-1, 0, 1], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.atan2(x, y)
    expected = MyArray(jax_xp.atan2(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_atanh():
    """Test `atanh`."""
    x = MyArray(xp.asarray([-1, 0, 1], dtype=float))
    got = xp.atanh(x)
    expected = MyArray(jax_xp.atanh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_and():
    """Test `bitwise_and`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    y = MyArray(xp.asarray([4, 5, 6], dtype=int))
    got = xp.bitwise_and(x, y)
    expected = MyArray(jax_xp.bitwise_and(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_left_shift():
    """Test `bitwise_left_shift`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    y = MyArray(xp.asarray([4, 5, 6], dtype=int))
    got = xp.bitwise_left_shift(x, y)
    expected = MyArray(jax_xp.bitwise_left_shift(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_invert():
    """Test `bitwise_invert`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    got = xp.bitwise_invert(x)
    expected = MyArray(jax_xp.bitwise_invert(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_or():
    """Test `bitwise_or`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    y = MyArray(xp.asarray([4, 5, 6], dtype=int))
    got = xp.bitwise_or(x, y)
    expected = MyArray(jax_xp.bitwise_or(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_right_shift():
    """Test `bitwise_right_shift`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    y = MyArray(xp.asarray([4, 5, 6], dtype=int))
    got = xp.bitwise_right_shift(x, y)
    expected = MyArray(jax_xp.bitwise_right_shift(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_bitwise_xor():
    """Test `bitwise_xor`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=int))
    y = MyArray(xp.asarray([4, 5, 6], dtype=int))
    got = xp.bitwise_xor(x, y)
    expected = MyArray(jax_xp.bitwise_xor(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_ceil():
    """Test `ceil`."""
    x = MyArray([1.1, 2.2, 3.3])
    got = xp.ceil(x)
    expected = MyArray(jax_xp.ceil(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_conj():
    """Test `conj`."""
    x = MyArray([1 + 2j, 3 + 4j])
    got = xp.conj(x)
    expected = MyArray(jax_xp.conj(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_cos():
    """Test `cos`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.cos(x)
    expected = MyArray(jax_xp.cos(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_cosh():
    """Test `cosh`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.cosh(x)
    expected = MyArray(jax_xp.cosh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_divide():
    """Test `divide`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.divide(x, y)
    expected = MyArray(jax_xp.divide(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_equal():
    """Test `equal`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.equal(x, y)
    expected = MyArray(jax_xp.equal(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_exp():
    """Test `exp`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.exp(x)
    expected = MyArray(jax_xp.exp(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_expm1():
    """Test `expm1`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.expm1(x)
    expected = MyArray(jax_xp.expm1(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_floor():
    """Test `floor`."""
    x = MyArray([1.1, 2.2, 3.3])
    got = xp.floor(x)
    expected = MyArray(jax_xp.floor(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_floor_divide():
    """Test `floor_divide`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.floor_divide(x, y)
    expected = MyArray(jax_xp.floor_divide(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_greater():
    """Test `greater`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.greater(x, y)
    expected = MyArray(jax_xp.greater(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_greater_equal():
    """Test `greater_equal`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.greater_equal(x, y)
    expected = MyArray(jax_xp.greater_equal(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_imag():
    """Test `imag`."""
    x = MyArray([1 + 2j, 3 + 4j])
    got = xp.imag(x)
    expected = MyArray(jax_xp.imag(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_isfinite():
    """Test `isfinite`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.isfinite(x)
    expected = MyArray(jax_xp.isfinite(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_isinf():
    """Test `isinf`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.isinf(x)
    expected = MyArray(jax_xp.isinf(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_isnan():
    """Test `isnan`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.isnan(x)
    expected = MyArray(jax_xp.isnan(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_less():
    """Test `less`."""
    x = MyArray([1, 5, 3])
    y = MyArray([4, 2, 6])
    got = xp.less(x, y)
    expected = MyArray(jax_xp.less(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_less_equal():
    """Test `less_equal`."""
    x = MyArray([1, 5, 3])
    y = MyArray([4, 2, 6])
    got = xp.less_equal(x, y)
    expected = MyArray(jax_xp.less_equal(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_log():
    """Test `log`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.log(x)
    expected = MyArray(jax_xp.log(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_log1p():
    """Test `log1p`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.log1p(x)
    expected = MyArray(jax_xp.log1p(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_log2():
    """Test `log2`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.log2(x)
    expected = MyArray(jax_xp.log2(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_log10():
    """Test `log10`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.log10(x)
    expected = MyArray(jax_xp.log10(x.array))

    assert isinstance(got, MyArray)
    assert jnp.allclose(got.array, expected.array)


def test_logaddexp():
    """Test `logaddexp`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.logaddexp(x, y)
    expected = MyArray(jax_xp.logaddexp(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_logical_and():
    """Test `logical_and`."""
    x = MyArray([True, False, True])
    y = MyArray([False, True, False])
    got = xp.logical_and(x, y)
    expected = MyArray(jax_xp.logical_and(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_logical_not():
    """Test `logical_not`."""
    x = MyArray([True, False, True])
    got = xp.logical_not(x)
    expected = MyArray(jax_xp.logical_not(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_logical_or():
    """Test `logical_or`."""
    x = MyArray([True, False, True])
    y = MyArray([False, True, False])
    got = xp.logical_or(x, y)
    expected = MyArray(jax_xp.logical_or(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_logical_xor():
    """Test `logical_xor`."""
    x = MyArray([True, False, True])
    y = MyArray([False, True, False])
    got = xp.logical_xor(x, y)
    expected = MyArray(jax_xp.logical_xor(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_multiply():
    """Test `multiply`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.multiply(x, y)
    expected = MyArray(jax_xp.multiply(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_negative():
    """Test `negative`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.negative(x)
    expected = MyArray(jax_xp.negative(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_not_equal():
    """Test `not_equal`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 2, 6], dtype=float))
    got = xp.not_equal(x, y)
    expected = MyArray(jax_xp.not_equal(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_positive():
    """Test `positive`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.positive(x)
    expected = MyArray(jax_xp.positive(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_pow():
    """Test `pow`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 2, 6], dtype=float))
    got = xp.pow(x, y)
    expected = MyArray(jax_xp.pow(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_real():
    """Test `real`."""
    x = MyArray([1 + 2j, 3 + 4j])
    got = xp.real(x)
    expected = MyArray(jax_xp.real(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_remainder():
    """Test `remainder`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.remainder(x, y)
    expected = MyArray(jax_xp.remainder(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_round():
    """Test `round`."""
    x = MyArray([1.1, 2.2, 3.3])
    got = xp.round(x)
    expected = MyArray(jax_xp.round(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_sign():
    """Test `sign`."""
    x = MyArray([-1, 2, -3])
    got = xp.sign(x)
    expected = MyArray(jax_xp.sign(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_sin():
    """Test `sin`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.sin(x)
    expected = MyArray(jax_xp.sin(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_sinh():
    """Test `sinh`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.sinh(x)
    expected = MyArray(jax_xp.sinh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_square():
    """Test `square`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.square(x)
    expected = MyArray(jax_xp.square(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_sqrt():
    """Test `sqrt`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.sqrt(x)
    expected = MyArray(jax_xp.sqrt(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_subtract():
    """Test `subtract`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.subtract(x, y)
    expected = MyArray(jax_xp.subtract(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_tan():
    """Test `tan`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.tan(x)
    expected = MyArray(jax_xp.tan(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_tanh():
    """Test `tanh`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.tanh(x)
    expected = MyArray(jax_xp.tanh(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_trunc():
    """Test `trunc`."""
    x = MyArray([1.1, 2.2, 3.3])
    got = xp.trunc(x)
    expected = MyArray(jax_xp.trunc(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Indexing functions


def test_take():
    """Test `take`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    indices = MyArray(xp.asarray([0, 1, 2], dtype=int))
    got = xp.take(x, indices)
    expected = MyArray(jax_xp.take(x.array, indices.array, axis=None))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Linear algebra functions


def test_matmul():
    """Test `matmul`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.matmul(x, y)
    expected = MyArray(jax_xp.matmul(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_matrix_transpose():
    """Test `matrix_transpose`."""
    x = MyArray(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float))
    got = xp.matrix_transpose(x)
    expected = MyArray(jax_xp.matrix_transpose(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_tensordot():
    """Test `tensordot`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    axes = 1
    got = xp.tensordot(x, y, axes=axes)
    expected = MyArray(jax_xp.tensordot(x.array, y.array, axes=axes))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_vecdot():
    """Test `vecdot`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.vecdot(x, y)
    expected = MyArray(jax_xp.vecdot(x.array, y.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Manipulation functions


def test_broadcast_arrays():
    """Test `broadcast_arrays`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4], dtype=float))
    got = xp.broadcast_arrays(x, y)
    expected = jax_xp.broadcast_arrays(x.array, y.array)

    assert isinstance(got, tuple | list)
    assert len(got) == len(expected)
    for got_, expected_ in zip(got, expected, strict=True):
        assert isinstance(got_, MyArray)
        assert jnp.array_equal(got_.array, expected_)


def test_broadcast_to():
    """Test `broadcast_to`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    shape = (2, 3)
    got = xp.broadcast_to(x, shape)
    expected = MyArray(jax_xp.broadcast_to(x.array, shape))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_concat():
    """Test `concat`."""
    # TODO: test the axis argument
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4], dtype=float))
    got = xp.concat((x, y))
    expected = MyArray(jax_xp.concat((x.array, y.array)))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_expand_dims():
    """Test `expand_dims`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.expand_dims(x, axis=0)
    expected = MyArray(jax_xp.expand_dims(x.array, axis=0))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_flip():
    """Test `flip`."""
    x = MyArray(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float))
    got = xp.flip(x)
    expected = MyArray(jax_xp.flip(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_permute_dims():
    """Test `permute_dims`."""
    x = MyArray(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float))
    got = xp.permute_dims(x, (1, 0))
    expected = MyArray(jax_xp.permute_dims(x.array, (1, 0)))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_reshape():
    """Test `reshape`."""
    x = MyArray(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float))
    got = xp.reshape(x, (3, 2))
    expected = MyArray(jax_xp.reshape(x.array, (3, 2)))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_roll():
    """Test `roll`."""
    x = MyArray(xp.asarray([[1, 2, 3], [4, 5, 6]], dtype=float))
    got = xp.roll(x, shift=1, axis=0)
    expected = MyArray(jax_xp.roll(x.array, shift=1, axis=0))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_squeeze():
    """Test `squeeze`."""
    x = MyArray(xp.asarray([[[0], [1], [2]]], dtype=float))
    got = xp.squeeze(x, axis=(0, 2))
    expected = MyArray(jax_xp.squeeze(x.array, axis=(0, 2)))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_stack():
    """Test `stack`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    y = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.stack((x, y))
    expected = MyArray(jax_xp.stack((x.array, y.array)))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Searching functions


def test_argmax():
    """Test `argmax`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.argmax(x)
    expected = MyArray(jax_xp.argmax(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_argmin():
    """Test `argmin`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    got = xp.argmin(x)
    expected = MyArray(jax_xp.argmin(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


def test_nonzero():
    """Test `nonzero`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))
    (got,) = xp.nonzero(x)
    (expected,) = jax_xp.nonzero(x.array)

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected)


def test_where():
    """Test `where`."""
    condition = MyArray(xp.asarray([True, False, True]))
    y = MyArray(xp.asarray([1, 2, 3], dtype=float))
    z = MyArray(xp.asarray([4, 5, 6], dtype=float))
    got = xp.where(condition, y, z)
    expected = MyArray(jax_xp.where(condition.array, y.array, z.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Set functions


@pytest.mark.xfail(reason="value is not a MyArray")
def test_unique_all():
    """Test `unique_all`."""
    x = MyArray(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_all(x)
    expected = jax_xp.unique_all(x.array)

    assert isinstance(got, _UniqueAllResult)

    assert isinstance(got.values, MyArray)
    assert jnp.array_equal(got.values, expected.values)

    assert isinstance(got.inverse, MyArray)
    assert jnp.array_equal(got.inverse, expected.inverse)

    assert isinstance(got.inverse_indices, Array)
    assert jnp.array_equal(got.inverse_indices, expected.inverse_indices)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a MyArray")
def test_unique_counts():
    """Test `unique_counts`."""
    x = MyArray(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_counts(x)
    expected = jax_xp.unique_counts(x.array)

    assert isinstance(got, _UniqueCountsResult)

    assert isinstance(got.values, MyArray)
    assert jnp.array_equal(got.values.array, expected.values)

    assert isinstance(got.counts, Array)
    assert jnp.array_equal(got.counts, expected.counts)


@pytest.mark.xfail(reason="value is not a MyArray")
def test_unique_inverse():
    """Test `unique_inverse`."""
    x = MyArray(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_inverse(x)
    expected = jax_xp.unique_inverse(x.array)

    assert isinstance(got, _UniqueInverseResult)

    assert isinstance(got.values, MyArray)
    assert jnp.array_equal(got.values.array, expected.values)

    assert isinstance(got.inverse, MyArray)
    assert jnp.array_equal(got.inverse.array, expected.inverse)


@pytest.mark.xfail(reason="value is not a MyArray")
def test_unique_values():
    """Test `unique_values`."""
    x = MyArray(xp.asarray([1, 2, 2, 3, 3, 3], dtype=float))
    got = xp.unique_values(x)
    expected = MyArray(jax_xp.unique_values(x.array))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


# =============================================================================
# Sorting functions


@pytest.mark.skip("TODO")
def test_argsort():
    """Test `argsort`."""


@pytest.mark.skip("TODO")
def test_sort():
    """Test `sort`."""


# =============================================================================
# Statistical functions


def test_cumulative_sum():
    """Test `cumulative_sum`."""
    x = MyArray(xp.asarray([1, 2, 3], dtype=float))

    # No arguments
    got = xp.cumulative_sum(x)
    expected = MyArray(xp.asarray([1, 3, 6], dtype=float))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)

    # axis
    got = xp.cumulative_sum(x, axis=0)
    expected = MyArray(xp.asarray([1, 3, 6], dtype=float))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)

    with pytest.raises(ValueError, match="axis 1"):
        _ = xp.cumulative_sum(x, axis=1)

    # dtype
    got = xp.cumulative_sum(x, dtype=int)
    expected = MyArray(xp.asarray([1, 3, 6], dtype=int))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)

    # initial
    got = xp.cumulative_sum(x, include_initial=True)
    expected = MyArray(xp.asarray([0, 1, 3, 6]))

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, expected.array)


@pytest.mark.skip("TODO")
def test_max():
    """Test `max`."""


@pytest.mark.skip("TODO")
def test_mean():
    """Test `mean`."""


@pytest.mark.skip("TODO")
def test_min():
    """Test `min`."""


@pytest.mark.skip("TODO")
def test_prod():
    """Test `prod`."""


@pytest.mark.skip("TODO")
def test_std():
    """Test `std`."""


@pytest.mark.skip("TODO")
def test_sum():
    """Test `sum`."""


@pytest.mark.skip("TODO")
def test_var():
    """Test `var`."""


# =============================================================================
# Utility functions


@pytest.mark.skip("TODO")
def test_all():
    """Test `all`."""


@pytest.mark.skip("TODO")
def test_any():
    """Test `any`."""


# =============================================================================
# FFT


@pytest.mark.skip("TODO")
def test_fft_fft():
    """Test `fft.fft`."""


@pytest.mark.skip("TODO")
def test_fft_ifft():
    """Test `fft.ifft`."""


@pytest.mark.skip("TODO")
def test_fft_fftn():
    """Test `fft.fftn`."""


@pytest.mark.skip("TODO")
def test_fft_ifftn():
    """Test `fft.ifftn`."""


@pytest.mark.skip("TODO")
def test_fft_rfft():
    """Test `fft.rfft`."""


@pytest.mark.skip("TODO")
def test_fft_irfft():
    """Test `fft.irfft`."""


@pytest.mark.skip("TODO")
def test_fft_rfftn():
    """Test `fft.rfftn`."""


@pytest.mark.skip("TODO")
def test_fft_irfftn():
    """Test `fft.irfftn`."""


@pytest.mark.skip("TODO")
def test_fft_hfft():
    """Test `fft.hfft`."""


@pytest.mark.skip("TODO")
def test_fft_ihfft():
    """Test `fft.ihfft`."""


@pytest.mark.skip("TODO")
def test_fft_fftfreq():
    """Test `fft.fftfreq`."""


@pytest.mark.skip("TODO")
def test_fft_rfftfreq():
    """Test `fft.rfftfreq`."""


@pytest.mark.skip("TODO")
def test_fft_fftshift():
    """Test `fft.fftshift`."""


@pytest.mark.skip("TODO")
def test_fft_ifftshift():
    """Test `fft.ifftshift`."""


# =============================================================================
# Linalg


@pytest.mark.skip("TODO")
def test_linalg_cholesky():
    """Test `linalg.cholesky`."""


@pytest.mark.skip("TODO")
def test_linalg_cross():
    """Test `linalg.cross`."""


@pytest.mark.skip("TODO")
def test_linalg_det():
    """Test `linalg.det`."""


@pytest.mark.skip("TODO")
def test_linalg_diagonal():
    """Test `linalg.diagonal`."""


@pytest.mark.skip("TODO")
def test_linalg_eigh():
    """Test `linalg.eigh`."""


@pytest.mark.skip("TODO")
def test_linalg_eigvalsh():
    """Test `linalg.eigvalsh`."""


@pytest.mark.skip("TODO")
def test_linalg_inv():
    """Test `linalg.inv`."""


@pytest.mark.skip("TODO")
def test_linalg_matmul():
    """Test `linalg.matmul`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_norm():
    """Test `linalg.matrix_norm`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_power():
    """Test `linalg.matrix_power`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_rank():
    """Test `linalg.matrix_rank`."""


@pytest.mark.skip("TODO")
def test_linalg_matrix_transpose():
    """Test `linalg.matrix_transpose`."""


@pytest.mark.skip("TODO")
def test_linalg_outer():
    """Test `linalg.outer`."""


@pytest.mark.skip("TODO")
def test_linalg_pinv():
    """Test `linalg.pinv`."""


@pytest.mark.skip("TODO")
def test_linalg_qr():
    """Test `linalg.qr`."""


@pytest.mark.skip("TODO")
def test_linalg_slogdet():
    """Test `linalg.slogdet`."""


@pytest.mark.skip("TODO")
def test_linalg_solve():
    """Test `linalg.solve`."""


@pytest.mark.skip("TODO")
def test_linalg_svd():
    """Test `linalg.svd`."""


@pytest.mark.skip("TODO")
def test_linalg_svdvals():
    """Test `linalg.svdvals`."""


@pytest.mark.skip("TODO")
def test_linalg_tensordot():
    """Test `linalg.tensordot`."""


@pytest.mark.skip("TODO")
def test_linalg_trace():
    """Test `linalg.trace`."""


@pytest.mark.skip("TODO")
def test_linalg_vecdot():
    """Test `linalg.vecdot`."""


@pytest.mark.skip("TODO")
def test_linalg_vector_norm():
    """Test `linalg.vector_norm`."""
