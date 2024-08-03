"""Test with JAX inputs."""

import jax.numpy as jnp
import pytest
from jax import lax

import quaxed.lax as qlax


def test_dir():
    """Test the `__dir__` method."""
    assert set(qlax.__dir__()) == set(qlax.__all__)


def test_linalg_dir():
    """Test the `__dir__` method."""
    assert set(qlax.linalg.__dir__()) == set(qlax.linalg.__all__)


def test_not_in_lax():
    with pytest.raises(AttributeError, match="Cannot get"):
        _ = qlax.for_sure_not_in_lax


def test_not_in_lax_linalg():
    with pytest.raises(AttributeError, match="Cannot get"):
        _ = qlax.linalg.for_sure_not_in_lax_linalg


# ==============================================================================
# Operators


def test_abs():
    """Test `quaxed.lax.abs`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.abs(x)
    exp = lax.abs(x)

    assert jnp.array_equal(got, exp)


def test_acos():
    """Test `quaxed.lax.acos`."""
    x = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    got = qlax.acos(x)
    exp = lax.acos(x)

    assert jnp.array_equal(got, exp)


def test_acosh():
    """Test `quaxed.lax.acosh`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.acosh(x)
    exp = lax.acosh(x)

    assert jnp.array_equal(got, exp)


def test_add():
    """Test `quaxed.lax.add`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.add(x, y)
    exp = lax.add(x, y)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="Not implemented.")
def test_after_all():
    pass


def test_approx_max_k():
    """Test `quaxed.lax.approx_max_k`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.approx_max_k(x, 2)
    exp = lax.approx_max_k(x, 2)

    assert jnp.array_equal(got, exp)


def test_approx_min_k():
    """Test `quaxed.lax.approx_min_k`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.approx_min_k(x, 2)
    exp = lax.approx_min_k(x, 2)

    assert jnp.array_equal(got, exp)


def test_argmax():
    """Test `quaxed.lax.argmax`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.argmax(x, axis=0, index_dtype=int)
    exp = lax.argmax(x, axis=0, index_dtype=int)

    assert jnp.array_equal(got, exp)


def test_argmin():
    """Test `quaxed.lax.argmin`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.argmin(x, axis=0, index_dtype=int)
    exp = lax.argmin(x, axis=0, index_dtype=int)

    assert jnp.array_equal(got, exp)


def test_asin():
    """Test `quaxed.lax.asin`."""
    x = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    got = qlax.asin(x)
    exp = lax.asin(x)

    assert jnp.array_equal(got, exp)


def test_asinh():
    """Test `quaxed.lax.asinh`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.asinh(x)
    exp = lax.asinh(x)

    assert jnp.array_equal(got, exp)


def test_atan():
    """Test `quaxed.lax.atan`."""
    x = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    got = qlax.atan(x)
    exp = lax.atan(x)

    assert jnp.array_equal(got, exp)


def test_atan2():
    """Test `quaxed.lax.atan2`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.atan2(x, y)
    exp = lax.atan2(x, y)

    assert jnp.array_equal(got, exp)


def test_atanh():
    """Test `quaxed.lax.atanh`."""
    x = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
    got = qlax.atanh(x)
    exp = lax.atanh(x)

    assert jnp.array_equal(got, exp)


def test_batch_matmul():
    """Test `quaxed.lax.batch_matmul`."""
    x = jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
    y = jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=float)
    got = qlax.batch_matmul(x, y)
    exp = lax.batch_matmul(x, y)

    assert jnp.array_equal(got, exp)


def test_bessel_i0e():
    """Test `quaxed.lax.bessel_i0e`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.bessel_i0e(x)
    exp = lax.bessel_i0e(x)

    assert jnp.array_equal(got, exp)


def test_bessel_i1e():
    """Test `quaxed.lax.bessel_i1e`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.bessel_i1e(x)
    exp = lax.bessel_i1e(x)

    assert jnp.array_equal(got, exp)


def test_betainc():
    """Test `quaxed.lax.betainc`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    y = jnp.array([[5, 6], [7, 8]], dtype=float) / 10
    got = qlax.betainc(1.0, x, y)
    exp = lax.betainc(1.0, x, y)

    assert jnp.array_equal(got, exp)


def test_bitcast_convert_type():
    """Test `quaxed.lax.bitcast_convert_type`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.bitcast_convert_type(x, jnp.int32)
    exp = lax.bitcast_convert_type(x, jnp.int32)

    assert jnp.array_equal(got, exp)


def test_bitwise_and():
    """Test `quaxed.lax.bitwise_and`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.bitwise_and(x, x)
    exp = lax.bitwise_and(x, x)

    assert jnp.array_equal(got, exp)


def test_bitwise_not():
    """Test `quaxed.lax.bitwise_not`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.bitwise_not(x)
    exp = lax.bitwise_not(x)

    assert jnp.array_equal(got, exp)


def test_bitwise_or():
    """Test `quaxed.lax.bitwise_or`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.bitwise_or(x, x)
    exp = lax.bitwise_or(x, x)

    assert jnp.array_equal(got, exp)


def test_bitwise_xor():
    """Test `quaxed.lax.bitwise_xor`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.bitwise_xor(x, x)
    exp = lax.bitwise_xor(x, x)

    assert jnp.array_equal(got, exp)


def test_population_count():
    """Test `quaxed.lax.population_count`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.population_count(x)
    exp = lax.population_count(x)

    assert jnp.array_equal(got, exp)


def test_broadcast():
    """Test `quaxed.lax.broadcast`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.broadcast(x, (1, 1))
    exp = lax.broadcast(x, (1, 1))

    assert jnp.array_equal(got, exp)


def test_broadcast_in_dim():
    """Test `quaxed.lax.broadcast_in_dim`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.broadcast_in_dim(x, (1, 1, 2, 2), (2, 3))
    exp = lax.broadcast_in_dim(x, (1, 1, 2, 2), (2, 3))

    assert jnp.array_equal(got, exp)


def test_broadcast_shapes():
    """Test `quaxed.lax.broadcast_shapes`."""
    x = (2, 3)
    y = (1, 3)
    got = qlax.broadcast_shapes(x, y)
    exp = lax.broadcast_shapes(x, y)

    assert got == exp


def test_broadcast_to_rank():
    """Test `quaxed.lax.broadcast_to_rank`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.broadcast_to_rank(x, 3)
    exp = lax.broadcast_to_rank(x, 3)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="Not implemented.")
def test_broadcasted_iota():
    pass


def test_cbrt():
    """Test `quaxed.lax.cbrt`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cbrt(x)
    exp = lax.cbrt(x)

    assert jnp.array_equal(got, exp)


def test_ceil():
    """Test `quaxed.lax.ceil`."""
    x = jnp.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)
    got = qlax.ceil(x)
    exp = lax.ceil(x)

    assert jnp.array_equal(got, exp)


def test_clamp():
    """Test `quaxed.lax.clamp`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.clamp(2.0, x, 3.0)
    exp = lax.clamp(2.0, x, 3.0)

    assert jnp.array_equal(got, exp)


def test_clz():
    """Test `quaxed.lax.clz`."""
    x = jnp.array([[0, 2], [0, 4]], dtype=int)
    got = qlax.clz(x)
    exp = lax.clz(x)

    assert jnp.array_equal(got, exp)


def test_collapse():
    """Test `quaxed.lax.collapse`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.collapse(x, 1)
    exp = lax.collapse(x, 1)

    assert jnp.array_equal(got, exp)


def test_concatenate():
    """Test `quaxed.lax.concatenate`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.concatenate((x, y), 0)
    exp = lax.concatenate((x, y), 0)

    assert jnp.array_equal(got, exp)


def test_conj():
    """Test `quaxed.lax.conj`."""
    x = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=complex)
    got = qlax.conj(x)
    exp = lax.conj(x)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_conv():
    """Test `quaxed.lax.conv`."""


def test_convert_element_type():
    """Test `quaxed.lax.convert_element_type`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.convert_element_type(x, jnp.int32)
    exp = lax.convert_element_type(x, jnp.int32)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_dimension_numbers():
    """Test `quaxed.lax.conv_dimension_numbers`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_general_dilated():
    """Test `quaxed.lax.conv_general_dilated`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_general_dilated_local():
    """Test `quaxed.lax.conv_general_dilated_local`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_general_dilated_patches():
    """Test `quaxed.lax.conv_general_dilated_patches`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_transpose():
    """Test `quaxed.lax.conv_transpose`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_conv_with_general_padding():
    """Test `quaxed.lax.conv_with_general_padding`."""


def test_cos():
    """Test `quaxed.lax.cos`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cos(x)
    exp = lax.cos(x)

    assert jnp.array_equal(got, exp)


def test_cosh():
    """Test `quaxed.lax.cosh`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cosh(x)
    exp = lax.cosh(x)

    assert jnp.array_equal(got, exp)


def test_cumlogsumexp():
    """Test `quaxed.lax.cumlogsumexp`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cumlogsumexp(x, axis=0)
    exp = lax.cumlogsumexp(x, axis=0)

    assert jnp.array_equal(got, exp)


def test_cummax():
    """Test `quaxed.lax.cummax`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cummax(x, axis=0)
    exp = lax.cummax(x, axis=0)

    assert jnp.array_equal(got, exp)


def test_cummin():
    """Test `quaxed.lax.cummin`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cummin(x, axis=0)
    exp = lax.cummin(x, axis=0)

    assert jnp.array_equal(got, exp)


def test_cumprod():
    """Test `quaxed.lax.cumprod`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cumprod(x, axis=0)
    exp = lax.cumprod(x, axis=0)

    assert jnp.array_equal(got, exp)


def test_cumsum():
    """Test `quaxed.lax.cumsum`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.cumsum(x, axis=0)
    exp = lax.cumsum(x, axis=0)

    assert jnp.array_equal(got, exp)


def test_digamma():
    """Test `quaxed.lax.digamma`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.digamma(x)
    exp = lax.digamma(x)

    assert jnp.array_equal(got, exp)


def test_div():
    """Test `quaxed.lax.div`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.div(x, y)
    exp = lax.div(x, y)

    assert jnp.array_equal(got, exp)


def test_dot():
    """Test `quaxed.lax.dot`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.dot(x, y)
    exp = lax.dot(x, y)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_dot_general():
    """Test `quaxed.lax.dot_general`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_dynamic_index_in_dim():
    """Test `quaxed.lax.dynamic_index_in_dim`."""


def test_dynamic_slice():
    """Test `quaxed.lax.dynamic_slice`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.dynamic_slice(x, (0, 0), (2, 2))
    exp = lax.dynamic_slice(x, (0, 0), (2, 2))

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_dynamic_slice_in_dim():
    """Test `quaxed.lax.dynamic_slice_in_dim`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_dynamic_update_index_in_dim():
    """Test `quaxed.lax.dynamic_update_index_in_dim`."""


def test_dynamic_update_slice():
    """Test `quaxed.lax.dynamic_update_slice`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.dynamic_update_slice(x, y, (0, 0))
    exp = lax.dynamic_update_slice(x, y, (0, 0))

    assert jnp.array_equal(got, exp)


def test_dynamics_update_slice_in_dim():
    """Test `quaxed.lax.dynamic_update_slice_in_dim`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.dynamic_update_slice_in_dim(x, y, 0, 0)
    exp = lax.dynamic_update_slice_in_dim(x, y, 0, 0)

    assert jnp.array_equal(got, exp)


def test_eq():
    """Test `quaxed.lax.eq`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.eq(x, x)
    exp = lax.eq(x, x)

    assert jnp.array_equal(got, exp)


def test_erf():
    """Test `quaxed.lax.erf`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.erf(x)
    exp = lax.erf(x)

    assert jnp.array_equal(got, exp)


def test_erfc():
    """Test `quaxed.lax.erfc`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.erfc(x)
    exp = lax.erfc(x)

    assert jnp.array_equal(got, exp)


def test_erf_inv():
    """Test `quaxed.lax.erf_inv`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.erf_inv(x)
    exp = lax.erf_inv(x)

    assert jnp.array_equal(got, exp)


def test_exp():
    """Test `quaxed.lax.exp`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.exp(x)
    exp = lax.exp(x)

    assert jnp.array_equal(got, exp)


def test_expand_dims():
    """Test `quaxed.lax.expand_dims`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.expand_dims(x, (0,))
    exp = lax.expand_dims(x, (0,))

    assert jnp.array_equal(got, exp)


def test_expm1():
    """Test `quaxed.lax.expm1`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.expm1(x)
    exp = lax.expm1(x)

    assert jnp.array_equal(got, exp)


def test_fft():
    """Test `quaxed.lax.fft`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.fft(x, fft_type="fft", fft_lengths=(2, 2))
    exp = lax.fft(x, fft_type="fft", fft_lengths=(2, 2))

    assert jnp.array_equal(got, exp)


def test_floor():
    """Test `quaxed.lax.floor`."""
    x = jnp.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)
    got = qlax.floor(x)
    exp = lax.floor(x)

    assert jnp.array_equal(got, exp)


def test_full():
    """Test `quaxed.lax.full`."""
    got = qlax.full((2, 2), 1.0)
    exp = lax.full((2, 2), 1.0)

    assert jnp.array_equal(got, exp)


def test_full_like():
    """Test `quaxed.lax.full_like`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.full_like(x, 1.0)
    exp = lax.full_like(x, 1.0)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_gather():
    """Test `quaxed.lax.gather`."""


def test_ge():
    """Test `quaxed.lax.ge`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 2], [7, 2]], dtype=float)
    got = qlax.ge(x, y)
    exp = lax.ge(x, y)

    assert jnp.array_equal(got, exp)


def test_gt():
    """Test `quaxed.lax.gt`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 1], [7, 2]], dtype=float)
    got = qlax.gt(x, y)
    exp = lax.gt(x, y)

    assert jnp.array_equal(got, exp)


def test_igamma():
    """Test `quaxed.lax.igamma`."""
    a = 1.0
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.igamma(a, x)
    exp = lax.igamma(a, x)

    assert jnp.array_equal(got, exp)


def test_igammac():
    """Test `quaxed.lax.igammac`."""
    a = 1.0
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.igammac(a, x)
    exp = lax.igammac(a, x)

    assert jnp.array_equal(got, exp)


def test_imag():
    """Test `quaxed.lax.imag`."""
    x = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=complex)
    got = qlax.imag(x)
    exp = lax.imag(x)

    assert jnp.array_equal(got, exp)


def test_index_in_dim():
    """Test `quaxed.lax.index_in_dim`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.index_in_dim(x, 0, 0)
    exp = lax.index_in_dim(x, 0, 0)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_index_take():
    """Test `quaxed.lax.index_take`."""


def test_integer_pow():
    """Test `quaxed.lax.integer_pow`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.integer_pow(x, 2)
    exp = lax.integer_pow(x, 2)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_iota():
    """Test `quaxed.lax.iota`."""


def test_is_finite():
    """Test `quaxed.lax.is_finite`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.is_finite(x)
    exp = lax.is_finite(x)

    assert jnp.array_equal(got, exp)


def test_le():
    """Test `quaxed.lax.le`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 2], [7, 2]], dtype=float)
    got = qlax.le(x, y)
    exp = lax.le(x, y)

    assert jnp.array_equal(got, exp)


def test_lgamma():
    """Test `quaxed.lax.lgamma`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.lgamma(x)
    exp = lax.lgamma(x)

    assert jnp.array_equal(got, exp)


def test_log():
    """Test `quaxed.lax.log`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.log(x)
    exp = lax.log(x)

    assert jnp.array_equal(got, exp)


def test_log1p():
    """Test `quaxed.lax.log1p`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.log1p(x)
    exp = lax.log1p(x)

    assert jnp.array_equal(got, exp)


def test_logistic():
    """Test `quaxed.lax.logistic`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.logistic(x)
    exp = lax.logistic(x)

    assert jnp.array_equal(got, exp)


def test_lt():
    """Test `quaxed.lax.lt`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 1], [7, 2]], dtype=float)
    got = qlax.lt(x, y)
    exp = lax.lt(x, y)

    assert jnp.array_equal(got, exp)


def test_max():
    """Test `quaxed.lax.max`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.max(x, y)
    exp = lax.max(x, y)

    assert jnp.array_equal(got, exp)


def test_min():
    """Test `quaxed.lax.min`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.min(x, y)
    exp = lax.min(x, y)

    assert jnp.array_equal(got, exp)


def test_mul():
    """Test `quaxed.lax.mul`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.mul(x, y)
    exp = lax.mul(x, y)

    assert jnp.array_equal(got, exp)


def test_ne():
    """Test `quaxed.lax.ne`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 2], [7, 2]], dtype=float)
    got = qlax.ne(x, y)
    exp = lax.ne(x, y)

    assert jnp.array_equal(got, exp)


def test_neg():
    """Test `quaxed.lax.neg`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.neg(x)
    exp = lax.neg(x)

    assert jnp.array_equal(got, exp)


def test_nextafter():
    """Test `quaxed.lax.nextafter`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.nextafter(x, y)
    exp = lax.nextafter(x, y)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_pad():
    """Test `quaxed.lax.pad`."""


def test_polygamma():
    """Test `quaxed.lax.polygamma`."""
    m = 1.0
    x = jnp.array([[1, 2], [3, 4]], dtype=float) / 10
    got = qlax.polygamma(m, x)
    exp = lax.polygamma(m, x)

    assert jnp.array_equal(got, exp)


def test_population_count():
    """Test `quaxed.lax.population_count`."""
    x = jnp.array([[1, 0], [0, 1]], dtype=int)
    got = qlax.population_count(x)
    exp = lax.population_count(x)

    assert jnp.array_equal(got, exp)


def test_pow():
    """Test `quaxed.lax.pow`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.pow(x, y)
    exp = lax.pow(x, y)

    assert jnp.array_equal(got, exp)


def test_random_gamma_grad():
    """Test `quaxed.lax.random_gamma_grad`."""
    a = 1.0
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.random_gamma_grad(a, x)
    exp = lax.random_gamma_grad(a, x)

    assert jnp.array_equal(got, exp)


def test_real():
    """Test `quaxed.lax.real`."""
    x = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=complex)
    got = qlax.real(x)
    exp = lax.real(x)

    assert jnp.array_equal(got, exp)


def test_reciprocal():
    """Test `quaxed.lax.reciprocal`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.reciprocal(x)
    exp = lax.reciprocal(x)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_reduce():
    """Test `quaxed.lax.reduce`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_reduce_precision():
    """Test `quaxed.lax.reduce_precision`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_reduce_window():
    """Test `quaxed.lax.reduce_window`."""


def test_rem():
    """Test `quaxed.lax.rem`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.rem(x, y)
    exp = lax.rem(x, y)

    assert jnp.array_equal(got, exp)


def test_reshape():
    """Test `quaxed.lax.reshape`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.reshape(x, (1, 4))
    exp = lax.reshape(x, (1, 4))

    assert jnp.array_equal(got, exp)


def test_rev():
    """Test `quaxed.lax.rev`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.rev(x, dimensions=(0,))
    exp = lax.rev(x, dimensions=(0,))

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_rng_bit_generator():
    """Test `quaxed.lax.rng_bit_generator`."""


def test_rng_uniform():
    """Test `quaxed.lax.rng_uniform`."""
    a, b = 0, 1
    got = qlax.rng_uniform(a, b, (2, 3))
    exp = lax.rng_uniform(a, b, (2, 3))

    assert jnp.array_equal(got, exp)


def test_round():
    """Test `quaxed.lax.round`."""
    x = jnp.array([[1.1, 2.2], [3.3, 4.4]], dtype=float)
    got = qlax.round(x)
    exp = lax.round(x)

    assert jnp.array_equal(got, exp)


def test_rsqrt():
    """Test `quaxed.lax.rsqrt`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.rsqrt(x)
    exp = lax.rsqrt(x)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_scatter():
    """Test `quaxed.lax.scatter`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_scatter_apply():
    """Test `quaxed.lax.scatter_apply`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_scatter_max():
    """Test `quaxed.lax.scatter_max`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_scatter_min():
    """Test `quaxed.lax.scatter_min`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_scatter_mul():
    """Test `quaxed.lax.scatter_mul`."""


def test_shift_left():
    """Test `quaxed.lax.shift_left`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=int)
    got = qlax.shift_left(x, 1)
    exp = lax.shift_left(x, 1)

    assert jnp.array_equal(got, exp)


def test_shift_right_arithmetic():
    """Test `quaxed.lax.shift_right_arithmetic`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=int)
    got = qlax.shift_right_arithmetic(x, 1)
    exp = lax.shift_right_arithmetic(x, 1)

    assert jnp.array_equal(got, exp)


def test_shift_right_logical():
    """Test `quaxed.lax.shift_right_logical`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=int)
    got = qlax.shift_right_logical(x, 1)
    exp = lax.shift_right_logical(x, 1)

    assert jnp.array_equal(got, exp)


def test_sign():
    """Test `quaxed.lax.sign`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.sign(x)
    exp = lax.sign(x)

    assert jnp.array_equal(got, exp)


def test_sin():
    """Test `quaxed.lax.sin`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.sin(x)
    exp = lax.sin(x)

    assert jnp.array_equal(got, exp)


def test_sinh():
    """Test `quaxed.lax.sinh`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.sinh(x)
    exp = lax.sinh(x)

    assert jnp.array_equal(got, exp)


def test_slice():
    """Test `quaxed.lax.slice`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.slice(x, (0, 0), (2, 2))
    exp = lax.slice(x, (0, 0), (2, 2))

    assert jnp.array_equal(got, exp)


def test_slice_in_dim():
    """Test `quaxed.lax.slice_in_dim`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.slice_in_dim(x, 0, 0, 2)
    exp = lax.slice_in_dim(x, 0, 0, 2)

    assert jnp.array_equal(got, exp)


def test_sort():
    """Test `quaxed.lax.sort`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.sort(x)
    exp = lax.sort(x)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_sort_key_val():
    """Test `quaxed.lax.sort_key_val`."""


def test_sqrt():
    """Test `quaxed.lax.sqrt`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.sqrt(x)
    exp = lax.sqrt(x)

    assert jnp.array_equal(got, exp)


def test_square():
    """Test `quaxed.lax.square`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.square(x)
    exp = lax.square(x)

    assert jnp.array_equal(got, exp)


def test_sub():
    """Test `quaxed.lax.sub`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    y = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.sub(x, y)
    exp = lax.sub(x, y)

    assert jnp.array_equal(got, exp)


def test_tan():
    """Test `quaxed.lax.tan`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.tan(x)
    exp = lax.tan(x)

    assert jnp.array_equal(got, exp)


def test_tanh():
    """Test `quaxed.lax.tanh`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.tanh(x)
    exp = lax.tanh(x)

    assert jnp.array_equal(got, exp)


def test_top_k():
    """Test `quaxed.lax.top_k`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.top_k(x, 1)
    exp = lax.top_k(x, 1)

    assert jnp.array_equal(got, exp)


def test_transpose():
    """Test `quaxed.lax.transpose`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.transpose(x, (1, 0))
    exp = lax.transpose(x, (1, 0))

    assert jnp.array_equal(got, exp)


def test_zeros_like_array():
    """Test `quaxed.lax.zeros_like_array`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.zeros_like_array(x)
    exp = lax.zeros_like_array(x)

    assert jnp.array_equal(got, exp)


def test_zeta():
    """Test `quaxed.lax.zeta`."""
    x = jnp.array([[5, 2], [3, 4]], dtype=float)
    q = 2.0
    got = qlax.zeta(x, q)
    exp = lax.zeta(x, q)

    assert jnp.array_equal(got, exp)


# ==============================================================================
# Control flow operators


@pytest.mark.skip(reason="TODO: implement.")
def test_associative_scan():
    """Test `quaxed.lax.associative_scan`."""


def test_cond():
    """Test `quaxed.lax.cond`."""
    pred = True
    true_fun = lambda: jnp.array([[1, 2], [3, 4]], dtype=float)  # noqa: E731  # pylint: disable=C3001
    false_fun = lambda: jnp.array([[5, 6], [7, 8]], dtype=float)  # noqa: E731 # pylint: disable=C3001
    got = qlax.cond(pred, true_fun, false_fun)
    exp = lax.cond(pred, true_fun, false_fun)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_fori_loop():
    """Test `quaxed.lax.fori_loop`."""


def test_map():
    """Test `quaxed.lax.map`."""
    fun = lambda x: x + 1  # noqa: E731  # pylint: disable=C3001
    xs = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.map(fun, xs)
    exp = lax.map(fun, xs)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_scan():
    """Test `quaxed.lax.scan`."""


def test_select():
    """Test `quaxed.lax.select`."""
    pred = jnp.array([[True, False], [True, False]], dtype=bool)
    on_true = jnp.array([[1, 2], [3, 4]], dtype=float)
    on_false = jnp.array([[5, 6], [7, 8]], dtype=float)
    got = qlax.select(pred, on_true, on_false)
    exp = lax.select(pred, on_true, on_false)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_select_n():
    """Test `quaxed.lax.select_n`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_switch():
    """Test `quaxed.lax.switch`."""


def test_while_loop():
    """Test `quaxed.lax.while_loop`."""
    cond_fun = lambda x: jnp.all(x < 10)  # noqa: E731  # pylint: disable=C3001
    body_fun = lambda x: x + 1  # noqa: E731  # pylint: disable=C3001
    init_val = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.while_loop(cond_fun, body_fun, init_val)
    exp = lax.while_loop(cond_fun, body_fun, init_val)

    assert jnp.array_equal(got, exp)


# ==============================================================================
# Custom gradient operators


def test_stop_gradient():
    """Test `quaxed.lax.stop_gradient`."""
    x = jnp.array([[1, 2], [3, 4]], dtype=float)
    got = qlax.stop_gradient(x)
    exp = lax.stop_gradient(x)

    assert jnp.array_equal(got, exp)


@pytest.mark.skip(reason="TODO: implement.")
def test_custom_linear_solve():
    """Test `quaxed.lax.custom_linear_solve`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_custom_root():
    """Test `quaxed.lax.custom_root`."""


# ==============================================================================
# Parallel operators


@pytest.mark.skip(reason="TODO: implement.")
def test_all_gather():
    """Test `quaxed.lax.all_gather`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_all_to_all():
    """Test `quaxed.lax.all_to_all`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_psum():
    """Test `quaxed.lax.psum`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_psum_scatter():
    """Test `quaxed.lax.psum_scatter`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pmax():
    """Test `quaxed.lax.pmax`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pmin():
    """Test `quaxed.lax.pmin`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pmin():
    """Test `quaxed.lax.pmin`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pmean():
    """Test `quaxed.lax.pmean`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_ppermute():
    """Test `quaxed.lax.ppermute`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pshuffle():
    """Test `quaxed.lax.pshuffle`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_pswapaxes():
    """Test `quaxed.lax.pswapaxes`."""


@pytest.mark.skip(reason="TODO: implement.")
def test_axis_index():
    """Test `quaxed.lax.axis_index`."""


# ==============================================================================
# Sharding-related operators


@pytest.mark.skip(reason="TODO: implement.")
def test_with_sharding_constraint():
    """Test `quaxed.lax.with_sharding_constraint`."""


# ==============================================================================
# Linear algebra operators


def test_cholesky():
    """Test `quaxed.lax.cholesky`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got = qlax.linalg.cholesky(x)
    exp = lax.linalg.cholesky(x)

    assert jnp.array_equal(got, exp)


def test_eig():
    """Test `quaxed.lax.eig`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_w, got_vl, got_vr = lax.linalg.eig(x)
    exp_w, exp_vl, exp_vr = qlax.linalg.eig(x)

    assert jnp.array_equal(got_w, exp_w)
    assert jnp.array_equal(got_vl, exp_vl)
    assert jnp.array_equal(got_vr, exp_vr)


def test_eigh():
    """Test `quaxed.lax.eigh`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_w, got_v = lax.linalg.eigh(x)
    exp_w, exp_v = qlax.linalg.eigh(x)

    assert jnp.array_equal(got_w, exp_w)
    assert jnp.array_equal(got_v, exp_v)


def test_hessenberg():
    """Test `quaxed.lax.hessenberg`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_A, got_taus = lax.linalg.hessenberg(x)  # noqa: N806
    exp_A, exp_taus = qlax.linalg.hessenberg(x)  # noqa: N806

    assert jnp.array_equal(got_A, exp_A)
    assert jnp.array_equal(got_taus, exp_taus)


def test_lu():
    """Test `quaxed.lax.lu`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_lu, got_pivots, got_permutation = lax.linalg.lu(x)
    exp_lu, exp_pivots, exp_permutation = qlax.linalg.lu(x)

    assert jnp.array_equal(got_lu, exp_lu)
    assert jnp.array_equal(got_pivots, exp_pivots)
    assert jnp.array_equal(got_permutation, exp_permutation)


def test_householder_product():
    """Test `quaxed.lax.householder_product`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    A, taus = lax.linalg.hessenberg(x)  # noqa: N806
    got = qlax.linalg.householder_product(A, taus)
    exp = lax.linalg.householder_product(A, taus)

    assert jnp.array_equal(got, exp)


def test_qdwh():
    """Test `quaxed.lax.qdwh`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_u, got_h, got_num_iters, got_is_converged = lax.linalg.qdwh(x)
    exp_u, exp_h, exp_num_iters, exp_is_converged = qlax.linalg.qdwh(x)

    assert jnp.array_equal(got_u, exp_u)
    assert jnp.array_equal(got_h, exp_h)
    assert jnp.array_equal(got_num_iters, exp_num_iters)
    assert jnp.array_equal(got_is_converged, exp_is_converged)


def test_qr():
    """Test `quaxed.lax.linalg.qr`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got = qlax.linalg.qr(x)
    exp = lax.linalg.qr(x)

    assert jnp.array_equal(got, exp)


def test_schur():
    """Test `quaxed.lax.linalg.schur`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got = qlax.linalg.schur(x)
    exp = lax.linalg.schur(x)

    assert jnp.array_equal(got, exp)


def test_svd():
    """Test `quaxed.lax.svd`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_u, got_s, got_v = lax.linalg.svd(x)
    exp_u, exp_s, exp_v = qlax.linalg.svd(x)

    assert jnp.array_equal(got_u, exp_u)
    assert jnp.array_equal(got_s, exp_s)
    assert jnp.array_equal(got_v, exp_v)


def test_triangular_solve():
    """Test `quaxed.lax.triangular_solve`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    y = jnp.array([[1, 2], [2, 5]], dtype=float)
    got = qlax.linalg.triangular_solve(x, y)
    exp = lax.linalg.triangular_solve(x, y)

    assert jnp.array_equal(got, exp)


def test_tridiagonal():
    """Test `quaxed.lax.tridiagonal`."""
    x = jnp.array([[1, 2], [2, 5]], dtype=float)
    got_a, got_d, got_e, got_taus = qlax.linalg.tridiagonal(x)
    exp_a, exp_d, exp_e, exp_taus = lax.linalg.tridiagonal(x)

    assert jnp.array_equal(got_a, exp_a)
    assert jnp.array_equal(got_d, exp_d)
    assert jnp.array_equal(got_e, exp_e)
    assert jnp.array_equal(got_taus, exp_taus)


@pytest.mark.skip(reason="TODO: implement.")
def test_tridiagonal_solve():
    """Test `quaxed.lax.tridiagonal_solve`."""
