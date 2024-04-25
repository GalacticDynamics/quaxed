"""Test with JAX inputs."""

import tempfile

import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jt
import numpy as np
import pytest

import quaxed.numpy as qnp


@pytest.fixture()
def x1():
    """Test input."""
    return jnp.array([1, 2, 3], dtype=float)


@pytest.fixture()
def x2():
    """Test input."""
    return jnp.array([4, 5, 6], dtype=float)


@pytest.fixture()
def xbool():
    """Test input."""
    return jnp.array([True, False, True], dtype=bool)


###############################################################################


@pytest.mark.parametrize("name", qnp._core._DIRECT_TRANSFER)  # noqa: SLF001
def test_direct_transfer(name):
    """Test direct transfers."""
    assert getattr(qnp, name) is getattr(jnp, name)


###############################################################################


def test_abs():
    """Test `quaxed.numpy.abs`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.abs(x) == jnp.abs(x))


def test_absolute():
    """Test `quaxed.numpy.absolute`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.absolute(x) == jnp.absolute(x))


def test_acos():
    """Test `quaxed.numpy.acos`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.acos(x) == jnp.acos(x))


def test_acosh():
    """Test `quaxed.numpy.acosh`."""
    x = jnp.array([1, 2, 3], dtype=float)
    assert jnp.all(qnp.acosh(x) == jnp.acosh(x))


def test_add(x1, x2):
    """Test `quaxed.numpy.add`."""
    assert jnp.all(qnp.add(x1, x2) == jnp.add(x1, x2))


def test_all(x1):
    """Test `quaxed.numpy.all`."""
    x = jnp.array([True, False, True], dtype=bool)
    assert qnp.all(x) == jnp.all(x)


def test_allclose(x1, x2):
    """Test `quaxed.numpy.allclose`."""
    x = jnp.array([1, 2, 3], dtype=float)
    y = jnp.array([1, 2, 3], dtype=float) + 1e-9
    assert qnp.allclose(x, y) == jnp.allclose(x, y)


def test_allclose(x1, x2):
    """Test `quaxed.numpy.allclose`."""
    assert qnp.allclose(x1, x2, atol=1e-8) == jnp.allclose(x1, x2, atol=1e-8)


def test_amax(x1):
    """Test `quaxed.numpy.amax`."""
    assert qnp.amax(x1) == jnp.amax(x1)


def test_amin(x1):
    """Test `quaxed.numpy.amin`."""
    assert qnp.amin(x1) == jnp.amin(x1)


def test_angle(x1):
    """Test `quaxed.numpy.angle`."""
    assert jnp.all(qnp.angle(x1) == jnp.angle(x1))


def test_array_equal(x1, x2):
    """Test `quaxed.numpy.array_equal`."""
    assert qnp.array_equal(x1, x2) == jnp.array_equal(x1, x2)


def test_any(x1):
    """Test `quaxed.numpy.any`."""
    x = jnp.array([True, False, True], dtype=bool)
    assert qnp.any(x) == jnp.any(x)


def test_append(x1):
    """Test `quaxed.numpy.append`."""
    assert jnp.all(qnp.append(x1, 4) == jnp.append(x1, 4))


def test_apply_along_axis(x1):
    """Test `quaxed.numpy.apply_along_axis`."""
    assert jnp.all(
        qnp.apply_along_axis(lambda x: x + 1, 0, x1)
        == jnp.apply_along_axis(lambda x: x + 1, 0, x1)
    )


def test_apply_over_axes(x1):
    """Test `quaxed.numpy.apply_over_axes`."""
    assert jnp.all(
        qnp.apply_over_axes(lambda x, _: x + 1, x1, [0, 1])
        == jnp.apply_over_axes(lambda x, _: x + 1, x1, [0, 1])
    )


def test_arange():
    """Test `quaxed.numpy.arange`."""
    assert jnp.all(qnp.arange(3) == jnp.arange(3))


def test_arccos():
    """Test `quaxed.numpy.arccos`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arccos(x) == jnp.arccos(x))


def test_arccosh():
    """Test `quaxed.numpy.arccosh`."""
    x = jnp.array([1, 2, 3], dtype=float)
    assert jnp.all(qnp.arccosh(x) == jnp.arccosh(x))


def test_arcsin():
    """Test `quaxed.numpy.arcsin`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arcsin(x) == jnp.arcsin(x))


def test_arcsinh():
    """Test `quaxed.numpy.arcsinh`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arcsinh(x) == jnp.arcsinh(x))


def test_arctan():
    """Test `quaxed.numpy.arctan`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arctan(x) == jnp.arctan(x))


def test_arctan2(x1, x2):
    """Test `quaxed.numpy.arctan2`."""
    assert jnp.all(qnp.arctan2(x1, x2) == jnp.arctan2(x1, x2))


def test_arctanh():
    """Test `quaxed.numpy.arctanh`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arctanh(x) == jnp.arctanh(x))


def test_argmax(x1):
    """Test `quaxed.numpy.argmax`."""
    assert qnp.argmax(x1) == jnp.argmax(x1)


def test_argmin(x1):
    """Test `quaxed.numpy.argmin`."""
    assert qnp.argmin(x1) == jnp.argmin(x1)


def test_argpartition(x1):
    """Test `quaxed.numpy.argpartition`."""
    assert jnp.all(qnp.argpartition(x1, 1) == jnp.argpartition(x1, 1))


def test_argsort(x1):
    """Test `quaxed.numpy.argsort`."""
    assert jnp.all(qnp.argsort(x1) == jnp.argsort(x1))


def test_argwhere(x1):
    """Test `quaxed.numpy.argwhere`."""
    assert jnp.all(qnp.argwhere(x1) == jnp.argwhere(x1))


def test_around(x1):
    """Test `quaxed.numpy.around`."""
    assert jnp.all(qnp.around(x1) == jnp.around(x1))


def test_array(x1):
    """Test `quaxed.numpy.array`."""
    assert jnp.all(qnp.array(x1) == jnp.array(x1))


def test_array_equal(x1, x2):
    """Test `quaxed.numpy.array_equal`."""
    assert qnp.array_equal(x1, x2) == jnp.array_equal(x1, x2)


def test_array_equiv(x1, x2):
    """Test `quaxed.numpy.array_equiv`."""
    assert qnp.array_equiv(x1, x2) == jnp.array_equiv(x1, x2)


@pytest.mark.xfail(reason="Not implemented.")
def test_array_str(x1):
    """Test `quaxed.numpy.array_str`."""
    assert qnp.array_str(x1) == jnp.array_str(x1)


def test_asarray():
    """Test `quaxed.numpy.asarray`."""
    x = [1.0, 2, 3, 4]
    assert jnp.all(qnp.asarray(x) == jnp.asarray(x))


def test_asin():
    """Test `quaxed.numpy.asin`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.asin(x) == jnp.asin(x))


def test_asinh():
    """Test `quaxed.numpy.asinh`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.asinh(x) == jnp.asinh(x))


def test_astype(x1):
    """Test `quaxed.numpy.astype`."""
    assert jnp.all(qnp.astype(x1, int) == jnp.astype(x1, int))


def test_atan():
    """Test `quaxed.numpy.atan`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arctan(x) == jnp.atan(x))


def test_atan2(x1, x2):
    """Test `quaxed.numpy.atan2`."""
    assert jnp.all(qnp.arctan2(x1, x2) == jnp.atan2(x1, x2))


def test_atanh():
    """Test `quaxed.numpy.atanh`."""
    x = jnp.array([-1, 0, 1], dtype=float)
    assert jnp.all(qnp.arctanh(x) == jnp.atanh(x))


def test_atleast_1d(x1):
    """Test `quaxed.numpy.atleast_1d`."""
    assert jnp.all(qnp.atleast_1d(x1) == jnp.atleast_1d(x1))


def test_atleast_2d(x1):
    """Test `quaxed.numpy.atleast_2d`."""
    assert jnp.all(qnp.atleast_2d(x1) == jnp.atleast_2d(x1))


def test_atleast_3d(x1):
    """Test `quaxed.numpy.atleast_3d`."""
    assert jnp.all(qnp.atleast_3d(x1) == jnp.atleast_3d(x1))


def test_average(x1):
    """Test `quaxed.numpy.average`."""
    assert qnp.average(x1) == jnp.average(x1)


def test_bartlett():
    """Test `quaxed.numpy.bartlett`."""
    assert jnp.all(qnp.bartlett(3) == jnp.bartlett(3))


def test_bincount():
    """Test `quaxed.numpy.bincount`."""
    x = jnp.asarray([0, 1, 1, 2, 2, 2])
    assert jnp.all(qnp.bincount(x) == jnp.bincount(x))


def test_bitwise_and(xbool):
    """Test `quaxed.numpy.bitwise_and`."""
    assert jnp.all(qnp.bitwise_and(xbool, xbool) == jnp.bitwise_and(xbool, xbool))


def test_bitwise_count(xbool):
    """Test `quaxed.numpy.bitwise_count`."""
    assert jnp.all(qnp.bitwise_count(xbool) == jnp.bitwise_count(xbool))


def test_bitwise_invert(xbool):
    """Test `quaxed.numpy.bitwise_invert`."""
    assert jnp.all(qnp.bitwise_invert(xbool) == jnp.bitwise_invert(xbool))


def test_bitwise_left_shift(xbool):
    """Test `quaxed.numpy.bitwise_left_shift`."""
    assert jnp.all(qnp.bitwise_left_shift(xbool, 1) == jnp.bitwise_left_shift(xbool, 1))


def test_bitwise_not(xbool):
    """Test `quaxed.numpy.bitwise_not`."""
    assert jnp.all(qnp.bitwise_not(xbool) == jnp.bitwise_not(xbool))


def test_bitwise_or(xbool):
    """Test `quaxed.numpy.bitwise_or`."""
    assert jnp.all(qnp.bitwise_or(xbool, xbool) == jnp.bitwise_or(xbool, xbool))


def test_bitwise_right_shift(xbool):
    """Test `quaxed.numpy.bitwise_right_shift`."""
    assert jnp.all(
        qnp.bitwise_right_shift(xbool, 1) == jnp.bitwise_right_shift(xbool, 1)
    )


def test_bitwise_xor(xbool):
    """Test `quaxed.numpy.bitwise_xor`."""
    assert jnp.all(qnp.bitwise_xor(xbool, xbool) == jnp.bitwise_xor(xbool, xbool))


def test_blackman():
    """Test `quaxed.numpy.blackman`."""
    assert jnp.all(qnp.blackman(3) == jnp.blackman(3))


def test_block(x1):
    """Test `quaxed.numpy.block`."""
    assert jnp.all(qnp.block([x1, x1]) == jnp.block([x1, x1]))


def test_bool():
    """Test `quaxed.numpy.bool`."""
    assert qnp.bool(1) == jnp.bool(1)


def test_bool_():
    """Test `quaxed.numpy.bool_`."""
    assert qnp.bool_(1) == jnp.bool_(1)


def test_broadcast_arrays(x1, x2):
    """Test `quaxed.numpy.broadcast_arrays`."""
    got1, got3 = qnp.broadcast_arrays(x1, x2)
    expected1, expected2 = jnp.broadcast_arrays(x1, x2)
    assert jnp.all(got1 == expected1)
    assert jnp.all(got3 == expected2)


def test_broadcast_shapes(x1, x2):
    """Test `quaxed.numpy.broadcast_shapes`."""
    assert jnp.all(
        qnp.broadcast_shapes(x1.shape, x2.shape)
        == jnp.broadcast_shapes(x1.shape, x2.shape)
    )


def test_broadcast_to(x1):
    """Test `quaxed.numpy.broadcast_to`."""
    assert jnp.all(qnp.broadcast_to(x1, (3, 3)) == jnp.broadcast_to(x1, (3, 3)))


@pytest.mark.xfail(reason="Not implemented.")
def test_c_():
    """Test `quaxed.numpy.c_`."""
    assert jnp.all(qnp.c_[1:3, 4:6] == jnp.c_[1:3, 4:6])


def test_can_cast(x1):
    """Test `quaxed.numpy.can_cast`."""
    assert qnp.can_cast(x1, int) == jnp.can_cast(x1, int)


def test_cbrt(x1):
    """Test `quaxed.numpy.cbrt`."""
    assert jnp.all(qnp.cbrt(x1) == jnp.cbrt(x1))


def test_ceil(x1):
    """Test `quaxed.numpy.ceil`."""
    assert jnp.all(qnp.ceil(x1) == jnp.ceil(x1))


def test_choose(x1):
    """Test `quaxed.numpy.choose`."""
    assert jnp.all(qnp.choose(0, [x1, x1]) == jnp.choose(0, [x1, x1]))


def test_clip(x1):
    """Test `quaxed.numpy.clip`."""
    assert jnp.all(qnp.clip(x1, 1, 2) == jnp.clip(x1, 1, 2))


def test_column_stack(x1):
    """Test `quaxed.numpy.column_stack`."""
    assert jnp.all(qnp.column_stack([x1, x1]) == jnp.column_stack([x1, x1]))


@pytest.mark.skip(reason="Not available.")
def test_complex128():
    """Test `quaxed.numpy.complex128`."""
    assert qnp.complex128(1) == jnp.complex128(1)


def test_complex64():
    """Test `quaxed.numpy.complex64`."""
    assert qnp.complex64(1) == jnp.complex64(1)


@pytest.mark.skip(reason="Not available.")
def test_complex_():
    """Test `quaxed.numpy.complex_`."""
    assert qnp.complex_(1) == jnp.complex_(1)


@pytest.mark.skip(reason="Not available.")
def test_complexfloating():
    """Test `quaxed.numpy.complexfloating`."""
    assert qnp.complexfloating(1) == jnp.complexfloating(1)


def test_compress(xbool, x1):
    """Test `quaxed.numpy.compress`."""
    assert jnp.all(qnp.compress(xbool, x1) == jnp.compress(xbool, x1))


def test_concat(x1):
    """Test `quaxed.numpy.concat`."""
    assert jnp.all(qnp.concat([x1, x1]) == jnp.concat([x1, x1]))


def test_concatenate(x1):
    """Test `quaxed.numpy.concatenate`."""
    assert jnp.all(qnp.concatenate([x1, x1]) == jnp.concatenate([x1, x1]))


def test_conj(x1):
    """Test `quaxed.numpy.conj`."""
    assert jnp.all(qnp.conj(x1) == jnp.conj(x1))


def test_conjugate(x1):
    """Test `quaxed.numpy.conjugate`."""
    assert jnp.all(qnp.conjugate(x1) == jnp.conjugate(x1))


def test_convolve(x1, x2):
    """Test `quaxed.numpy.convolve`."""
    assert jnp.all(qnp.convolve(x1, x2) == jnp.convolve(x1, x2))


def test_copy(x1):
    """Test `quaxed.numpy.copy`."""
    assert jnp.all(qnp.copy(x1) == jnp.copy(x1))


def test_copysign(x1, x2):
    """Test `quaxed.numpy.copysign`."""
    assert jnp.all(qnp.copysign(x1, x2) == jnp.copysign(x1, x2))


def test_corrcoef(x1):
    """Test `quaxed.numpy.corrcoef`."""
    assert jnp.all(qnp.corrcoef(x1) == jnp.corrcoef(x1))


def test_correlate(x1, x2):
    """Test `quaxed.numpy.correlate`."""
    assert jnp.all(qnp.correlate(x1, x2) == jnp.correlate(x1, x2))


def test_cos(x1):
    """Test `quaxed.numpy.cos`."""
    assert jnp.all(qnp.cos(x1) == jnp.cos(x1))


def test_cosh(x1):
    """Test `quaxed.numpy.cosh`."""
    assert jnp.all(qnp.cosh(x1) == jnp.cosh(x1))


def test_count_nonzero(x1):
    """Test `quaxed.numpy.count_nonzero`."""
    assert qnp.count_nonzero(x1) == jnp.count_nonzero(x1)


def test_cov(x1):
    """Test `quaxed.numpy.cov`."""
    assert jnp.all(qnp.cov(x1) == jnp.cov(x1))


def test_cross(x1, x2):
    """Test `quaxed.numpy.cross`."""
    assert jnp.all(qnp.cross(x1, x2) == jnp.cross(x1, x2))


def test_csingle():
    """Test `quaxed.numpy.csingle`."""
    assert qnp.csingle(1) == jnp.csingle(1)


def test_cumprod(x1):
    """Test `quaxed.numpy.cumprod`."""
    assert jnp.all(qnp.cumprod(x1) == jnp.cumprod(x1))


def test_cumsum(x1):
    """Test `quaxed.numpy.cumsum`."""
    assert jnp.all(qnp.cumsum(x1) == jnp.cumsum(x1))


def test_deg2rad(x1):
    """Test `quaxed.numpy.deg2rad`."""
    assert jnp.all(qnp.deg2rad(x1) == jnp.deg2rad(x1))


def test_degrees(x1):
    """Test `quaxed.numpy.degrees`."""
    assert jnp.all(qnp.degrees(x1) == jnp.degrees(x1))


def test_delete(x1):
    """Test `quaxed.numpy.delete`."""
    assert jnp.all(qnp.delete(x1, 1) == jnp.delete(x1, 1))


def test_diag(x1):
    """Test `quaxed.numpy.diag`."""
    assert jnp.all(qnp.diag(x1) == jnp.diag(x1))


def test_diag_indices():
    """Test `quaxed.numpy.diag_indices`."""
    got1, got2 = qnp.diag_indices(5)
    exp1, exp2 = jnp.diag_indices(5)
    assert jnp.all(got1 == exp1)
    assert jnp.all(got2 == exp2)


def test_diag_indices_from():
    """Test `quaxed.numpy.diag_indices_from`."""
    x = jnp.eye(4)
    got1, got2 = qnp.diag_indices_from(x)
    exp1, exp2 = jnp.diag_indices_from(x)
    assert jnp.all(got1 == exp1)
    assert jnp.all(got2 == exp2)


def test_diagflat(x1):
    """Test `quaxed.numpy.diagflat`."""
    assert jnp.all(qnp.diagflat(x1) == jnp.diagflat(x1))


def test_diagonal():
    """Test `quaxed.numpy.diagonal`."""
    x = jnp.eye(4)
    assert jnp.all(qnp.diagonal(x) == jnp.diagonal(x))


def test_diff(x1):
    """Test `quaxed.numpy.diff`."""
    assert jnp.all(qnp.diff(x1) == jnp.diff(x1))


def test_digitize(x1):
    """Test `quaxed.numpy.digitize`."""
    bins = jnp.asarray([0, 2, 4])
    assert jnp.all(qnp.digitize(x1, bins) == jnp.digitize(x1, bins))


def test_divide(x1, x2):
    """Test `quaxed.numpy.divide`."""
    assert jnp.all(qnp.divide(x1, x2) == jnp.divide(x1, x2))


def test_divmod(x1):
    """Test `quaxed.numpy.divmod`."""
    got1, got2 = qnp.divmod(x1, 2)
    exp1, exp2 = jnp.divmod(x1, 2)
    assert jnp.all(got1 == exp1)
    assert jnp.all(got2 == exp2)


def test_dot(x1, x2):
    """Test `quaxed.numpy.dot`."""
    assert jnp.all(qnp.dot(x1, x2) == jnp.dot(x1, x2))


@pytest.mark.skip(reason="no float64.")
def test_double(x1):
    """Test `quaxed.numpy.double`."""
    assert jnp.all(qnp.double(x1) == jnp.double(x1))


def test_dsplit():
    """Test `quaxed.numpy.dsplit`."""
    x = jnp.arange(16.0).reshape(2, 2, 4)
    got1, got2 = qnp.dsplit(x, 2)
    exp1, exp2 = jnp.dsplit(x, 2)
    assert jnp.all(got1 == exp1)
    assert jnp.all(got2 == exp2)


def test_dstack(x1):
    """Test `quaxed.numpy.dstack`."""
    assert jnp.all(qnp.dstack(x1) == jnp.dstack(x1))


def test_dtype(x1):
    """Test `quaxed.numpy.dtype`."""
    assert jnp.all(qnp.dtype(x1) == jnp.dtype(x1))


def test_ediff1d(x1):
    """Test `quaxed.numpy.ediff1d`."""
    assert jnp.all(qnp.ediff1d(x1) == jnp.ediff1d(x1))


def test_einsum(x1):
    """Test `quaxed.numpy.einsum`."""
    x = jnp.arange(25).reshape(5, 5)
    assert jnp.all(qnp.einsum("ii", x) == jnp.einsum("ii", x))


def test_einsum_path(x1):
    """Test `quaxed.numpy.einsum_path`."""
    key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
    a = jr.uniform(key1, (2, 2))
    b = jr.uniform(key2, (2, 5))
    c = jr.uniform(key3, (5, 2))

    got = qnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")
    exp = jnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")

    assert jnp.array_equal(got[0], exp[0])


def test_empty():
    """Test `quaxed.numpy.empty`."""
    assert jnp.all(qnp.empty(4) == jnp.empty(4))


def test_empty_like(x1):
    """Test `quaxed.numpy.empty_like`."""
    assert jnp.all(qnp.empty_like(x1) == jnp.empty_like(x1))


def test_equal(x1, x2):
    """Test `quaxed.numpy.equal`."""
    assert jnp.all(qnp.equal(x1, x2) == jnp.equal(x1, x2))


def test_euler_gamma(x1):
    """Test `quaxed.numpy.euler_gamma`."""
    assert qnp.euler_gamma == jnp.euler_gamma


def test_exp(x1):
    """Test `quaxed.numpy.exp`."""
    assert jnp.all(qnp.exp(x1) == jnp.exp(x1))


def test_exp2(x1):
    """Test `quaxed.numpy.exp2`."""
    assert jnp.all(qnp.exp2(x1) == jnp.exp2(x1))


def test_expand_dims(x1):
    """Test `quaxed.numpy.expand_dims`."""
    assert jnp.all(qnp.expand_dims(x1, 0) == jnp.expand_dims(x1, 0))


def test_expm1(x1):
    """Test `quaxed.numpy.expm1`."""
    assert jnp.all(qnp.expm1(x1) == jnp.expm1(x1))


def test_extract(x1):
    """Test `quaxed.numpy.extract`."""
    condition = jnp.array([True])
    assert jnp.all(qnp.extract(condition, x1) == jnp.extract(condition, x1))


def test_eye():
    """Test `quaxed.numpy.eye`."""
    assert jnp.all(qnp.eye(2) == jnp.eye(2))


def test_fabs(x1):
    """Test `quaxed.numpy.fabs`."""
    assert jnp.all(qnp.fabs(x1) == jnp.fabs(x1))


def test_fill_diagonal(x1):
    """Test `quaxed.numpy.fill_diagonal`."""
    x = jnp.eye(3)
    assert jnp.all(
        qnp.fill_diagonal(x, 2, inplace=False) == jnp.fill_diagonal(x, 2, inplace=False)
    )


def test_finfo(x1):
    """Test `quaxed.numpy.finfo`."""
    assert jnp.all(qnp.finfo(x1) == jnp.finfo(x1))


def test_fix(x1):
    """Test `quaxed.numpy.fix`."""
    assert jnp.all(qnp.fix(x1) == jnp.fix(x1))


def test_flatnonzero(x1):
    """Test `quaxed.numpy.flatnonzero`."""
    assert jnp.all(qnp.flatnonzero(x1) == jnp.flatnonzero(x1))


def test_flip(x1):
    """Test `quaxed.numpy.flip`."""
    assert jnp.all(qnp.flip(x1) == jnp.flip(x1))


def test_fliplr(x1):
    """Test `quaxed.numpy.fliplr`."""
    x = jnp.arange(4).reshape(2, 2)
    assert jnp.all(qnp.fliplr(x) == jnp.fliplr(x))


def test_flipud(x1):
    """Test `quaxed.numpy.flipud`."""
    assert jnp.all(qnp.flipud(x1) == jnp.flipud(x1))


def test_float16(x1):
    """Test `quaxed.numpy.float16`."""
    assert jnp.all(qnp.float16(x1) == jnp.float16(x1))


def test_float32(x1):
    """Test `quaxed.numpy.float32`."""
    assert jnp.all(qnp.float32(x1) == jnp.float32(x1))


@pytest.mark.skip(reason="not always available.")
def test_float64(x1):
    """Test `quaxed.numpy.float64`."""
    assert jnp.all(qnp.float64(x1) == jnp.float64(x1))


def test_float8_e4m3b11fnuz(x1):
    """Test `quaxed.numpy.float8_e4m3b11fnuz`."""
    assert jnp.all(qnp.float8_e4m3b11fnuz(x1) == jnp.float8_e4m3b11fnuz(x1))


def test_float8_e4m3fn(x1):
    """Test `quaxed.numpy.float8_e4m3fn`."""
    assert jnp.all(qnp.float8_e4m3fn(x1) == jnp.float8_e4m3fn(x1))


def test_float8_e4m3fnuz(x1):
    """Test `quaxed.numpy.float8_e4m3fnuz`."""
    assert jnp.all(qnp.float8_e4m3fnuz(x1) == jnp.float8_e4m3fnuz(x1))


def test_float8_e5m2(x1):
    """Test `quaxed.numpy.float8_e5m2`."""
    assert jnp.all(qnp.float8_e5m2(x1) == jnp.float8_e5m2(x1))


def test_float8_e5m2fnuz(x1):
    """Test `quaxed.numpy.float8_e5m2fnuz`."""
    assert jnp.all(qnp.float8_e5m2fnuz(x1) == jnp.float8_e5m2fnuz(x1))


@pytest.mark.skip(reason="not always available.")
def test_float_(x1):
    """Test `quaxed.numpy.float_`."""
    assert jnp.all(qnp.float_(x1) == jnp.float_(x1))


def test_float_power(x1, x2):
    """Test `quaxed.numpy.float_power`."""
    assert jnp.all(qnp.float_power(x1, x2) == jnp.float_power(x1, x2))


def test_floor(x1):
    """Test `quaxed.numpy.floor`."""
    assert jnp.all(qnp.floor(x1) == jnp.floor(x1))


def test_floor_divide(x1, x2):
    """Test `quaxed.numpy.floor_divide`."""
    assert jnp.all(qnp.floor_divide(x2, x1) == jnp.floor_divide(x2, x1))


def test_fmax(x1, x2):
    """Test `quaxed.numpy.fmax`."""
    assert jnp.all(qnp.fmax(x1, x2) == jnp.fmax(x1, x2))


def test_fmin(x1, x2):
    """Test `quaxed.numpy.fmin`."""
    assert jnp.all(qnp.fmin(x1, x2) == jnp.fmin(x1, x2))


def test_fmod(x1, x2):
    """Test `quaxed.numpy.fmod`."""
    assert jnp.all(qnp.fmod(x1, x2) == jnp.fmod(x1, x2))


def test_frexp(x1):
    """Test `quaxed.numpy.frexp`."""
    gotmantissa, gotexp = qnp.frexp(x1)
    expmantissa, expexp = jnp.frexp(x1)
    assert jnp.all(gotmantissa == expmantissa)
    assert jnp.all(gotexp == expexp)


def test_from_dlpack(x1):
    """Test `quaxed.numpy.from_dlpack`."""
    assert jnp.all(qnp.from_dlpack(x1) == jnp.from_dlpack(x1))


def test_frombuffer(x1):
    """Test `quaxed.numpy.frombuffer`."""
    assert jnp.all(
        qnp.frombuffer(b"\x01\x02", dtype=jnp.uint8)
        == jnp.frombuffer(b"\x01\x02", dtype=jnp.uint8)
    )


@pytest.mark.xfail(reason="JAX doesn't implement fromfile.")
def test_fromfile(x1):
    """Test `quaxed.numpy.fromfile`."""
    dt = np.dtype([("time", [("min", np.int64), ("sec", np.int64)]), ("temp", float)])
    x = np.zeros((1,), dtype=dt)
    x["time"]["min"] = 10
    x["temp"] = 98.25

    fname = tempfile.mkstemp()[1]
    x.tofile(fname)

    assert jnp.all(qnp.fromfile(x1, dtype=dt) == jnp.fromfile(x1, dtype=dt))


def test_fromfunction(x1):
    """Test `quaxed.numpy.fromfunction`."""

    def func(i, j):
        return i

    assert jnp.all(
        qnp.fromfunction(func, (2, 2), dtype=float)
        == jnp.fromfunction(func, (2, 2), dtype=float)
    )


@pytest.mark.xfail(reason="Not implemented.")
def test_fromiter():
    """Test `quaxed.numpy.fromiter`."""
    iterable = [x * x for x in range(5)]
    assert jnp.all(qnp.fromiter(iterable, float) == jnp.fromiter(iterable, float))


def test_frompyfunc(x1):
    """Test `quaxed.numpy.frompyfunc`."""
    assert jnp.all(
        qnp.frompyfunc(lambda x: x, 1, 1)(x1) == jnp.frompyfunc(lambda x: x, 1, 1)(x1)
    )


def test_fromstring():
    """Test `quaxed.numpy.fromstring`."""
    assert jnp.all(
        qnp.fromstring("1 2", dtype=int, sep=" ")
        == jnp.fromstring("1 2", dtype=int, sep=" ")
    )


def test_full():
    """Test `quaxed.numpy.full`."""
    assert jnp.all(qnp.full((2, 2), 4.0) == jnp.full((2, 2), 4.0))


def test_full_like(x1):
    """Test `quaxed.numpy.full_like`."""
    assert jnp.all(qnp.full_like(x1, 4.0) == jnp.full_like(x1, 4.0))


def test_gcd():
    """Test `quaxed.numpy.gcd`."""
    x1 = jnp.array([12, 8, 32])
    x2 = jnp.array([4, 4, 4])
    assert jnp.all(qnp.gcd(x1, x2) == jnp.gcd(x1, x2))


def test_geomspace():
    """Test `quaxed.numpy.geomspace`."""
    assert jnp.array_equal(qnp.geomspace(1.0, 100.0), jnp.geomspace(1.0, 100.0))


def test_get_printoptions():
    """Test `quaxed.numpy.get_printoptions`."""
    assert jnp.all(qnp.get_printoptions() == jnp.get_printoptions())


def test_gradient(x1):
    """Test `quaxed.numpy.gradient`."""
    assert jnp.all(qnp.gradient(x1) == jnp.gradient(x1))


def test_greater(x1, x2):
    """Test `quaxed.numpy.greater`."""
    assert jnp.all(qnp.greater(x1, x2) == jnp.greater(x1, x2))


def test_greater_equal(x1, x2):
    """Test `quaxed.numpy.greater_equal`."""
    assert jnp.all(qnp.greater_equal(x1, x2) == jnp.greater_equal(x1, x2))


def test_hamming():
    """Test `quaxed.numpy.hamming`."""
    assert jnp.all(qnp.hamming(2) == jnp.hamming(2))


def test_hanning():
    """Test `quaxed.numpy.hanning`."""
    assert jnp.all(qnp.hanning(2) == jnp.hanning(2))


def test_heaviside(x1, x2):
    """Test `quaxed.numpy.heaviside`."""
    assert jnp.all(qnp.heaviside(x1, x2) == jnp.heaviside(x1, x2))


def test_histogram(x1):
    """Test `quaxed.numpy.histogram`."""
    got = qnp.histogram(x1)
    exp = jnp.histogram(x1)
    assert all(jnp.all(g == e) for g, e in zip(got, exp, strict=True))


def test_histogram2d(x1, x2):
    """Test `quaxed.numpy.histogram2d`."""
    jnp.eye(3)
    got = qnp.histogram2d(x1, x2)
    exp = jnp.histogram2d(x1, x2)
    assert all(jnp.all(g == e) for g, e in zip(got, exp, strict=True))


def test_histogram_bin_edges(x1):
    """Test `quaxed.numpy.histogram_bin_edges`."""
    assert jnp.all(qnp.histogram_bin_edges(x1, 3) == jnp.histogram_bin_edges(x1, 3))


def test_histogramdd():
    """Test `quaxed.numpy.histogramdd`."""
    x = jnp.eye(3)
    got = qnp.histogramdd(x, bins=3)
    exp = jnp.histogramdd(x, bins=3)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=True))


def test_hsplit(x1):
    """Test `quaxed.numpy.hsplit`."""
    got = qnp.hsplit(x1, 3)
    exp = jnp.hsplit(x1, 3)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=True))


def test_hstack(x1):
    """Test `quaxed.numpy.hstack`."""
    assert jnp.all(qnp.hstack(x1) == jnp.hstack(x1))


def test_hypot(x1, x2):
    """Test `quaxed.numpy.hypot`."""
    assert jnp.all(qnp.hypot(x1, x2) == jnp.hypot(x1, x2))


def test_i0(x1):
    """Test `quaxed.numpy.i0`."""
    assert jnp.all(qnp.i0(x1) == jnp.i0(x1))


def test_identity():
    """Test `quaxed.numpy.identity`."""
    assert jnp.array_equal(qnp.identity(4), jnp.identity(4))


def test_iinfo(x1):
    """Test `quaxed.numpy.iinfo`."""
    x1 = x1.astype(int)
    got = qnp.iinfo(x1)
    expect = jnp.iinfo(x1)
    assert jt.tree_structure(got) == jt.tree_structure(expect)
    assert all(getattr(got, k) == getattr(expect, k) for k in ("min", "max", "dtype"))


def test_imag(x1):
    """Test `quaxed.numpy.imag`."""
    assert jnp.all(qnp.imag(x1) == jnp.imag(x1))


def test_inner(x1, x2):
    """Test `quaxed.numpy.inner`."""
    assert jnp.all(qnp.inner(x1, x2) == jnp.inner(x1, x2))


def test_insert(x1):
    """Test `quaxed.numpy.insert`."""
    assert jnp.all(qnp.insert(x1, 1, 2.0) == jnp.insert(x1, 1, 2.0))


def test_interp(x1):
    """Test `quaxed.numpy.interp`."""
    assert jnp.array_equal(qnp.interp(x1, x1, 2 * x1), jnp.interp(x1, x1, 2 * x1))


def test_intersect1d(x1):
    """Test `quaxed.numpy.intersect1d`."""
    assert jnp.array_equal(qnp.intersect1d(x1, x1), jnp.intersect1d(x1, x1))


def test_invert(x1):
    """Test `quaxed.numpy.invert`."""
    x1 = x1.astype(int)
    assert jnp.all(qnp.invert(x1) == jnp.invert(x1))


def test_isclose(x1, x2):
    """Test `quaxed.numpy.isclose`."""
    assert jnp.all(qnp.isclose(x1, x2) == jnp.isclose(x1, x2))


def test_iscomplex(x1):
    """Test `quaxed.numpy.iscomplex`."""
    assert jnp.all(qnp.iscomplex(x1) == jnp.iscomplex(x1))


def test_iscomplexobj(x1):
    """Test `quaxed.numpy.iscomplexobj`."""
    assert jnp.all(qnp.iscomplexobj(x1) == jnp.iscomplexobj(x1))


def test_isdtype(x1):
    """Test `quaxed.numpy.isdtype`."""
    assert jnp.all(qnp.isdtype(x1, "real floating") == jnp.isdtype(x1, "real floating"))


def test_isfinite(x1):
    """Test `quaxed.numpy.isfinite`."""
    assert jnp.all(qnp.isfinite(x1) == jnp.isfinite(x1))


def test_isin(x1):
    """Test `quaxed.numpy.isin`."""
    element = 2 * jnp.arange(4).reshape((2, 2))
    test_elements = jnp.asarray([1, 2, 4, 8])
    assert jnp.array_equal(
        qnp.isin(element, test_elements), jnp.isin(element, test_elements)
    )


def test_isinf(x1):
    """Test `quaxed.numpy.isinf`."""
    assert jnp.array_equal(qnp.isinf(x1), jnp.isinf(x1))


def test_isnan(x1):
    """Test `quaxed.numpy.isnan`."""
    assert jnp.array_equal(qnp.isnan(x1), jnp.isnan(x1))


def test_isneginf(x1):
    """Test `quaxed.numpy.isneginf`."""
    assert jnp.array_equal(qnp.isneginf(x1), jnp.isneginf(x1))


def test_isposinf(x1):
    """Test `quaxed.numpy.isposinf`."""
    assert jnp.array_equal(qnp.isposinf(x1), jnp.isposinf(x1))


def test_isreal(x1):
    """Test `quaxed.numpy.isreal`."""
    assert jnp.array_equal(qnp.isreal(x1), jnp.isreal(x1))


def test_isrealobj(x1):
    """Test `quaxed.numpy.isrealobj`."""
    assert jnp.array_equal(qnp.isrealobj(x1), jnp.isrealobj(x1))


def test_isscalar(x1):
    """Test `quaxed.numpy.isscalar`."""
    assert jnp.array_equal(qnp.isscalar(x1), jnp.isscalar(x1))


def test_issubdtype(x1):
    """Test `quaxed.numpy.issubdtype`."""
    assert jnp.all(
        qnp.issubdtype(x1.dtype, x1.dtype) == jnp.issubdtype(x1.dtype, x1.dtype)
    )


def test_iterable(x1):
    """Test `quaxed.numpy.iterable`."""
    assert jnp.all(qnp.iterable(x1) == jnp.iterable(x1))


def test_ix_(x1):
    """Test `quaxed.numpy.ix_`."""
    assert jnp.array_equal(qnp.ix_(x1), jnp.ix_(x1))


def test_kaiser():
    """Test `quaxed.numpy.kaiser`."""
    M = 4  # noqa: N806
    beta = jnp.asarray(3.0)
    assert jnp.array_equal(qnp.kaiser(M, beta), jnp.kaiser(M, beta))


def test_kron(x1):
    """Test `quaxed.numpy.kron`."""
    assert jnp.array_equal(qnp.kron(x1, x1), jnp.kron(x1, x1))


def test_lcm(x1, x2):
    """Test `quaxed.numpy.lcm`."""
    x1 = x1.astype(int)
    x2 = x2.astype(int)
    assert jnp.array_equal(qnp.lcm(x1, x2), jnp.lcm(x1, x2))


def test_ldexp(x1, x2):
    """Test `quaxed.numpy.ldexp`."""
    x2 = x2.astype(int)
    assert jnp.array_equal(qnp.ldexp(x1, x2), jnp.ldexp(x1, x2))


def test_left_shift(x1, x2):
    """Test `quaxed.numpy.left_shift`."""
    x1 = x1.astype(int)
    x2 = x2.astype(int)
    assert jnp.array_equal(qnp.left_shift(x1, x2), jnp.left_shift(x1, x2))


def test_less(x1, x2):
    """Test `quaxed.numpy.less`."""
    assert jnp.array_equal(qnp.less(x1, x2), jnp.less(x1, x2))


def test_less_equal(x1, x2):
    """Test `quaxed.numpy.less_equal`."""
    assert jnp.array_equal(qnp.less_equal(x1, x2), jnp.less_equal(x1, x2))


def test_lexsort(x1):
    """Test `quaxed.numpy.lexsort`."""
    assert jnp.array_equal(qnp.lexsort(x1), jnp.lexsort(x1))


def test_linspace():
    """Test `quaxed.numpy.linspace`."""
    start = 2.0
    stop = 3.0
    num = 5
    assert jnp.all(qnp.linspace(start, stop, num) == jnp.linspace(start, stop, num))


def test_load(tmp_path, x1):
    """Test `quaxed.numpy.load`."""
    path = tmp_path / "test.npy"
    jnp.save(path, x1)
    assert jnp.array_equal(qnp.load(path), jnp.load(path))


def test_log(x1):
    """Test `quaxed.numpy.log`."""
    assert jnp.array_equal(qnp.log(x1), jnp.log(x1))


def test_log10(x1):
    """Test `quaxed.numpy.log10`."""
    assert jnp.array_equal(qnp.log10(x1), jnp.log10(x1))


def test_log1p(x1):
    """Test `quaxed.numpy.log1p`."""
    assert jnp.array_equal(qnp.log1p(x1), jnp.log1p(x1))


def test_log2(x1):
    """Test `quaxed.numpy.log2`."""
    assert jnp.array_equal(qnp.log2(x1), jnp.log2(x1))


def test_logaddexp(x1, x2):
    """Test `quaxed.numpy.logaddexp`."""
    assert jnp.array_equal(qnp.logaddexp(x1, x2), jnp.logaddexp(x1, x2))


def test_logaddexp2(x1, x2):
    """Test `quaxed.numpy.logaddexp2`."""
    assert jnp.array_equal(qnp.logaddexp2(x1, x2), jnp.logaddexp2(x1, x2))


def test_logical_and(x1, x2):
    """Test `quaxed.numpy.logical_and`."""
    assert jnp.array_equal(qnp.logical_and(x1, x2), jnp.logical_and(x1, x2))


def test_logical_not(x1):
    """Test `quaxed.numpy.logical_not`."""
    assert jnp.array_equal(qnp.logical_not(x1), jnp.logical_not(x1))


def test_logical_or(x1, x2):
    """Test `quaxed.numpy.logical_or`."""
    assert jnp.array_equal(qnp.logical_or(x1, x2), jnp.logical_or(x1, x2))


def test_logical_xor(x1, x2):
    """Test `quaxed.numpy.logical_xor`."""
    assert jnp.array_equal(qnp.logical_xor(x1, x2), jnp.logical_xor(x1, x2))


def test_logspace():
    """Test `quaxed.numpy.logspace`."""
    start, stop = 0.0, 10.0
    assert jnp.array_equal(qnp.logspace(start, stop), jnp.logspace(start, stop))


def test_matmul(x1, x2):
    """Test `quaxed.numpy.matmul`."""
    assert jnp.array_equal(qnp.matmul(x1, x2), jnp.matmul(x1, x2))


def test_matrix_transpose(x1):
    """Test `quaxed.numpy.matrix_transpose`."""
    x1 = x1[..., None]  # must be 2D
    assert jnp.array_equal(qnp.matrix_transpose(x1), jnp.matrix_transpose(x1))


def test_max(x1):
    """Test `quaxed.numpy.max`."""
    assert jnp.all(qnp.max(x1) == jnp.max(x1))


def test_maximum(x1, x2):
    """Test `quaxed.numpy.maximum`."""
    assert jnp.array_equal(qnp.maximum(x1, x2), jnp.maximum(x1, x2))


def test_mean(x1):
    """Test `quaxed.numpy.mean`."""
    assert qnp.mean(x1) == jnp.mean(x1)


def test_median(x1):
    """Test `quaxed.numpy.median`."""
    assert qnp.median(x1) == jnp.median(x1)


def test_meshgrid(x1, x2):
    """Test `quaxed.numpy.meshgrid`."""
    assert jnp.array_equal(qnp.meshgrid(x1, x2), jnp.meshgrid(x1, x2))


def test_min(x1):
    """Test `quaxed.numpy.min`."""
    assert qnp.min(x1) == jnp.min(x1)


def test_minimum(x1, x2):
    """Test `quaxed.numpy.minimum`."""
    assert jnp.array_equal(qnp.minimum(x1, x2), jnp.minimum(x1, x2))


def test_mod(x1):
    """Test `quaxed.numpy.mod`."""
    assert jnp.array_equal(qnp.mod(x1, 2), jnp.mod(x1, 2))


def test_modf(x1):
    """Test `quaxed.numpy.modf`."""
    got_frac, got_intg = qnp.modf(x1)
    exp_frac, exp_intg = jnp.modf(x1)
    assert jnp.array_equal(got_frac, exp_frac)
    assert jnp.array_equal(got_intg, exp_intg)


def test_moveaxis(x1):
    """Test `quaxed.numpy.moveaxis`."""
    assert jnp.array_equal(qnp.moveaxis(x1[None], 0, 1), jnp.moveaxis(x1[None], 0, 1))


def test_multiply(x1, x2):
    """Test `quaxed.numpy.multiply`."""
    assert jnp.array_equal(qnp.multiply(x1, x2), jnp.multiply(x1, x2))


def test_nan_to_num(x1):
    """Test `quaxed.numpy.nan_to_num`."""
    assert jnp.all(qnp.nan_to_num(x1) == jnp.nan_to_num(x1))


def test_nanargmax(x1):
    """Test `quaxed.numpy.nanargmax`."""
    assert jnp.all(qnp.nanargmax(x1) == jnp.nanargmax(x1))


def test_nanargmin(x1):
    """Test `quaxed.numpy.nanargmin`."""
    assert jnp.all(qnp.nanargmin(x1) == jnp.nanargmin(x1))


def test_nancumprod(x1):
    """Test `quaxed.numpy.nancumprod`."""
    assert jnp.all(qnp.nancumprod(x1) == jnp.nancumprod(x1))


def test_nancumsum(x1):
    """Test `quaxed.numpy.nancumsum`."""
    assert jnp.all(qnp.nancumsum(x1) == jnp.nancumsum(x1))


def test_nanmax(x1):
    """Test `quaxed.numpy.nanmax`."""
    assert jnp.all(qnp.nanmax(x1) == jnp.nanmax(x1))


def test_nanmean(x1):
    """Test `quaxed.numpy.nanmean`."""
    assert jnp.all(qnp.nanmean(x1) == jnp.nanmean(x1))


def test_nanmedian(x1):
    """Test `quaxed.numpy.nanmedian`."""
    assert jnp.all(qnp.nanmedian(x1) == jnp.nanmedian(x1))


def test_nanmin(x1):
    """Test `quaxed.numpy.nanmin`."""
    assert jnp.all(qnp.nanmin(x1) == jnp.nanmin(x1))


def test_nanpercentile(x1):
    """Test `quaxed.numpy.nanpercentile`."""
    assert jnp.array_equal(qnp.nanpercentile(x1, 50), jnp.nanpercentile(x1, 50))


def test_nanprod(x1):
    """Test `quaxed.numpy.nanprod`."""
    assert jnp.array_equal(qnp.nanprod(x1), jnp.nanprod(x1))


def test_nanquantile(x1):
    """Test `quaxed.numpy.nanquantile`."""
    assert jnp.array_equal(qnp.nanquantile(x1, 0.5), jnp.nanquantile(x1, 0.5))


def test_nanstd(x1):
    """Test `quaxed.numpy.nanstd`."""
    assert qnp.nanstd(x1) == jnp.nanstd(x1)


def test_nansum(x1):
    """Test `quaxed.numpy.nansum`."""
    assert qnp.nansum(x1) == jnp.nansum(x1)


def test_nanvar(x1):
    """Test `quaxed.numpy.nanvar`."""
    assert qnp.nanvar(x1) == jnp.nanvar(x1)


def test_ndim(x1):
    """Test `quaxed.numpy.ndim`."""
    assert qnp.ndim(x1) == jnp.ndim(x1)


def test_negative(x1):
    """Test `quaxed.numpy.negative`."""
    assert jnp.array_equal(qnp.negative(x1), jnp.negative(x1))


def test_nextafter(x1, x2):
    """Test `quaxed.numpy.nextafter`."""
    assert jnp.array_equal(qnp.nextafter(x1, x2), jnp.nextafter(x1, x2))


def test_nonzero(x1):
    """Test `quaxed.numpy.nonzero`."""
    assert jnp.array_equal(qnp.nonzero(x1), jnp.nonzero(x1))


def test_not_equal(x1, x2):
    """Test `quaxed.numpy.not_equal`."""
    assert jnp.array_equal(qnp.not_equal(x1, x2), jnp.not_equal(x1, x2))


def test_ones():
    """Test `quaxed.numpy.ones`."""
    assert jnp.array_equal(qnp.ones(5), jnp.ones(5))


def test_ones_like(x1):
    """Test `quaxed.numpy.ones_like`."""
    assert jnp.array_equal(qnp.ones_like(x1), jnp.ones_like(x1))


def test_outer(x1, x2):
    """Test `quaxed.numpy.outer`."""
    assert jnp.array_equal(qnp.outer(x1, x2), jnp.outer(x1, x2))


def test_packbits():
    """Test `quaxed.numpy.packbits`."""
    x = jnp.array([[[1, 0, 1], [0, 1, 0]]], dtype=jnp.uint8)
    assert jnp.array_equal(qnp.packbits(x), jnp.packbits(x))


def test_pad(x1):
    """Test `quaxed.numpy.pad`."""
    assert jnp.array_equal(qnp.pad(x1, 20), jnp.pad(x1, 20))


def test_partition(x1):
    """Test `quaxed.numpy.partition`."""
    assert jnp.all(qnp.partition(x1, 1) == jnp.partition(x1, 1))


def test_percentile(x1):
    """Test `quaxed.numpy.percentile`."""
    assert qnp.percentile(x1, 50) == jnp.percentile(x1, 50)


def test_permute_dims(x1):
    """Test `quaxed.numpy.permute_dims`."""
    x1 = x1[None]
    assert jnp.all(qnp.permute_dims(x1, (0, 1)) == jnp.permute_dims(x1, (0, 1)))


def test_piecewise(x1):
    """Test `quaxed.numpy.piecewise`."""
    got = qnp.piecewise(x1, [x1 < 0, x1 >= 0], [-1, 1])
    exp = jnp.piecewise(x1, [x1 < 0, x1 >= 0], [-1, 1])
    assert jnp.array_equal(got, exp)


def test_place(x1):
    """Test `quaxed.numpy.place`."""
    mask = x1 > qnp.mean(x1)
    assert jnp.array_equal(
        qnp.place(x1, mask, 0, inplace=False), jnp.place(x1, mask, 0, inplace=False)
    )


def test_poly(x1):
    """Test `quaxed.numpy.poly`."""
    assert jnp.all(qnp.poly(x1) == jnp.poly(x1))


@pytest.mark.skip("TODO")
def test_polyadd(x1):
    """Test `quaxed.numpy.polyadd`."""
    assert jnp.all(qnp.polyadd(x1) == jnp.polyadd(x1))


@pytest.mark.skip("TODO")
def test_polyder(x1):
    """Test `quaxed.numpy.polyder`."""
    assert jnp.all(qnp.polyder(x1) == jnp.polyder(x1))


@pytest.mark.skip("TODO")
def test_polydiv(x1):
    """Test `quaxed.numpy.polydiv`."""
    assert jnp.all(qnp.polydiv(x1) == jnp.polydiv(x1))


@pytest.mark.skip("TODO")
def test_polyfit(x1):
    """Test `quaxed.numpy.polyfit`."""
    assert jnp.all(qnp.polyfit(x1) == jnp.polyfit(x1))


@pytest.mark.skip("TODO")
def test_polyint(x1):
    """Test `quaxed.numpy.polyint`."""
    assert jnp.all(qnp.polyint(x1) == jnp.polyint(x1))


@pytest.mark.skip("TODO")
def test_polymul(x1):
    """Test `quaxed.numpy.polymul`."""
    assert jnp.all(qnp.polymul(x1) == jnp.polymul(x1))


@pytest.mark.skip("TODO")
def test_polysub(x1):
    """Test `quaxed.numpy.polysub`."""
    assert jnp.all(qnp.polysub(x1) == jnp.polysub(x1))


@pytest.mark.skip("TODO")
def test_polyval(x1):
    """Test `quaxed.numpy.polyval`."""
    assert jnp.all(qnp.polyval(x1) == jnp.polyval(x1))


def test_positive(x1):
    """Test `quaxed.numpy.positive`."""
    assert jnp.array_equal(qnp.positive(x1), jnp.positive(x1))


def test_pow(x1):
    """Test `quaxed.numpy.pow`."""
    assert jnp.array_equal(qnp.pow(x1, 2), jnp.pow(x1, 2))


def test_power(x1):
    """Test `quaxed.numpy.power`."""
    assert jnp.array_equal(qnp.power(x1, 2.5), jnp.power(x1, 2.5))


def test_prod(x1):
    """Test `quaxed.numpy.prod`."""
    assert qnp.prod(x1) == jnp.prod(x1)


def test_ptp(x1):
    """Test `quaxed.numpy.ptp`."""
    assert jnp.all(qnp.ptp(x1) == jnp.ptp(x1))


@pytest.mark.skip("TODO")
def test_put(x1):
    """Test `quaxed.numpy.put`."""
    assert jnp.all(qnp.put(x1) == jnp.put(x1))


def test_quantile(x1):
    """Test `quaxed.numpy.quantile`."""
    assert qnp.quantile(x1, 0.5) == jnp.quantile(x1, 0.5)


def test_r_(x1, x2):
    """Test `quaxed.numpy.r_`."""
    assert jnp.all(qnp.r_[x1, x2] == jnp.r_[x1, x2])


def test_rad2deg(x1):
    """Test `quaxed.numpy.rad2deg`."""
    assert jnp.array_equal(qnp.rad2deg(x1), jnp.rad2deg(x1))


def test_radians(x1):
    """Test `quaxed.numpy.radians`."""
    assert jnp.all(qnp.radians(x1) == jnp.radians(x1))


def test_ravel(x1):
    """Test `quaxed.numpy.ravel`."""
    assert jnp.all(qnp.ravel(x1) == jnp.ravel(x1))


@pytest.mark.skip("TODO")
def test_ravel_multi_index(x1):
    """Test `quaxed.numpy.ravel_multi_index`."""
    assert jnp.all(qnp.ravel_multi_index(x1) == jnp.ravel_multi_index(x1))


def test_real(x1):
    """Test `quaxed.numpy.real`."""
    assert jnp.array_equal(qnp.real(x1), jnp.real(x1))


def test_reciprocal(x1):
    """Test `quaxed.numpy.reciprocal`."""
    assert jnp.array_equal(qnp.reciprocal(x1), jnp.reciprocal(x1))


def test_remainder(x1, x2):
    """Test `quaxed.numpy.remainder`."""
    assert jnp.all(qnp.remainder(x1, x2) == jnp.remainder(x1, x2))


def test_repeat(x1):
    """Test `quaxed.numpy.repeat`."""
    assert jnp.array_equal(qnp.repeat(x1, 3), jnp.repeat(x1, 3))


def test_reshape(x1):
    """Test `quaxed.numpy.reshape`."""
    assert jnp.array_equal(qnp.reshape(x1, (1, -1)), jnp.reshape(x1, (1, -1)))


def test_resize(x1):
    """Test `quaxed.numpy.resize`."""
    assert jnp.array_equal(qnp.resize(x1, (1, len(x1))), jnp.resize(x1, (1, len(x1))))


def test_result_type(x1):
    """Test `quaxed.numpy.result_type`."""
    assert qnp.result_type(3, x1) == jnp.result_type(2, x1)


def test_right_shift(xbool):
    """Test `quaxed.numpy.right_shift`."""
    assert jnp.array_equal(qnp.right_shift(xbool, xbool), jnp.right_shift(xbool, xbool))


def test_rint(x1):
    """Test `quaxed.numpy.rint`."""
    assert jnp.array_equal(qnp.rint(x1), jnp.rint(x1))


def test_roll(x1):
    """Test `quaxed.numpy.roll`."""
    assert jnp.array_equal(qnp.roll(x1, 4), jnp.roll(x1, 4))


def test_rollaxis(x1):
    """Test `quaxed.numpy.rollaxis`."""
    assert jnp.array_equal(qnp.rollaxis(x1, -1), jnp.rollaxis(x1, -1))


def test_roots(x1):
    """Test `quaxed.numpy.roots`."""
    assert jnp.array_equal(qnp.roots(x1), jnp.roots(x1))


def test_rot90(x1):
    """Test `quaxed.numpy.rot90`."""
    x1 = x1[None]
    assert jnp.array_equal(qnp.rot90(x1), jnp.rot90(x1))


def test_round(x1):
    """Test `quaxed.numpy.round`."""
    assert jnp.all(qnp.round(x1) == jnp.round(x1))


def test_round_(x1):
    """Test `quaxed.numpy.round_`."""
    assert jnp.all(qnp.round_(x1) == jnp.round_(x1))


@pytest.mark.skip("TODO")
def test_save(x1):
    """Test `quaxed.numpy.save`."""
    assert jnp.all(qnp.save(x1) == jnp.save(x1))


@pytest.mark.skip("TODO")
def test_savez(x1):
    """Test `quaxed.numpy.savez`."""
    assert jnp.all(qnp.savez(x1) == jnp.savez(x1))


@pytest.mark.skip("TODO")
def test_searchsorted(x1):
    """Test `quaxed.numpy.searchsorted`."""
    assert jnp.all(qnp.searchsorted(x1) == jnp.searchsorted(x1))


@pytest.mark.skip("TODO")
def test_select(x1):
    """Test `quaxed.numpy.select`."""
    assert jnp.all(qnp.select(x1) == jnp.select(x1))


def test_setdiff1d(x1, x2):
    """Test `quaxed.numpy.setdiff1d`."""
    assert jnp.array_equal(qnp.setdiff1d(x1, x2), jnp.setdiff1d(x1, x2))


def test_setxor1d(x1, x2):
    """Test `quaxed.numpy.setxor1d`."""
    assert jnp.array_equal(qnp.setxor1d(x1, x2), jnp.setxor1d(x1, x2))


def test_shape(x1):
    """Test `quaxed.numpy.shape`."""
    assert qnp.shape(x1) == jnp.shape(x1)


def test_sign(x1):
    """Test `quaxed.numpy.sign`."""
    assert jnp.array_equal(qnp.sign(x1), jnp.sign(x1))


def test_signbit(x1):
    """Test `quaxed.numpy.signbit`."""
    assert jnp.all(qnp.signbit(x1) == jnp.signbit(x1))


def test_sin(x1):
    """Test `quaxed.numpy.sin`."""
    assert jnp.array_equal(qnp.sin(x1), jnp.sin(x1))


def test_sinc(x1):
    """Test `quaxed.numpy.sinc`."""
    assert jnp.array_equal(qnp.sinc(x1), jnp.sinc(x1))


def test_sinh(x1):
    """Test `quaxed.numpy.sinh`."""
    assert jnp.array_equal(qnp.sinh(x1), jnp.sinh(x1))


def test_size(x1):
    """Test `quaxed.numpy.size`."""
    assert qnp.size(x1) == jnp.size(x1)


def test_sort(x1):
    """Test `quaxed.numpy.sort`."""
    assert jnp.array_equal(qnp.sort(x1), jnp.sort(x1))


def test_sort_complex(x1):
    """Test `quaxed.numpy.sort_complex`."""
    assert jnp.array_equal(qnp.sort_complex(x1), jnp.sort_complex(x1))


def test_split(x1):
    """Test `quaxed.numpy.split`."""
    got = qnp.split(x1, 3)
    exp = qnp.split(x1, 3)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=True))


def test_sqrt(x1):
    """Test `quaxed.numpy.sqrt`."""
    assert jnp.array_equal(qnp.sqrt(x1), jnp.sqrt(x1))


def test_square(x1):
    """Test `quaxed.numpy.square`."""
    assert jnp.array_equal(qnp.square(x1), jnp.square(x1))


def test_squeeze(x1):
    """Test `quaxed.numpy.squeeze`."""
    x1 = x1[None]
    assert jnp.array_equal(qnp.squeeze(x1), jnp.squeeze(x1))


def test_stack(x1, x2):
    """Test `quaxed.numpy.stack`."""
    assert jnp.array_equal(qnp.stack((x1, x2)), jnp.stack((x1, x2)))


def test_std(x1):
    """Test `quaxed.numpy.std`."""
    assert qnp.std(x1) == jnp.std(x1)


def test_subtract(x1, x2):
    """Test `quaxed.numpy.subtract`."""
    assert jnp.array_equal(qnp.subtract(x1, x2), jnp.subtract(x1, x2))


def test_sum(x1):
    """Test `quaxed.numpy.sum`."""
    assert qnp.sum(x1) == jnp.sum(x1)


def test_swapaxes(x1):
    """Test `quaxed.numpy.swapaxes`."""
    x1 = x1[None]
    assert jnp.all(qnp.swapaxes(x1, 0, 1) == jnp.swapaxes(x1, 0, 1))


@pytest.mark.skip("TODO")
def test_take(x1):
    """Test `quaxed.numpy.take`."""
    assert jnp.all(qnp.take(x1) == jnp.take(x1))


@pytest.mark.skip("TODO")
def test_take_along_axis(x1):
    """Test `quaxed.numpy.take_along_axis`."""
    assert jnp.all(qnp.take_along_axis(x1) == jnp.take_along_axis(x1))


def test_tan(x1):
    """Test `quaxed.numpy.tan`."""
    assert jnp.array_equal(qnp.tan(x1), jnp.tan(x1))


def test_tanh(x1):
    """Test `quaxed.numpy.tanh`."""
    assert jnp.array_equal(qnp.tanh(x1), jnp.tanh(x1))


@pytest.mark.skip("TODO")
def test_tensordot(x1, x2):
    """Test `quaxed.numpy.tensordot`."""
    assert jnp.array_equal(qnp.tensordot(x1, x2), jnp.tensordot(x1, x2))


def test_tile(x1):
    """Test `quaxed.numpy.tile`."""
    assert jnp.all(qnp.tile(x1, 3) == jnp.tile(x1, 3))


def test_trace(x1):
    """Test `quaxed.numpy.trace`."""
    x1 = x1[None]
    assert jnp.array_equal(qnp.trace(x1), jnp.trace(x1))


def test_transpose(x1):
    """Test `quaxed.numpy.transpose`."""
    x1 = x1[None]
    assert jnp.array_equal(qnp.transpose(x1), jnp.transpose(x1))


def test_tril():
    """Test `quaxed.numpy.tril`."""
    x = jnp.eye(4)
    assert jnp.array_equal(qnp.tril(x), jnp.tril(x))


def test_tril_indices_from():
    """Test `quaxed.numpy.tril_indices_from`."""
    x = jnp.eye(4)
    assert jnp.array_equal(qnp.tril_indices_from(x), jnp.tril_indices_from(x))


def test_trim_zeros(x1):
    """Test `quaxed.numpy.trim_zeros`."""
    x = jnp.concatenate((np.asarray([0.0, 0, 0]), x1, jnp.asarray([0.0, 0, 0])))
    assert jnp.array_equal(qnp.trim_zeros(x), jnp.trim_zeros(x))


def test_triu(x1):
    """Test `quaxed.numpy.triu`."""
    x = jnp.eye(4)
    assert jnp.array_equal(qnp.triu(x), jnp.triu(x))


def test_triu_indices_from():
    """Test `quaxed.numpy.triu_indices_from`."""
    x = jnp.eye(4)
    assert jnp.array_equal(qnp.triu_indices_from(x), jnp.triu_indices_from(x))


def test_true_divide(x1, x2):
    """Test `quaxed.numpy.true_divide`."""
    assert jnp.array_equal(qnp.true_divide(x1, x2), jnp.true_divide(x1, x2))


def test_trunc(x1):
    """Test `quaxed.numpy.trunc`."""
    assert jnp.array_equal(qnp.trunc(x1), jnp.trunc(x1))


@pytest.mark.skip("TODO")
def test_ufunc(x1):
    """Test `quaxed.numpy.ufunc`."""
    assert jnp.all(qnp.ufunc(x1) == jnp.ufunc(x1))


def test_union1d(x1, x2):
    """Test `quaxed.numpy.union1d`."""
    assert jnp.array_equal(qnp.union1d(x1, x2), jnp.union1d(x1, x2))


def test_unique(x1):
    """Test `quaxed.numpy.unique`."""
    assert jnp.array_equal(qnp.unique(x1), jnp.unique(x1))


def test_unique_all(x1):
    """Test `quaxed.numpy.unique_all`."""
    got = qnp.unique_all(x1)
    exp = jnp.unique_all(x1)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=True))


def test_unique_counts(x1):
    """Test `quaxed.numpy.unique_counts`."""
    assert jnp.array_equal(qnp.unique_counts(x1), jnp.unique_counts(x1))


def test_unique_inverse(x1):
    """Test `quaxed.numpy.unique_inverse`."""
    assert jnp.array_equal(qnp.unique_inverse(x1), jnp.unique_inverse(x1))


def test_unique_values(x1):
    """Test `quaxed.numpy.unique_values`."""
    assert jnp.array_equal(qnp.unique_values(x1), jnp.unique_values(x1))


@pytest.mark.skip("TODO")
def test_unpackbits(x1):
    """Test `quaxed.numpy.unpackbits`."""
    assert jnp.all(qnp.unpackbits(x1) == jnp.unpackbits(x1))


@pytest.mark.skip("TODO")
def test_unravel_index(x1):
    """Test `quaxed.numpy.unravel_index`."""
    assert jnp.all(qnp.unravel_index(x1) == jnp.unravel_index(x1))


def test_unwrap(x1):
    """Test `quaxed.numpy.unwrap`."""
    assert jnp.array_equal(qnp.unwrap(x1), jnp.unwrap(x1))


@pytest.mark.skip("TODO")
def test_vander(x1):
    """Test `quaxed.numpy.vander`."""
    assert jnp.all(qnp.vander(x1) == jnp.vander(x1))


def test_var(x1):
    """Test `quaxed.numpy.var`."""
    assert qnp.var(x1) == jnp.var(x1)


def test_vdot(x1, x2):
    """Test `quaxed.numpy.vdot`."""
    assert jnp.all(qnp.vdot(x1, x2) == jnp.vdot(x1, x2))


def test_vecdot(x1, x2):
    """Test `quaxed.numpy.vecdot`."""
    assert jnp.all(qnp.vecdot(x1, x2) == jnp.vecdot(x1, x2))


def test_vectorize(x1):
    """Test `quaxed.numpy.vectorize`."""

    @qnp.vectorize
    def f(x):
        return x + 1

    assert jnp.all(f(x1) == jnp.vectorize(lambda x: x + 1)(x1))


def test_vsplit(x1):
    """Test `quaxed.numpy.vsplit`."""
    got = qnp.vsplit(x1, 3)
    exp = jnp.vsplit(x1, 3)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=True))


def test_vstack(x1, x2):
    """Test `quaxed.numpy.vstack`."""
    assert jnp.array_equal(qnp.vstack((x1, x2)), jnp.vstack((x1, x2)))


def test_where(x1, x2):
    """Test `quaxed.numpy.where`."""
    cond = jnp.ones_like(x1, dtype=bool)
    assert jnp.array_equal(qnp.where(cond, x1, x2), jnp.where(cond, x1, x2))


def test_zeros(x1):
    """Test `quaxed.numpy.zeros`."""
    assert jnp.array_equal(qnp.zeros(4), jnp.zeros(4))


def test_zeros_like(x1):
    """Test `quaxed.numpy.zeros_like`."""
    assert jnp.array_equal(qnp.zeros_like(x1), jnp.zeros_like(x1))
