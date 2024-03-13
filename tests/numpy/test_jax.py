"""Test with JAX inputs."""

import jax.numpy as jnp
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


###############################################################################


def test_allclose(x1, x2):
    """Test `quaxed.numpy.allclose`."""
    assert qnp.allclose(x1, x2) == jnp.allclose(x1, x2)


def test_array_equal(x1, x2):
    """Test `quaxed.numpy.array_equal`."""
    assert qnp.array_equal(x1, x2) == jnp.array_equal(x1, x2)


def test_cbrt(x1):
    """Test `quaxed.numpy.cbrt`."""
    assert jnp.all(qnp.cbrt(x1) == jnp.cbrt(x1))


def test_copy(x1):
    """Test `quaxed.numpy.copy`."""
    assert jnp.all(qnp.copy(x1) == jnp.copy(x1))


def test_equal(x1, x2):
    """Test `quaxed.numpy.equal`."""
    assert jnp.all(qnp.equal(x1, x2) == jnp.equal(x1, x2))


def test_exp2(x1):
    """Test `quaxed.numpy.exp2`."""
    assert jnp.all(qnp.exp2(x1) == jnp.exp2(x1))


def test_greater(x1, x2):
    """Test `quaxed.numpy.greater`."""
    assert jnp.all(qnp.greater(x1, x2) == jnp.greater(x1, x2))


def test_hypot(x1, x2):
    """Test `quaxed.numpy.hypot`."""
    assert jnp.all(qnp.hypot(x1, x2) == jnp.hypot(x1, x2))


def test_matmul(x1, x2):
    """Test `quaxed.numpy.matmul`."""
    assert jnp.all(qnp.matmul(x1, x2) == jnp.matmul(x1, x2))


def test_moveaxis(x1):
    """Test `quaxed.numpy.moveaxis`."""
    assert jnp.all(qnp.moveaxis(x1[None], 0, 1) == jnp.moveaxis(x1[None], 0, 1))


def test_vectorize(x1):
    """Test `quaxed.numpy.vectorize`."""

    @qnp.vectorize
    def f(x):
        return x + 1

    assert jnp.all(f(x1) == jnp.vectorize(lambda x: x + 1)(x1))
