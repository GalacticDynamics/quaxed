"""Test with JAX inputs."""

import jax.numpy as jnp
import pytest
from jax.experimental import array_api as jax_xp

import quaxed.array_api as xp

# =============================================================================
# Constants


def test_e():
    """Test `e`."""
    assert xp.e is jax_xp.e


def test_inf():
    """Test `inf`."""
    assert xp.inf is jax_xp.inf


def test_nan():
    """Test `nan`."""
    assert xp.nan is jax_xp.nan


def test_newaxis():
    """Test `newaxis`."""
    assert xp.newaxis is jax_xp.newaxis


def test_pi():
    """Test `pi`."""
    assert xp.pi is jax_xp.pi


# =============================================================================
# Creation functions


@pytest.mark.parametrize(
    ("start", "stop", "step"),
    [
        # int
        (3, None, 1),
        (3, 1, 1),
        (4, None, 2),
        (3, 1, 2),
        # float
        (3.0, None, 1),
        (3.0, 1.0, 1),
        (4.0, None, 2.0),
        (3.0, 1.0, 2.0),
        # Array
        (jnp.array(3), None, 1),
        (jnp.array(3), jnp.array(1), 1),
        (jnp.array(4), None, jnp.array(2)),
        (jnp.array(3), jnp.array(1), jnp.array(2)),
        # TODO: mixed
    ],
)
def test_arange(start, stop, step):
    """Test `arange`."""
    # TODO: test the start, stop, step, dtype, device arguments
    got = xp.arange(start, stop, step)
    expected = jax_xp.arange(start, stop, step)

    assert isinstance(got, jnp.ndarray)
    assert jnp.array_equal(got, expected)


# def test_asarray():
#     """Test `asarray`."""
#     got = xp.asarray([1, 2, 3])
#     expected = jax_xp.asarray([1, 2, 3])

#     assert isinstance(got, jnp.ndarray)
#     assert jnp.array_equal(got, expected)
