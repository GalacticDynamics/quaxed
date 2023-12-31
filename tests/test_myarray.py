"""Test with :class:`quax.DenseArrayValue` inputs."""

import jax.experimental.array_api as jax_xp
from myarray import MyArray

import array_api_jax_compat as xp

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
    # assert jnp.array_equal(got, expected)
