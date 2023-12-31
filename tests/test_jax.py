"""Test with JAX inputs."""


from jax.experimental import array_api as jax_xp

import array_api_jax_compat as xp

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


# def test_arange():
#     """Test `arange`."""
#     # TODO: test the start, stop, step, dtype, device arguments
#     got = xp.arange(3)
#     expected = jax_xp.arange(3)

#     assert isinstance(got, jnp.ndarray)
#     assert jnp.array_equal(got, expected)


# def test_asarray():
#     """Test `asarray`."""
#     got = xp.asarray([1, 2, 3])
#     expected = jax_xp.asarray([1, 2, 3])

#     assert isinstance(got, jnp.ndarray)
#     assert jnp.array_equal(got, expected)
