"""Test with JAX inputs."""

import jax
import jax.numpy as jnp

import quaxed


def test_svd():
    """Test `quaxed.scipy.linalg.svd`."""
    assert hasattr(jax.scipy.linalg, "svd")
    assert hasattr(quaxed.scipy.linalg, "svd")

    x = jnp.array([[1, 2], [3, 4]])
    got = quaxed.scipy.linalg.svd(x, compute_uv=False)
    expected = jax.scipy.linalg.svd(x, compute_uv=False)

    assert jnp.array_equal(got, expected)
