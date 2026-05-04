"""Test with JAX inputs."""

import jax.numpy as jnp
import pytest

import quaxed.numpy as qnp


@pytest.mark.parametrize("name", qnp._core._DIRECT_TRANSFER)  # noqa: SLF001
def test_direct_transfer(name):
    """Test direct transfers."""
    assert getattr(qnp, name) is getattr(jnp, name)


def test_linalg_dir():
    """Test the __dir__ function."""
    exp = set(qnp.linalg.__dir__())
    assert exp == set(qnp.linalg.__all__) | {"__all__"}
    assert (exp - {"__all__"}).issubset(set(dir(jnp.linalg)))
