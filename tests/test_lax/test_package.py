"""Test with JAX inputs."""

import pytest

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
