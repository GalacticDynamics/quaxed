"""Test with JAX inputs."""

import pytest

import quaxed


def test_dir():
    """Test the `__dir__` method."""
    assert quaxed.scipy.linalg.__dir__() == quaxed.scipy.linalg.__all__


def test_not_in_scipy_special():
    """Test error message for non-members."""
    with pytest.raises(AttributeError, match="Cannot get"):
        _ = quaxed.scipy.special.not_a_member


def test_not_in_scipy_linalg():
    """Test error message for non-members."""
    with pytest.raises(AttributeError, match="Cannot get"):
        _ = quaxed.scipy.linalg.not_a_member
