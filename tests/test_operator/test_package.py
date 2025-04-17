"""Test with JAX inputs."""

import operator as ops

import quaxed.operator as qops


def test_dir():
    """Test the __dir__ function."""
    assert set(qops.__dir__()) == set(qops.__all__)
    assert set(qops.__dir__()).issubset(set(dir(ops)))
