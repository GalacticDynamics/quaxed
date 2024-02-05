"""Test the package itself."""

import importlib.metadata

import array_api_jax_compat as m


def test_version() -> None:
    assert importlib.metadata.version("array_api_jax_compat") == m.__version__
