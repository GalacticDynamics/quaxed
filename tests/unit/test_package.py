"""Test the package itself."""

import importlib.metadata

import quaxed as pkg


def test_version() -> None:
    assert importlib.metadata.version("quaxed") == pkg.__version__
