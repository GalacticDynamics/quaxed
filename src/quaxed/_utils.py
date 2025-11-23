"""Utility functions for quaxed."""

import quax


def quaxify[T](func: T, *, filter_spec: bool | tuple[bool, ...] = True) -> T:
    """Quaxify, but makes mypy happy."""
    return quax.quaxify(func, filter_spec=filter_spec)
