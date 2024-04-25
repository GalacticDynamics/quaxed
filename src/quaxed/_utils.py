from typing import TypeVar

import quax

T = TypeVar("T")


def quaxify(func: T, *, filter_spec: bool | tuple[bool, ...] = True) -> T:
    """Quaxify, but makes mypy happy."""
    return quax.quaxify(func, filter_spec=filter_spec)
