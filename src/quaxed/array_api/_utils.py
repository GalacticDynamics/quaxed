from typing import TypeVar

import quax

T = TypeVar("T")


def quaxify(func: T) -> T:
    """Quaxify, but makes mypy happy."""
    return quax.quaxify(func)
