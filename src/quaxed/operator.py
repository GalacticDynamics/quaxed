"""Quaxed :mod:`operator`."""

import operator
from collections.abc import Callable
from typing import Any

from quax import quaxify

__all__ = operator.__all__


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the operator."""
    return quaxify(getattr(operator, name))


def __dir__() -> list[str]:
    """List the operators."""
    return sorted(__all__)
