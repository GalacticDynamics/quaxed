"""Quaxed :mod:`operator`."""

import operator
import sys
from collections.abc import Callable
from typing import Any

from quax import quaxify

__all__ = operator.__all__


def __dir__() -> list[str]:
    """List the operators."""
    return sorted(__all__)


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the operator."""
    # Quaxify the operator
    out = quaxify(getattr(operator, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
