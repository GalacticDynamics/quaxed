"""Quaxed `operator`.

This module wraps the functions in `operator` with `quax.quaxify`. The wrapping
happens dynamically through a module-level ``__dir__`` and ``__getattr__``. The
list of available functions is in ``__all__`` and documented in the built-in
`operator` library.

"""

import operator
import sys
from collections.abc import Callable
from typing import Any

from quax import quaxify

__all__ = operator.__all__


def __dir__() -> list[str]:
    """List the module contents."""
    return sorted(__all__)


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the :external:`quax.quaxify`'ed function."""
    # Quaxify the operator
    out = quaxify(getattr(operator, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
