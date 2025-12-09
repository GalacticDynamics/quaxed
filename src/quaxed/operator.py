"""Quaxed `operator`.

This module wraps the functions in `operator` with `quax.quaxify`. The wrapping
happens dynamically through a module-level ``__dir__`` and ``__getattr__``. The
list of available functions is in ``__all__`` and documented in the built-in
`operator` library.

"""
# pyright: reportUnsupportedDunderAll=false
# pylint: disable=undefined-all-variable

__all__: tuple[str, ...] = (  # noqa: F822
    "abs",
    "add",
    "and_",
    "attrgetter",
    "call",
    "concat",
    "contains",
    "countOf",
    "delitem",
    "eq",
    "floordiv",
    "ge",
    "getitem",
    "gt",
    "index",
    "indexOf",
    "inv",
    "invert",
    "is_",
    "is_not",
    "itemgetter",
    "le",
    "length_hint",
    "lshift",
    "lt",
    "matmul",
    "methodcaller",
    "mod",
    "mul",
    "ne",
    "neg",
    "not_",
    "or_",
    "pos",
    "pow",
    "rshift",
    "setitem",
    "sub",
    "truediv",
    "truth",
    "xor",
)

import operator
import sys
from collections.abc import Callable
from operator import is_  # imported directly
from typing import Any

from quax import quaxify


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
