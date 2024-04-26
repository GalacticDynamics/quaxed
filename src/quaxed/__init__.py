"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Array-API JAX compatibility
"""

# pylint: disable=redefined-builtin

__all__ = ["__version__", "array_api"]

import sys
from typing import Any

import plum
from jaxtyping import ArrayLike

from . import _jax, array_api
from ._jax import *
from ._version import version as __version__

__all__ += _jax.__all__

# Simplify the display of ArrayLike
plum.activate_union_aliases()
plum.set_union_alias(ArrayLike, "ArrayLike")


def __getattr__(name: str) -> Any:  # TODO: fuller annotation
    """Forward all other attribute accesses to Quaxified JAX."""
    import jax  # pylint: disable=C0415,W0621
    from quax import quaxify  # pylint: disable=C0415,W0621

    # TODO: detect if the attribute is a function or a module.
    # If it is a function, quaxify it. If it is a module, return a proxy object
    # that quaxifies all of its attributes.
    out = quaxify(getattr(jax, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
