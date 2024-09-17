"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Pre-quaxed libraries for multiple dispatch over abstract array types in JAX
"""
# pylint: disable=C0415,W0621

from __future__ import annotations

from typing import TYPE_CHECKING

from . import _jax, lax, numpy, scipy
from ._jax import *
from ._setup import JAX_VERSION
from ._version import version as __version__  # noqa: F401

__all__ = ["lax", "numpy", "scipy"]
__all__ += _jax.__all__

if JAX_VERSION < (0, 4, 32):
    from . import array_api

    __all__ += ["array_api"]


if TYPE_CHECKING:
    from typing import Any


def __getattr__(name: str) -> Any:  # TODO: fuller annotation
    """Forward all other attribute accesses to Quaxified JAX."""
    import sys

    import jax
    from quax import quaxify

    # TODO: detect if the attribute is a function or a module.
    # If it is a function, quaxify it. If it is a module, return a proxy object
    # that quaxifies all of its attributes.
    out = quaxify(getattr(jax, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out


# Clean up the namespace
del TYPE_CHECKING
