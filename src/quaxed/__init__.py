"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Array-API JAX compatibility
"""

# pylint: disable=redefined-builtin


__all__ = ["__version__", "__array_api_version__", "array_api"]

from typing import Any

import plum
from jaxtyping import ArrayLike

from . import array_api

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
    return quaxify(getattr(jax, name))
