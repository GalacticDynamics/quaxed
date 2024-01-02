"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

array-api-jax-compat: Array-API JAX compatibility
"""

# pylint: disable=redefined-builtin


from __future__ import annotations

from typing import Any

from jaxtyping import install_import_hook

with install_import_hook("array_api_jax_compat", None):
    from . import _array_api, _grad
    from ._array_api import *
    from ._grad import *
    from ._version import version as __version__

__all__ = ["__version__"]
__all__ += _array_api.__all__
__all__ += _grad.__all__


def __getattr__(name: str) -> Any:  # TODO: fuller annotation
    """Forward all other attribute accesses to Quaxified JAX."""
    import jax  # pylint: disable=C0415,W0621
    from quax import quaxify  # pylint: disable=C0415,W0621

    # TODO: detect if the attribute is a function or a module.
    # If it is a function, quaxify it. If it is a module, return a proxy object
    # that quaxifies all of its attributes.
    return quaxify(getattr(jax, name))
