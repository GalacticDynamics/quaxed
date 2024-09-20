"""Quaxed :mod:`jax.numpy`."""
# pylint: disable=redefined-builtin

from typing import Any

from jaxtyping import install_import_hook

from . import _core, _creation_functions, _higher_order
from ._creation_functions import *
from ._higher_order import *

__all__: list[str] = []
__all__ += _core.__all__
__all__ += _higher_order.__all__
__all__ += _creation_functions.__all__


# TODO: consolidate with ``_core.__getattr__``.
def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_core, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# TODO: figure out how to install this import hook, with the __getattr__.
install_import_hook("quaxed.numpy", None)
