"""Quaxed :mod:`jax.numpy`."""

# pylint: disable=redefined-builtin

from jaxtyping import install_import_hook

with install_import_hook("quaxed.numpy", None):
    from . import _core, _creation_functions, _dispatch, _higher_order
    from ._core import *  # TODO: make this lazy
    from ._creation_functions import *
    from ._dispatch import *
    from ._higher_order import *

__all__: list[str] = []
__all__ += _core.__all__
__all__ += _higher_order.__all__
__all__ += _creation_functions.__all__
__all__ += _dispatch.__all__
