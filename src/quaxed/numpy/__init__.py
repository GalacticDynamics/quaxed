"""Quaxed :mod:`jax.numpy`."""

# pylint: disable=redefined-builtin

from jaxtyping import install_import_hook

with install_import_hook("quaxed", None):
    from . import _core
    from ._core import *

__all__: list[str] = []
__all__ += _core.__all__
