"""Quaxified `jax.numpy`.

This module wraps the functions in `jax.numpy` with `quax.quaxify`. The wrapping
happens dynamically through a module-level ``__dir__`` and ``__getattr__``. The
list of available functions is in ``__all__`` and documented in the `jax.numpy`
library.

In addition the following modules are supported:

- `quaxed.numpy.fft`
- `quaxed.numpy.linalg`

The contents of these modules are likewise dynamically wrapped with
`quax.quaxify` and their contents is listed in their respective ``__all__`` and
documented in their respective libraries.

If a function is missing, please file an Issue.

"""
# pylint: disable=redefined-builtin

from typing import Any

from jaxtyping import install_import_hook

from . import _core, _creation_functions, _higher_order, fft, linalg
from ._creation_functions import *
from ._higher_order import *

__all__ = (
    ["fft", "linalg"]  # noqa: RUF005
    + _core.__all__
    + _higher_order.__all__
    + _creation_functions.__all__
)


# TODO: consolidate with ``_core.__getattr__``.
def __getattr__(name: str) -> Any:
    if name in __all__:
        return getattr(_core, name)

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# TODO: figure out how to install this import hook, with the __getattr__.
install_import_hook("quaxed.numpy", None)
