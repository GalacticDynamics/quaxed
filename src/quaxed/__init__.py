"""Pre-`quaxify`ed jax and related libraries.

`quax` is JAX + multiple dispatch + custom array-ish objects. `quaxed` is a
drop-in replacement for many JAX and related libraries that applies
`quax.quaxify` to the original JAX functions, enabling custom array-ish objects
to be used with those functions, not only jax arrays.

"""

__all__ = [
    # Modules
    "lax",
    "numpy",
    "scipy",
    "experimental",
    # Jax functions
    "device_put",
    "grad",
    "hessian",
    "jacfwd",
    "jacrev",
    "value_and_grad",
]

from typing import TYPE_CHECKING, Any

from . import experimental, lax, numpy, scipy
from ._jax import device_put, grad, hessian, jacfwd, jacrev, value_and_grad
from ._setup import JAX_VERSION
from ._version import version as __version__  # noqa: F401

if JAX_VERSION < (0, 4, 32):
    from . import array_api

    __all__ += ["array_api"]


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
del TYPE_CHECKING, Any
