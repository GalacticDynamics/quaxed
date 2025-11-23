"""Patches for `quax`."""

__all__: tuple[str, ...] = ()

from typing import Any

import quax
from jax import lax
from jaxtyping import Array, ArrayLike


@quax.register(lax.scan_p)  # type: ignore[misc]
def scan_p(*args: ArrayLike, **kw: Any) -> Array:
    """Patched implementation of lax.map."""
    return lax.scan_p.bind(*args, **kw)
