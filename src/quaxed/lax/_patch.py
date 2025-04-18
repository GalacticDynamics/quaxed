"""Patches for `quax`.

TODO: remove this file when `quax` issues are fixed:

- https://github.com/issues/created?issue=patrick-kidger%7Cquax%7C57

"""

__all__: list[str] = []

from typing import Any

import quax
from jax import lax
from jaxtyping import Array, ArrayLike


@quax.register(lax.regularized_incomplete_beta_p)  # type: ignore[misc]
def regularized_incomplete_beta_p(
    a: ArrayLike,
    b: ArrayLike,
    x: ArrayLike,
) -> Array:
    """Patched implementation regularized incomplete beta function."""
    return lax.regularized_incomplete_beta_p.bind(a, b, x)


@quax.register(lax.scan_p)  # type: ignore[misc]
def scan_p(*args: ArrayLike, **kw: Any) -> Array:
    """Patched implementation of lax.map."""
    return lax.scan_p.bind(*args, **kw)
