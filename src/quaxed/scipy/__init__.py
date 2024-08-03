"""Quaxed :mod:`jax.scipy`.

This module wraps the functions in :external:`jax.scipy` with
:external:`quax.quaxify`.

"""

__all__ = ["linalg", "special"]

from . import linalg, special
