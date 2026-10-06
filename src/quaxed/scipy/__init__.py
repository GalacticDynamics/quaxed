# Copyright (c) 2023, Nathaniel Starkman.
"""Quaxed `jax.scipy`.

This module wraps the functions in `jax.scipy` with `quax.quaxify`.

"""

__all__ = ("linalg", "special")

from . import linalg, special
