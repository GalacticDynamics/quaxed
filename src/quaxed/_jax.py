"""Quaxed :mod:`jax`."""

__all__ = [
    "device_put",
]

import jax
from quax import quaxify

device_put = quaxify(jax.device_put)
