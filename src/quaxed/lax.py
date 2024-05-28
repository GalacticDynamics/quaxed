"""Quaxed :mod:`jax.lax`."""

__all__ = [
    "clamp",
    "complex",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "digamma",
    "erf",
    "erfc",
    "erf_inv",
    "eq",
    "select",
]


from jax import lax
from quax import quaxify

clamp = quaxify(lax.clamp)
complex = quaxify(lax.complex)
cumlogsumexp = quaxify(lax.cumlogsumexp)
cummax = quaxify(lax.cummax)
cummin = quaxify(lax.cummin)
cumprod = quaxify(lax.cumprod)
cumsum = quaxify(lax.cumsum)
digamma = quaxify(lax.digamma)
erf = quaxify(lax.erf)
erfc = quaxify(lax.erfc)
erf_inv = quaxify(lax.erf_inv)
eq = quaxify(lax.eq)
select = quaxify(lax.select)
