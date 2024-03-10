"""Quaxed :mod:`jax.scipy.special`."""

__all__ = [  # noqa: F822
    "bernoulli",
    "betainc",
    "betaln",
    "beta",
    "bessel_jn",
    "digamma",
    "entr",
    "erf",
    "erfc",
    "erfinv",
    "exp1",
    "expi",
    "expit",
    "expn",
    "factorial",
    "gammainc",
    "gammaincc",
    "gammaln",
    "gamma",
    "i0",
    "i0e",
    "i1",
    "i1e",
    "logit",
    "logsumexp",
    "lpmn",
    "lpmn_values",
    "multigammaln",
    "log_ndtr",
    "ndtr",
    "ndtri",
    "polygamma",
    "spence",
    "sph_harm",
    "xlogy",
    "xlog1py",
    "zeta",
    "kl_div",
    "rel_entr",
    "poch",
    "hyp1f1",
]

import sys
from collections.abc import Callable
from typing import Any

from jax.scipy import special as jsp
from quax import quaxify


# TODO: better return type annotation
def __getattr__(name: str) -> Callable[..., Any]:
    func = quaxify(getattr(jsp, name))
    setattr(sys.modules[__name__], name, func)
    return func


def __dir__() -> list[str]:
    return sorted(__all__)
