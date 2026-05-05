"""Quaxed `jax.scipy.special`.

This module wraps the functions in `jax.scipy.special` with `quax.quaxify`. The
wrapping happens dynamically through a module-level ``__dir__`` and
``__getattr__``. The list of available functions is in ``__all__`` and
documented in the built-in `jax.scipy.special` library.

"""
# ruff:noqa: F822
# pyright: reportUnsupportedDunderAll=false
# pylint: disable=undefined-all-variable

__all__ = (
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
)

import sys
from collections.abc import Callable
from typing import Any

from jax.scipy import special as jsp
from quax import quaxify


def __dir__() -> list[str]:
    """List the module contents."""
    return ["__all__", *sorted(__all__)]


# TODO: better return type annotation
def __getattr__(name: str) -> Callable[..., Any]:
    if name not in __all__:
        msg = f"Cannot get {name} from quaxed.scipy.special."
        raise AttributeError(msg)

    # Quaxify the func
    func = quaxify(getattr(jsp, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, func)

    return func
