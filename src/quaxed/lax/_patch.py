"""Patches for `quax`."""

__all__: tuple[str, ...] = ()

from typing import Any

import jax
import jax.extend.core as jexc
import jax.tree_util as jtu
import quax
from jax import lax
from jaxtyping import Array, ArrayLike


@quax.register(lax.scan_p)  # type: ignore[misc]
def scan_p(*args: ArrayLike, **kw: Any) -> Array:
    """Patched implementation of lax.map."""
    return lax.scan_p.bind(*args, **kw)


# =========================================================
# https://github.com/patrick-kidger/quax/pull/64

_sentinel = object()


@quax.register(lax.cond_p)  # type: ignore[misc]
def cond_quax(
    index: ArrayLike,
    *args: quax.ArrayValue | ArrayLike,
    branches: tuple[Any, ...],
    linear: Any = _sentinel,
    branches_platforms: Any = _sentinel,
) -> quax.ArrayValue:
    flat_args, in_tree = jtu.tree_flatten(args)

    out_trees = []
    quax_branches = []
    for jaxpr in branches:

        def flat_quax_call(flat_args: Any) -> Any:
            args = jtu.tree_unflatten(in_tree, flat_args)
            out = quax.quaxify(jexc.jaxpr_as_fun(jaxpr))(*args)  # noqa: B023
            flat_out, out_tree = jtu.tree_flatten(out)
            out_trees.append(out_tree)
            return flat_out

        quax_jaxpr = jax.make_jaxpr(flat_quax_call)(flat_args)
        quax_branches.append(quax_jaxpr)

    if any(tree_outs_i != out_trees[0] for tree_outs_i in out_trees[1:]):
        msg = "all branches output must have the same pytree."
        raise TypeError(msg)

    kwargs = {}
    if linear is not _sentinel:
        kwargs["linear"] = linear
    if branches_platforms is not _sentinel:
        kwargs["branches_platforms"] = branches_platforms

    out_val = jax.lax.cond_p.bind(
        index, *flat_args, branches=tuple(quax_branches), **kwargs
    )
    return jtu.tree_unflatten(out_trees[0], out_val)
