"""Quaxed :mod:`jax.lax`."""
# pylint: disable=undefined-all-variable

__all__ = [
    # ----- Operators -----
    "abs",
    "acos",
    "acosh",
    "add",
    # "after_all",
    "approx_max_k",
    "approx_min_k",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "batch_matmul",
    "bessel_i0e",
    "bessel_i1e",
    "betainc",
    "bitcast_convert_type",
    "bitwise_and",
    "bitwise_not",
    "bitwise_or",
    "bitwise_xor",
    "population_count",
    "broadcast",
    "broadcast_in_dim",
    "broadcast_shapes",
    "broadcast_to_rank",
    "broadcasted_iota",
    "cbrt",
    "ceil",
    "clamp",
    "clz",
    "collapse",
    "complex",
    "concatenate",
    "conj",
    "conv",
    "convert_element_type",
    "conv_dimension_numbers",
    "conv_general_dilated",
    "conv_general_dilated_local",
    "conv_general_dilated_patches",
    "conv_transpose",
    "conv_with_general_padding",
    "cos",
    "cosh",
    "cumlogsumexp",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "digamma",
    "div",
    "dot",
    "dot_general",
    "dynamic_index_in_dim",
    "dynamic_slice",
    "dynamic_slice_in_dim",
    "dynamic_update_index_in_dim",
    "dynamic_update_slice",
    "dynamic_update_slice_in_dim",
    "eq",
    "erf",
    "erfc",
    "erf_inv",
    "exp",
    "expand_dims",
    "expm1",
    "fft",
    "floor",
    "full",
    "full_like",
    "gather",
    "ge",
    "gt",
    "igamma",
    "igammac",
    "imag",
    "index_in_dim",
    "index_take",
    "integer_pow",
    "iota",
    "is_finite",
    "le",
    "lgamma",
    "log",
    "log1p",
    "logistic",
    "lt",
    "max",
    "min",
    "mul",
    "ne",
    "neg",
    "nextafter",
    "pad",
    "polygamma",
    "population_count",
    "pow",
    "random_gamma_grad",
    "real",
    "reciprocal",
    "reduce",
    "reduce_precision",
    "reduce_window",
    "rem",
    "reshape",
    "rev",
    "rng_bit_generator",
    "rng_uniform",
    "round",
    "rsqrt",
    "scatter",
    "scatter_add",
    "scatter_apply",
    "scatter_max",
    "scatter_min",
    "scatter_mul",
    "shift_left",
    "shift_right_arithmetic",
    "shift_right_logical",
    "sign",
    "sin",
    "sinh",
    "slice",
    "slice_in_dim",
    "sort",
    "sort_key_val",
    "sqrt",
    "square",
    "squeeze",
    "sub",
    "tan",
    "tanh",
    "top_k",
    "transpose",
    "zeros_like_array",
    "zeta",
    # ----- Control Flow Operators -----
    "associative_scan",
    "cond",
    "fori_loop",
    "map",
    "scan",
    "select",
    "select_n",
    "switch",
    "while_loop",
    # ----- Custom Gradient Operators -----
    "stop_gradient",
    "custom_linear_solve",
    "custom_root",
    # ----- Parallel Operators -----
    "all_gather",
    "all_to_all",
    "psum",
    "psum_scatter",
    "pmax",
    "pmin",
    "pmean",
    "ppermute",
    "pshuffle",
    "pswapaxes",
    "axis_index",
    # ----- Sharding-related Operators -----
    "with_sharding_constraint",
    # ----- Linear Algebra Operators -----
    "linalg",
    # ----- Argument classes -----
    "ConvDimensionNumbers",
    "ConvGeneralDilatedDimensionNumbers",
    "DotAlgorithm",
    "DotAlgorithmPreset",
    "FftType",
    "GatherDimensionNumbers",
    "GatherScatterMode",
    "Precision",
    "PrecisionLike",
    "RandomAlgorithm",
    "RoundingMethod",
    "ScatterDimensionNumbers",
]


import sys
from collections.abc import Callable
from typing import Any

from jax import lax
from quax import quaxify

from . import linalg

# Explicit imports that don't need to be quaxified
# isort: split
from jax.lax import (
    ConvDimensionNumbers,
    ConvGeneralDilatedDimensionNumbers,
    DotAlgorithm,
    DotAlgorithmPreset,
    FftType,
    GatherDimensionNumbers,
    GatherScatterMode,
    Precision,
    PrecisionLike,
    RandomAlgorithm,
    RoundingMethod,
    ScatterDimensionNumbers,
)


def __dir__() -> list[str]:
    """List the module contents."""
    return sorted(__all__)


# TODO: return type hint signature
def __getattr__(name: str) -> Callable[..., Any]:
    """Get the :external:`quax.quaxify`'ed function."""
    if name not in __all__:
        msg = f"Cannot get {name} from quaxed.lax."
        raise AttributeError(msg)

    # Quaxify the operator
    out = quaxify(getattr(lax, name))

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
