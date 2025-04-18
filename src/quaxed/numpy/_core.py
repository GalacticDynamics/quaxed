# ruff: noqa: F822
"""Quaxed :mod:`jax.numpy`."""
# pylint: disable=undefined-all-variable

__all__ = [
    # modules
    "fft",
    "linalg",
    # contents
    "abs",
    "absolute",
    "acos",
    "acosh",
    "add",
    "all",
    "allclose",
    "amax",
    "amin",
    "angle",
    "any",
    "append",
    "apply_along_axis",
    "apply_over_axes",
    # "arange",  # in _creation_functions
    "arccos",
    "arccosh",
    "arcsin",
    "arcsinh",
    "arctan",
    "arctan2",
    "arctanh",
    "argmax",
    "argmin",
    "argpartition",
    "argsort",
    "argwhere",
    "around",
    "array",
    "array_equal",
    "array_equiv",
    "array_repr",
    "array_split",
    # "array_str",  # TODO:  why is this erroring?
    # "asarray",  # in _creation_functions
    "asin",
    "asinh",
    "astype",
    "atan",
    "atan2",
    "atanh",
    "atleast_1d",
    "atleast_2d",
    "atleast_3d",
    "average",
    "bartlett",
    "bfloat16",
    "bincount",
    "bitwise_and",
    "bitwise_count",
    "bitwise_invert",
    "bitwise_left_shift",
    "bitwise_not",
    "bitwise_or",
    "bitwise_right_shift",
    "bitwise_xor",
    "blackman",
    "block",
    "bool",
    "bool_",
    "broadcast_arrays",
    "broadcast_shapes",
    "broadcast_to",
    "c_",  # not subscriptable
    "can_cast",
    "cbrt",
    "cdouble",
    "ceil",
    "character",
    "choose",
    "clip",
    "column_stack",
    "complex128",
    "complex64",
    "complex_",
    "complexfloating",
    "compress",
    "concat",
    "concatenate",
    "conj",
    "conjugate",
    "convolve",
    "copy",
    "copysign",
    "corrcoef",
    "correlate",
    "cos",
    "cosh",
    "count_nonzero",
    "cov",
    "cross",
    "csingle",
    "cumprod",
    "cumsum",
    "deg2rad",
    "degrees",
    "delete",
    "diag",
    "diag_indices",
    "diag_indices_from",
    "diagflat",
    "diagonal",
    "diff",
    "digitize",
    "divide",
    "divmod",
    "dot",
    "double",
    "dsplit",
    "dstack",
    "dtype",
    "e",
    "ediff1d",
    "einsum",
    "einsum_path",
    "empty",
    # "empty_like",  # in _creation_functions
    "equal",
    "euler_gamma",
    "exp",
    "exp2",
    "expand_dims",
    "expm1",
    "extract",
    "eye",
    "fabs",
    "fill_diagonal",
    "finfo",
    "fix",
    "flatnonzero",
    "flexible",
    "flip",
    "fliplr",
    "flipud",
    "float16",
    "float32",
    "float64",
    "float8_e4m3b11fnuz",
    "float8_e4m3fn",
    "float8_e4m3fnuz",
    "float8_e5m2",
    "float8_e5m2fnuz",
    "float_",
    "float_power",
    "floating",
    "floor",
    "floor_divide",
    "fmax",
    "fmin",
    "fmod",
    "frexp",
    "from_dlpack",
    "frombuffer",
    # "fromfile",
    "fromfunction",
    "fromiter",
    "frompyfunc",
    "fromstring",
    # "full",  # in _creation_functions
    # "full_like",  # in _creation_functions
    "gcd",
    "generic",
    "geomspace",
    "get_printoptions",
    "gradient",
    "greater",
    "greater_equal",
    "hamming",
    "hanning",
    "heaviside",
    "histogram",
    "histogram2d",
    "histogram_bin_edges",
    "histogramdd",
    "hsplit",
    "hstack",
    "hypot",
    "i0",
    "identity",
    "iinfo",
    "imag",
    "index_exp",
    "indices",
    "inexact",
    "inf",
    "inner",
    "insert",
    "int16",
    "int32",
    "int4",
    "int64",
    "int8",
    "int_",
    "integer",
    "interp",
    "intersect1d",
    "invert",
    "isclose",
    "iscomplex",
    "iscomplexobj",
    "isdtype",
    "isfinite",
    "isin",
    "isinf",
    "isnan",
    "isneginf",
    "isposinf",
    "isreal",
    "isrealobj",
    "isscalar",
    "issubdtype",
    "iterable",
    "ix_",
    "kaiser",
    "kron",
    "lcm",
    "ldexp",
    "left_shift",
    "less",
    "less_equal",
    "lexsort",
    # "linspace",  # in _creation_functions
    "load",
    "log",
    "log10",
    "log1p",
    "log2",
    "logaddexp",
    "logaddexp2",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "logspace",
    "mask_indices",
    "matmul",
    "matrix_transpose",
    "max",
    "maximum",
    "mean",
    "median",
    # "meshgrid",  # in _creation_functions
    "mgrid",
    "min",
    "minimum",
    "mod",
    "modf",
    "moveaxis",
    "multiply",
    "nan",
    "nan_to_num",
    "nanargmax",
    "nanargmin",
    "nancumprod",
    "nancumsum",
    "nanmax",
    "nanmean",
    "nanmedian",
    "nanmin",
    "nanpercentile",
    "nanprod",
    "nanquantile",
    "nanstd",
    "nansum",
    "nanvar",
    "ndarray",
    "ndim",
    "negative",
    "newaxis",
    "nextafter",
    "nonzero",
    "not_equal",
    "number",
    "object_",
    "ogrid",
    "ones",
    # "ones_like",  # in _creation_functions
    "outer",
    "packbits",
    "pad",
    "partition",
    "percentile",
    "permute_dims",
    "pi",
    "piecewise",
    "place",
    "poly",
    "polyadd",
    "polyder",
    "polydiv",
    "polyfit",
    "polyint",
    "polymul",
    "polysub",
    "polyval",
    "positive",
    "pow",
    "power",
    "printoptions",
    "prod",
    "promote_types",
    "ptp",
    "put",
    "quantile",
    "r_",  # not subscriptable
    "rad2deg",
    "radians",
    "ravel",
    "ravel_multi_index",
    "real",
    "reciprocal",
    "remainder",
    "repeat",
    "reshape",
    "resize",
    "result_type",
    "right_shift",
    "rint",
    "roll",
    "rollaxis",
    "roots",
    "rot90",
    "round",
    "round_",
    "s_",  # not subscriptable
    "save",
    "savez",
    "searchsorted",
    "select",
    "set_printoptions",
    "setdiff1d",
    "setxor1d",
    "shape",
    "sign",
    "signbit",
    "signedinteger",
    "sin",
    "sinc",
    "single",
    "sinh",
    "size",
    "sort",
    "sort_complex",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "std",
    "subtract",
    "sum",
    "swapaxes",
    "take",
    "take_along_axis",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "trace",
    "transpose",
    "tri",
    # "tril",  # in _creation_functions
    "tril_indices",
    "tril_indices_from",
    "trim_zeros",
    # "triu",  # in _creation_functions
    "triu_indices",
    "triu_indices_from",
    "true_divide",
    "trunc",
    # "ufunc",  # higher-order function
    "uint",
    "uint16",
    "uint32",
    "uint4",
    "uint64",
    "uint8",
    "union1d",
    "unique",
    "unique_all",
    "unique_counts",
    "unique_inverse",
    "unique_values",
    "unpackbits",
    "unravel_index",
    "unsignedinteger",
    "unwrap",
    "vander",
    "var",
    "vdot",
    "vecdot",
    # "vectorize",
    "vsplit",
    "vstack",
    "where",
    "zeros",
    # "zeros_like",  # in _creation_functions
]

import sys
from collections.abc import Callable
from typing import Any, Literal, TypeVar

import jax.numpy as jnp
from jax._src.numpy.index_tricks import CClass, RClass
from jaxtyping import ArrayLike

from quaxed._types import DType
from quaxed._utils import quaxify

from . import fft, linalg

# =============================================================================
# Explicit constructions
# TODO: not need to do these by specifying a `.pyi` file.
#       but right now, `_higher_order.py` needs this for mypy to pass.

_FuncT = TypeVar("_FuncT")


def _set_docstring(doc: str | None) -> Callable[[_FuncT], _FuncT]:
    def decorator(func: _FuncT) -> _FuncT:
        func.__doc__ = doc
        return func

    return decorator


@quaxify
@_set_docstring(jnp.asarray.__doc__)
def asarray(
    a: ArrayLike,
    dtype: DType | None = None,
    order: Literal["C", "F", "A", "K"] | None = None,
) -> ArrayLike:
    return jnp.asarray(a, dtype=dtype, order=order)


@quaxify
@_set_docstring(jnp.expand_dims.__doc__)
def expand_dims(a: ArrayLike, axis: int | tuple[int, ...]) -> ArrayLike:
    return jnp.expand_dims(a, axis=axis)


@quaxify
@_set_docstring(jnp.squeeze.__doc__)
def squeeze(a: ArrayLike, axis: int | tuple[int, ...] | None = None) -> ArrayLike:
    return jnp.squeeze(a, axis=axis)


class QuaxedCClass(CClass):  # type: ignore[misc]
    """Quaxed version of `jax.numpy.CClass`.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> x = jnp.asarray(jnp.linspace(0, 11, 11), dtype=int)
    >>> jnp.c_[x, x].T
    Array([[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11],
           [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 11]], dtype=int32)

    """

    @quaxify
    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)


c_ = QuaxedCClass()


class QuaxedRClass(RClass):  # type: ignore[misc]
    """Quaxed version of `jax.numpy.RClass`.

    Examples
    --------
    >>> import quaxed.numpy as jnp
    >>> x = jnp.asarray(jnp.linspace(0, 4, 4), dtype=int)
    >>> jnp.r_[x, x]
    Array([0, 1, 2, 4, 0, 1, 2, 4], dtype=int32)

    """

    @quaxify
    def __getitem__(self, key: Any) -> Any:
        return super().__getitem__(key)


r_ = QuaxedRClass()


# =============================================================================
# Automated lazy construction

# Direct transfers
_DIRECT_TRANSFER: frozenset[str] = frozenset(
    (
        "bfloat16",
        "bool",
        "bool_",
        "character",
        "dtype",
        "e",
        "euler_gamma",
        "flexible",
        "floating",
        "float16",
        "float32",
        "float64",
        "generic",
        "index_exp",
        "indices",
        "inexact",
        "inf",
        "int16",
        "int32",
        "int4",
        "int64",
        "int8",
        "int_",
        "integer",
        "mask_indices",
        "mgrid",
        "nan",
        "ndarray",
        "newaxis",
        "number",
        "object_",
        "ogrid",
        "pi",
        "printoptions",
        "promote_types",
        "s_",
        "set_printoptions",
        "signedinteger",
        "single",
        "tri",
        "tril_indices",
        "triu_indices",
        "uint",
        "uint16",
        "uint32",
        "uint4",
        "uint64",
        "uint8",
        "unsignedinteger",
    )
)


_FILTER_SPEC: dict[str, tuple[bool, ...]] = {
    "kaiser": (False, True, True),
    "swapaxes": (True, False, False),
}


def __getattr__(name: str) -> Callable[..., Any]:  # TODO: better type hint
    """Get the object from the `jax.numpy` module."""
    if name not in __all__:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)

    # Get the object
    jnp_obj = getattr(jnp, name)

    # Quaxify?
    out = (
        jnp_obj
        if name in _DIRECT_TRANSFER
        else quaxify(jnp_obj, filter_spec=_FILTER_SPEC.get(name, True))
    )

    # Cache the function in this module
    setattr(sys.modules[__name__], name, out)

    return out
