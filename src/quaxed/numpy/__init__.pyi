from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
import os
import quax
from typing import (
    Any,
    IO,
    Literal,
    NamedTuple,
    Protocol,
    TypeAlias,
    TypeVar,
    Union,
    overload,
)

from jax._src import core as _core
from jax._src import dtypes as _dtypes
from jax._src.lax.lax import PrecisionLike
from jax._src.lax.slicing import GatherScatterMode
from jax._src.lib import Device
from jax._src.numpy.index_tricks import (
    _Mgrid,
    _Ogrid,
    CClass as _CClass,
    RClass as _RClass,
)
from jax._src.numpy.array_api_metadata import ArrayNamespaceInfo
from jax._src.typing import (
    Array,
    ArrayLike as _ArrayLike,
    DType,
    DTypeLike,
    DeprecatedArg,
    DimSize,
    DuckTypedArray,
    Shape,
    StaticScalar,
    SupportsNdim,
    SupportsShape,
    SupportsSize,
)
from jax._src.sharding_impls import NamedSharding, PartitionSpec as P
from jax.numpy import fft as fft, linalg as linalg
from jax.sharding import Sharding as _Sharding
import numpy as _np

ArrayLike: TypeAlias = _ArrayLike | quax.ArrayValue

_T = TypeVar("_T")
_ArrayValueT = TypeVar("_ArrayValueT", bound=quax.ArrayValue)

_Axis = Union[None, int, Sequence[int]]

_Device = Device

ComplexWarning: type

class ufunc:
    def __init__(
        self,
        func: Callable[..., Any],
        /,
        nin: int,
        nout: int,
        *,
        name: str | None = None,
        nargs: int | None = None,
        identity: Any = None,
        call: Callable[..., Any] | None = None,
        reduce: Callable[..., Any] | None = None,
        accumulate: Callable[..., Any] | None = None,
        at: Callable[..., Any] | None = None,
        reduceat: Callable[..., Any] | None = None,
    ): ...
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def identity(self) -> builtins.bool | int | float: ...
    def __call__(self, *args: ArrayLike) -> Any: ...
    def reduce(
        self,
        a: ArrayLike,
        /,
        *,
        axis: int | None = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: builtins.bool = False,
        initial: ArrayLike | None = None,
        where: ArrayLike | None = None,
    ) -> Array: ...
    def accumulate(
        self,
        a: ArrayLike,
        /,
        *,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def at(
        self,
        a: ArrayLike,
        indices: Any,
        b: ArrayLike | None = None,
        /,
        *,
        inplace: builtins.bool = True,
    ) -> Array: ...
    def reduceat(
        self,
        a: ArrayLike,
        indices: Any,
        *,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def outer(self, a: ArrayLike, b: ArrayLike, /) -> Array: ...

class BinaryUfunc(Protocol):
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def identity(self) -> builtins.bool | int | float: ...
    @overload
    def __call__(self, x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
    @overload
    def __call__(self, x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
    @overload
    def __call__(self, x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
    @overload
    def __call__(self, x: ArrayLike, y: ArrayLike, /) -> Array: ...
    def reduce(
        self,
        a: ArrayLike,
        /,
        *,
        axis: int | None = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
        keepdims: builtins.bool = False,
        initial: ArrayLike | None = None,
        where: ArrayLike | None = None,
    ) -> Array: ...
    def accumulate(
        self,
        a: ArrayLike,
        /,
        *,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def at(
        self,
        a: ArrayLike,
        indices: Any,
        b: ArrayLike | None = None,
        /,
        *,
        inplace: builtins.bool = True,
    ) -> Array: ...
    def reduceat(
        self,
        a: ArrayLike,
        indices: Any,
        *,
        axis: int = 0,
        dtype: DTypeLike | None = None,
        out: None = None,
    ) -> Array: ...
    def outer(
        self,
        a: ArrayLike,
        b: ArrayLike,
        /,
    ) -> Array: ...

__array_api_version__: str

def __array_namespace_info__() -> ArrayNamespaceInfo: ...

_deprecations: dict[str, tuple[str, Any]]

@overload
def abs(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def abs(x: ArrayLike, /) -> Array: ...
@overload
def absolute(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def absolute(x: ArrayLike, /) -> Array: ...
@overload
def acos(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def acos(x: ArrayLike, /) -> Array: ...
@overload
def acosh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def acosh(x: ArrayLike, /) -> Array: ...

add: BinaryUfunc

def amax(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def amin(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def all(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def allclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: builtins.bool = ...,
) -> Array: ...
def angle(z: ArrayLike, deg: builtins.bool = ...) -> Array: ...
def any(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def append(arr: ArrayLike, values: ArrayLike, axis: int | None = ...) -> Array: ...
def apply_along_axis(
    func1d: Callable, axis: int, arr: ArrayLike, *args, **kwargs
) -> Array: ...
def apply_over_axes(func: Callable, a: ArrayLike, axes: Sequence[int]) -> Array: ...
def arange(
    start: ArrayLike | DimSize,
    stop: ArrayLike | DimSize | None = ...,
    step: ArrayLike | None = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def arccos(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arccos(x: ArrayLike, /) -> Array: ...
@overload
def arccosh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arccosh(x: ArrayLike, /) -> Array: ...
@overload
def arcsin(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arcsin(x: ArrayLike, /) -> Array: ...
@overload
def arcsinh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arcsinh(x: ArrayLike, /) -> Array: ...
@overload
def arctan(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arctan(x: ArrayLike, /) -> Array: ...
@overload
def arctan2(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def arctan2(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arctan2(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arctan2(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def arctanh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def arctanh(x: ArrayLike, /) -> Array: ...
def argmax(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: builtins.bool | None = ...,
) -> Array: ...
def argmin(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: builtins.bool | None = ...,
) -> Array: ...
def argpartition(a: ArrayLike, kth: int, axis: int = ...) -> Array: ...
def argsort(
    a: ArrayLike,
    axis: int | None = ...,
    *,
    stable: builtins.bool = ...,
    descending: builtins.bool = ...,
    kind: str | None = ...,
    order: None = ...,
) -> Array: ...
def argwhere(
    a: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
) -> Array: ...
def around(a: ArrayLike, decimals: int = ..., out: None = ...) -> Array: ...
def array(
    object: Any,
    dtype: DTypeLike | None = ...,
    copy: builtins.bool = True,
    order: str | None = ...,
    ndmin: int = ...,
    *,
    device: _Device | _Sharding | None = None,
) -> Array: ...
def array_equal(
    a1: ArrayLike, a2: ArrayLike, equal_nan: builtins.bool = ...
) -> Array: ...
def array_equiv(a1: ArrayLike, a2: ArrayLike) -> Array: ...

array_repr = _np.array_repr

def array_split(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
    axis: int = ...,
) -> list[Array]: ...

array_str = _np.array_str

def asarray(
    a: Any,
    dtype: DTypeLike | None = ...,
    order: str | None = ...,
    *,
    copy: builtins.bool | None = ...,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def asin(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def asin(x: ArrayLike, /) -> Array: ...
@overload
def asinh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def asinh(x: ArrayLike, /) -> Array: ...
def astype(
    a: ArrayLike,
    dtype: DTypeLike | None,
    /,
    *,
    copy: builtins.bool = ...,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def atan(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atan(x: ArrayLike, /) -> Array: ...
@overload
def atan2(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def atan2(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atan2(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atan2(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def atanh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atanh(x: ArrayLike, /) -> Array: ...
@overload
def atleast_1d() -> list[Array]: ...
@overload
@overload
def atleast_1d(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atleast_1d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_1d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_2d() -> list[Array]: ...
@overload
@overload
def atleast_2d(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atleast_2d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_2d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def atleast_3d() -> list[Array]: ...
@overload
@overload
def atleast_3d(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def atleast_3d(x: ArrayLike, /) -> Array: ...
@overload
def atleast_3d(x: ArrayLike, y: ArrayLike, /, *arys: ArrayLike) -> list[Array]: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    returned: Literal[False] = False,
    keepdims: builtins.bool = False,
) -> Array: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    *,
    returned: Literal[True],
    keepdims: builtins.bool = False,
) -> tuple[Array, Array]: ...
@overload
def average(
    a: ArrayLike,
    axis: _Axis = ...,
    weights: ArrayLike | None = ...,
    returned: builtins.bool = False,
    keepdims: builtins.bool = False,
) -> Array | tuple[Array, Array]: ...
def bartlett(M: int) -> Array: ...

bfloat16: Any

def bincount(
    x: ArrayLike,
    weights: ArrayLike | None = ...,
    minlength: int = ...,
    *,
    length: int | None = ...,
) -> Array: ...

bitwise_and: BinaryUfunc

@overload
def bitwise_count(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_count(x: ArrayLike, /) -> Array: ...
@overload
def bitwise_invert(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_invert(x: ArrayLike, /) -> Array: ...
@overload
def bitwise_left_shift(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def bitwise_left_shift(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_left_shift(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_left_shift(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def bitwise_not(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_not(x: ArrayLike, /) -> Array: ...

bitwise_or: BinaryUfunc

@overload
def bitwise_right_shift(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def bitwise_right_shift(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_right_shift(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def bitwise_right_shift(x: ArrayLike, y: ArrayLike, /) -> Array: ...

bitwise_xor: BinaryUfunc

def blackman(M: int) -> Array: ...
def block(
    arrays: ArrayLike | Sequence[ArrayLike] | Sequence[Sequence[ArrayLike]],
) -> Array: ...

bool: Any
bool_: Any

def broadcast_arrays(*args: ArrayLike) -> list[Array]: ...
@overload
def broadcast_shapes(*shapes: Sequence[int]) -> tuple[int, ...]: ...
@overload
def broadcast_shapes(
    *shapes: Sequence[int | _core.Tracer],
) -> tuple[int | _core.Tracer, ...]: ...
def broadcast_to(
    array: ArrayLike,
    shape: DimSize | Shape,
    *,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...

c_: _CClass
can_cast = _np.can_cast

@overload
def cbrt(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def cbrt(x: ArrayLike, /) -> Array: ...

cdouble: Any

@overload
def ceil(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def ceil(x: ArrayLike, /) -> Array: ...

character = _np.character

def choose(
    a: ArrayLike,
    choices: Array | _np.ndarray | Sequence[ArrayLike],
    out: None = ...,
    mode: str = ...,
) -> Array: ...
def clip(
    x: ArrayLike | None = ...,
    /,
    min: ArrayLike | None = ...,
    max: ArrayLike | None = ...,
    *,
    a: ArrayLike | DeprecatedArg | None = ...,
    a_min: ArrayLike | DeprecatedArg | None = ...,
    a_max: ArrayLike | DeprecatedArg | None = ...,
) -> Array: ...
def column_stack(tup: _np.ndarray | Array | Sequence[ArrayLike]) -> Array: ...

complex128: Any
complex64: Any
complex_: Any
complexfloating = _np.complexfloating

def compress(
    condition: ArrayLike,
    a: ArrayLike,
    axis: int | None = ...,
    *,
    size: int | None = ...,
    fill_value: ArrayLike = ...,
    out: None = ...,
) -> Array: ...
def concat(arrays: Sequence[ArrayLike], /, *, axis: int | None = 0) -> Array: ...
def concatenate(
    arrays: _np.ndarray | Array | Sequence[ArrayLike],
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
@overload
def conjugate(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def conjugate(x: ArrayLike, /) -> Array: ...
@overload
def conj(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def conj(x: ArrayLike, /) -> Array: ...
def convolve(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = ...,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def copy(a: ArrayLike, order: str | None = ...) -> Array: ...
@overload
def copysign(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def copysign(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def copysign(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def copysign(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def corrcoef(
    x: ArrayLike, y: ArrayLike | None = ..., rowvar: builtins.bool = ...
) -> Array: ...
def correlate(
    a: ArrayLike,
    v: ArrayLike,
    mode: str = ...,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
@overload
def cos(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def cos(x: ArrayLike, /) -> Array: ...
@overload
def cosh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def cosh(x: ArrayLike, /) -> Array: ...
def count_nonzero(
    a: ArrayLike, axis: _Axis = ..., keepdims: builtins.bool = ...
) -> Array: ...
def cov(
    m: ArrayLike,
    y: ArrayLike | None = ...,
    rowvar: builtins.bool = ...,
    bias: builtins.bool = ...,
    ddof: int | None = ...,
    fweights: ArrayLike | None = ...,
    aweights: ArrayLike | None = ...,
) -> Array: ...
def cross(
    a: ArrayLike,
    b: ArrayLike,
    axisa: int = -1,
    axisb: int = -1,
    axisc: int = -1,
    axis: int | None = ...,
) -> Array: ...

csingle: Any

def cumprod(
    a: ArrayLike, axis: int | None = ..., dtype: DTypeLike | None = ..., out: None = ...
) -> Array: ...
def cumsum(
    a: ArrayLike, axis: int | None = ..., dtype: DTypeLike | None = ..., out: None = ...
) -> Array: ...
@overload
def cumulative_prod(
    x: _ArrayValueT,
    /,
    *,
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
    include_initial: builtins.bool = ...,
) -> _ArrayValueT: ...
@overload
def cumulative_prod(
    x: ArrayLike,
    /,
    *,
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
    include_initial: builtins.bool = ...,
) -> Array: ...
@overload
def cumulative_sum(
    x: _ArrayValueT,
    /,
    *,
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
    include_initial: builtins.bool = ...,
) -> _ArrayValueT: ...
@overload
def cumulative_sum(
    x: ArrayLike,
    /,
    *,
    axis: int | None = ...,
    dtype: DTypeLike | None = ...,
    include_initial: builtins.bool = ...,
) -> Array: ...
@overload
def deg2rad(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def deg2rad(x: ArrayLike, /) -> Array: ...
@overload
def degrees(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def degrees(x: ArrayLike, /) -> Array: ...
def delete(
    arr: ArrayLike,
    obj: ArrayLike | slice,
    axis: int | None = ...,
    *,
    assume_unique_indices: builtins.bool = ...,
) -> Array: ...
def diag(v: ArrayLike, k: int = 0) -> Array: ...
def diag_indices(n: int, ndim: int = ...) -> tuple[Array, ...]: ...
def diag_indices_from(arr: ArrayLike) -> tuple[Array, ...]: ...
def diagflat(v: ArrayLike, k: int = 0) -> Array: ...
def diagonal(
    a: ArrayLike, offset: ArrayLike = ..., axis1: int = ..., axis2: int = ...
): ...
def diff(
    a: ArrayLike,
    n: int = ...,
    axis: int = ...,
    prepend: ArrayLike | None = ...,
    append: ArrayLike | None = ...,
) -> Array: ...
def digitize(
    x: ArrayLike,
    bins: ArrayLike,
    right: builtins.bool = ...,
    *,
    method: str | None = ...,
) -> Array: ...
@overload
def divide(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def divide(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def divide(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def divide(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def divmod(x: ArrayLike, y: ArrayLike, /) -> tuple[Array, Array]: ...
def dot(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    out_sharding: NamedSharding | P | None = ...,
) -> Array: ...

double: Any

def dsplit(ary: ArrayLike, indices_or_sections: int | ArrayLike) -> list[Array]: ...
def dstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike], dtype: DTypeLike | None = ...
) -> Array: ...

dtype = _np.dtype
e: float

def ediff1d(
    ary: ArrayLike, to_end: ArrayLike | None = ..., to_begin: ArrayLike | None = ...
) -> Array: ...
@overload
def einsum(
    subscript: str,
    /,
    *operands: ArrayLike,
    out: None = ...,
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _dot_general: Callable[..., Array] = ...,
    out_sharding: NamedSharding | P | None = ...,
) -> Array: ...
@overload
def einsum(
    arr: ArrayLike,
    axes: Sequence[Any],
    /,
    *operands: ArrayLike | Sequence[Any],
    out: None = ...,
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _dot_general: Callable[..., Array] = ...,
    out_sharding: NamedSharding | P | None = ...,
) -> Array: ...
@overload
def einsum(
    subscripts,
    /,
    *operands,
    out: None = ...,
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
    _dot_general: Callable[..., Array] = ...,
    out_sharding: NamedSharding | P | None = ...,
) -> Array: ...
@overload
def einsum_path(
    subscripts: str,
    /,
    *operands: ArrayLike,
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
) -> tuple[list[tuple[int, ...]], Any]: ...
@overload
def einsum_path(
    arr: ArrayLike,
    axes: Sequence[Any],
    /,
    *operands: ArrayLike | Sequence[Any],
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
) -> tuple[list[tuple[int, ...]], Any]: ...
@overload
def einsum_path(
    subscripts,
    /,
    *operands: ArrayLike,
    optimize: str | builtins.bool | list[tuple[int, ...]] = ...,
) -> tuple[list[tuple[int, ...]], Any]: ...
def empty(
    shape: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def empty_like(
    prototype: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def equal(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def equal(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def equal(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...

euler_gamma: float

@overload
def exp(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def exp(x: ArrayLike, /) -> Array: ...
@overload
def exp2(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def exp2(x: ArrayLike, /) -> Array: ...
def expand_dims(a: ArrayLike, axis: int | Sequence[int]) -> Array: ...
@overload
def expm1(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def expm1(x: ArrayLike, /) -> Array: ...
def extract(
    condition: ArrayLike,
    arr: ArrayLike,
    *,
    size: int | None = None,
    fill_value: ArrayLike = 0,
) -> Array: ...
def eye(
    N: DimSize,
    M: DimSize | None = ...,
    k: int | ArrayLike = ...,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def fabs(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fabs(x: ArrayLike, /) -> Array: ...

finfo = _dtypes.finfo

def fix(x: ArrayLike, out: None = ...) -> Array: ...
def flatnonzero(
    a: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike] = ...,
) -> Array: ...

flexible = _np.flexible

def flip(m: ArrayLike, axis: int | Sequence[int] | None = ...) -> Array: ...
def fliplr(m: ArrayLike) -> Array: ...
def flipud(m: ArrayLike) -> Array: ...

float16: Any
float32: Any
float64: Any
float8_e4m3b11fnuz: Any
float8_e4m3fn: Any
float8_e4m3fnuz: Any
float8_e5m2: Any
float8_e5m2fnuz: Any
float_: Any

@overload
def float_power(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def float_power(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def float_power(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def float_power(x: ArrayLike, y: ArrayLike, /) -> Array: ...

floating = _np.floating

@overload
def floor(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def floor(x: ArrayLike, /) -> Array: ...
@overload
def floor_divide(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def floor_divide(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def floor_divide(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def floor_divide(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def fmax(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def fmax(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmax(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmax(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def fmin(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def fmin(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmin(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmin(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def fmod(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def fmod(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmod(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def fmod(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def frexp(x: ArrayLike, /) -> tuple[Array, Array]: ...
def from_dlpack(
    x: Any,
    /,
    *,
    device: _Device | _Sharding | None = None,
    copy: builtins.bool | None = None,
) -> Array: ...
def frombuffer(
    buffer: bytes | Any, dtype: DTypeLike = ..., count: int = ..., offset: int = ...
) -> Array: ...
def fromfile(*args, **kwargs): ...
def fromfunction(
    function: Callable[..., Array], shape: Any, *, dtype: DTypeLike = ..., **kwargs
) -> Array: ...
def fromiter(*args, **kwargs): ...
def frompyfunc(
    func: Callable[..., Any], /, nin: int, nout: int, *, identity: Any = None
) -> ufunc: ...
def fromstring(
    string: str, dtype: DTypeLike = ..., count: int = ..., *, sep: str
) -> Array: ...
def full(
    shape: Any,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def full_like(
    a: ArrayLike | DuckTypedArray,
    fill_value: ArrayLike,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def gcd(x1: ArrayLike, x2: ArrayLike) -> Array: ...

generic = _np.generic

def geomspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = ...,
    endpoint: builtins.bool = ...,
    dtype: DTypeLike | None = ...,
    axis: int = ...,
) -> Array: ...

get_printoptions = _np.get_printoptions

def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: int | Sequence[int] | None = ...,
    edge_order: int | None = ...,
) -> Array | list[Array]: ...
@overload
def greater(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def greater(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def greater(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def greater(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def greater_equal(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def greater_equal(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def greater_equal(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def greater_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def hamming(M: int) -> Array: ...
def hanning(M: int) -> Array: ...
@overload
def heaviside(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def heaviside(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def heaviside(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def heaviside(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def histogram(
    a: ArrayLike,
    bins: ArrayLike = ...,
    range: Sequence[ArrayLike] | None = ...,
    weights: ArrayLike | None = ...,
    density: builtins.bool | None = ...,
) -> tuple[Array, Array]: ...
def histogram2d(
    x: ArrayLike,
    y: ArrayLike,
    bins: ArrayLike | Sequence[ArrayLike] = ...,
    range: Sequence[None | Array | Sequence[ArrayLike]] | None = ...,
    weights: ArrayLike | None = ...,
    density: builtins.bool | None = ...,
) -> tuple[Array, Array, Array]: ...
def histogram_bin_edges(
    a: ArrayLike,
    bins: ArrayLike = ...,
    range: None | Array | Sequence[ArrayLike] = ...,
    weights: ArrayLike | None = ...,
) -> Array: ...
def histogramdd(
    sample: ArrayLike,
    bins: ArrayLike | Sequence[ArrayLike] = ...,
    range: Sequence[None | Array | Sequence[ArrayLike]] | None = ...,
    weights: ArrayLike | None = ...,
    density: builtins.bool | None = ...,
) -> tuple[Array, list[Array]]: ...
def hsplit(ary: ArrayLike, indices_or_sections: int | ArrayLike) -> list[Array]: ...
def hstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike], dtype: DTypeLike | None = ...
) -> Array: ...
@overload
def hypot(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def hypot(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def hypot(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def hypot(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def i0(x: ArrayLike) -> Array: ...
def identity(n: DimSize, dtype: DTypeLike | None = ...) -> Array: ...

iinfo = _dtypes.iinfo

@overload
def imag(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def imag(x: ArrayLike, /) -> Array: ...

index_exp = _np.index_exp

@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = None,
    sparse: Literal[False] = False,
) -> Array: ...
@overload
def indices(
    dimensions: Sequence[int], dtype: DTypeLike | None = None, *, sparse: Literal[True]
) -> tuple[Array, ...]: ...
@overload
def indices(
    dimensions: Sequence[int],
    dtype: DTypeLike | None = None,
    sparse: builtins.bool = False,
) -> Array | tuple[Array, ...]: ...

inexact = _np.inexact
inf: float

def inner(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def insert(
    arr: ArrayLike, obj: ArrayLike | slice, values: ArrayLike, axis: int | None = ...
) -> Array: ...

int16: Any
int32: Any
int4: Any
int64: Any
int8: Any
int_: Any
integer = _np.integer

def interp(
    x: ArrayLike,
    xp: ArrayLike,
    fp: ArrayLike,
    left: ArrayLike | str | None = ...,
    right: ArrayLike | str | None = ...,
    period: ArrayLike | None = ...,
) -> Array: ...
def intersect1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: builtins.bool = ...,
    return_indices: builtins.bool = ...,
    *,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
) -> Array | tuple[Array, Array, Array]: ...
@overload
def invert(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def invert(x: ArrayLike, /) -> Array: ...
def isclose(
    a: ArrayLike,
    b: ArrayLike,
    rtol: ArrayLike = ...,
    atol: ArrayLike = ...,
    equal_nan: builtins.bool = ...,
) -> Array: ...
def iscomplex(x: ArrayLike) -> Array: ...
def iscomplexobj(x: Any) -> builtins.bool: ...
def isdtype(
    dtype: DTypeLike, kind: DType | str | tuple[DType | str, ...]
) -> builtins.bool: ...
@overload
def isfinite(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def isfinite(x: ArrayLike, /) -> Array: ...
def isin(
    element: ArrayLike,
    test_elements: ArrayLike,
    assume_unique: builtins.bool = ...,
    invert: builtins.bool = ...,
    *,
    method: str = ...,
) -> Array: ...
@overload
def isinf(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def isinf(x: ArrayLike, /) -> Array: ...
@overload
def isnan(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def isnan(x: ArrayLike, /) -> Array: ...
@overload
def isneginf(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def isneginf(x: ArrayLike, /) -> Array: ...
@overload
def isposinf(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def isposinf(x: ArrayLike, /) -> Array: ...
def isreal(x: ArrayLike) -> Array: ...
def isrealobj(x: Any) -> builtins.bool: ...
def isscalar(element: Any) -> builtins.bool: ...
def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> builtins.bool: ...

iterable = _np.iterable

def ix_(*args: ArrayLike) -> tuple[Array, ...]: ...
def kaiser(M: int, beta: ArrayLike) -> Array: ...
def kron(a: ArrayLike, b: ArrayLike) -> Array: ...
def lcm(x1: ArrayLike, x2: ArrayLike) -> Array: ...
@overload
def ldexp(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def ldexp(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def ldexp(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def ldexp(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def left_shift(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def left_shift(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def left_shift(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def left_shift(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def less(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def less(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def less(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def less(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def less_equal(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def less_equal(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def less_equal(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def less_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def lexsort(
    keys: Array | _np.ndarray | Sequence[ArrayLike], axis: int = ...
) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: builtins.bool = True,
    retstep: Literal[False] = False,
    dtype: DTypeLike | None = ...,
    axis: int = 0,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int,
    endpoint: builtins.bool,
    retstep: Literal[True],
    dtype: DTypeLike | None = ...,
    axis: int = 0,
    *,
    device: _Device | _Sharding | None = ...,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: builtins.bool = True,
    *,
    retstep: Literal[True],
    dtype: DTypeLike | None = ...,
    axis: int = 0,
    device: _Device | _Sharding | None = ...,
) -> tuple[Array, Array]: ...
@overload
def linspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = 50,
    endpoint: builtins.bool = True,
    retstep: builtins.bool = False,
    dtype: DTypeLike | None = ...,
    axis: int = 0,
    *,
    device: _Device | _Sharding | None = ...,
) -> Union[Array, tuple[Array, Array]]: ...
def load(
    file: IO[bytes] | str | os.PathLike[Any], *args: Any, **kwargs: Any
) -> Array: ...
@overload
def log(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def log(x: ArrayLike, /) -> Array: ...
@overload
def log10(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def log10(x: ArrayLike, /) -> Array: ...
@overload
def log1p(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def log1p(x: ArrayLike, /) -> Array: ...
@overload
def log2(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def log2(x: ArrayLike, /) -> Array: ...

logaddexp: BinaryUfunc
logaddexp2: BinaryUfunc
logical_and: BinaryUfunc

@overload
def logical_not(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def logical_not(x: ArrayLike, /) -> Array: ...

logical_or: BinaryUfunc
logical_xor: BinaryUfunc

def logspace(
    start: ArrayLike,
    stop: ArrayLike,
    num: int = ...,
    endpoint: builtins.bool = ...,
    base: ArrayLike = ...,
    dtype: DTypeLike | None = ...,
    axis: int = ...,
) -> Array: ...
def mask_indices(
    n: int, mask_func: Callable, k: int = ..., *, size: int | None = ...
) -> tuple[Array, ...]: ...
def matmul(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
@overload
def matrix_transpose(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def matrix_transpose(x: ArrayLike, /) -> Array: ...
def matvec(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
def max(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
@overload
def maximum(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def maximum(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def maximum(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def maximum(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def mean(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    *,
    where: ArrayLike | None = ...,
) -> Array: ...
def median(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    keepdims: builtins.bool = ...,
) -> Array: ...
def meshgrid(
    *xi: ArrayLike,
    copy: builtins.bool = ...,
    sparse: builtins.bool = ...,
    indexing: str = ...,
) -> list[Array]: ...

mgrid: _Mgrid

def min(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
@overload
def minimum(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def minimum(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def minimum(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def minimum(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def mod(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def mod(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def mod(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def mod(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def modf(x: ArrayLike, /, out=None) -> tuple[Array, Array]: ...
def moveaxis(
    a: ArrayLike, source: int | Sequence[int], destination: int | Sequence[int]
) -> Array: ...

multiply: BinaryUfunc
nan: float

def nan_to_num(
    x: ArrayLike,
    copy: builtins.bool = ...,
    nan: ArrayLike = ...,
    posinf: ArrayLike | None = ...,
    neginf: ArrayLike | None = ...,
) -> Array: ...
def nanargmax(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: builtins.bool | None = ...,
) -> Array: ...
def nanargmin(
    a: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    keepdims: builtins.bool | None = ...,
) -> Array: ...
def nancumprod(
    a: ArrayLike, axis: int | None = ..., dtype: DTypeLike | None = ..., out: None = ...
) -> Array: ...
def nancumsum(
    a: ArrayLike, axis: int | None = ..., dtype: DTypeLike | None = ..., out: None = ...
) -> Array: ...
def nanmax(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanmean(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanmedian(
    a: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    keepdims: builtins.bool = ...,
) -> Array: ...
def nanmin(
    a: ArrayLike,
    axis: _Axis = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanpercentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    method: str = ...,
    keepdims: builtins.bool = ...,
    *,
    interpolation: DeprecatedArg | str = ...,
) -> Array: ...
def nanprod(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanquantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    method: str = ...,
    keepdims: builtins.bool = ...,
    *,
    interpolation: DeprecatedArg | str = ...,
) -> Array: ...
def nanstd(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: builtins.bool = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nansum(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
) -> Array: ...
def nanvar(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    ddof: int = 0,
    keepdims: builtins.bool = False,
    where: ArrayLike | None = ...,
) -> Array: ...

ndarray = Array

def ndim(a: ArrayLike | SupportsNdim) -> int: ...
@overload
def negative(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def negative(x: ArrayLike, /) -> Array: ...

newaxis = None

@overload
def nextafter(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def nextafter(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def nextafter(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def nextafter(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def nonzero(
    a: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> tuple[Array, ...]: ...
@overload
def not_equal(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def not_equal(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def not_equal(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def not_equal(x: ArrayLike, y: ArrayLike, /) -> Array: ...

number = _np.number
object_ = _np.object_
ogrid: _Ogrid

def ones(
    shape: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def ones_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def outer(a: ArrayLike, b: Array, out: None = ...) -> Array: ...
def packbits(a: ArrayLike, axis: int | None = ..., bitorder: str = ...) -> Array: ...

PadValueLike = Union[_T, Sequence[_T], Sequence[Sequence[_T]]]

def pad(
    array: ArrayLike,
    pad_width: PadValueLike[int | Array | _np.ndarray],
    mode: str | Callable[..., Any] = ...,
    **kwargs,
) -> Array: ...
def partition(a: ArrayLike, kth: int, axis: int = ...) -> Array: ...
def percentile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    method: str = ...,
    keepdims: builtins.bool = ...,
    *,
    interpolation: DeprecatedArg | str = ...,
) -> Array: ...
@overload
def permute_dims(x: _ArrayValueT, /, axes: tuple[int, ...]) -> _ArrayValueT: ...
@overload
def permute_dims(x: ArrayLike, /, axes: tuple[int, ...]) -> Array: ...

pi: float

def piecewise(
    x: ArrayLike,
    condlist: Array | Sequence[ArrayLike],
    funclist: Sequence[ArrayLike | Callable[..., Array]],
    *args,
    **kw,
) -> Array: ...
def place(
    arr: ArrayLike, mask: ArrayLike, vals: ArrayLike, *, inplace: builtins.bool = ...
) -> Array: ...
def poly(seq_of_zeros: ArrayLike) -> Array: ...
def polyadd(a1: ArrayLike, a2: ArrayLike) -> Array: ...
def polyder(p: ArrayLike, m: int = ...) -> Array: ...
def polydiv(
    u: ArrayLike, v: ArrayLike, *, trim_leading_zeros: builtins.bool = ...
) -> tuple[Array, Array]: ...
def polyfit(
    x: ArrayLike,
    y: ArrayLike,
    deg: int,
    rcond: float | None = ...,
    full: builtins.bool = ...,
    w: ArrayLike | None = ...,
    cov: builtins.bool = ...,
) -> Array | tuple[Array, ...]: ...
def polyint(p: ArrayLike, m: int = ..., k: int | ArrayLike | None = ...) -> Array: ...
def polymul(
    a1: ArrayLike, a2: ArrayLike, *, trim_leading_zeros: builtins.bool = ...
) -> Array: ...
def polysub(a1: ArrayLike, a2: ArrayLike) -> Array: ...
def polyval(p: ArrayLike, x: ArrayLike, *, unroll: int = ...) -> Array: ...
@overload
def positive(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def positive(x: ArrayLike, /) -> Array: ...
@overload
def pow(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def pow(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def pow(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def pow(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def power(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def power(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def power(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def power(x: ArrayLike, y: ArrayLike, /) -> Array: ...

printoptions = _np.printoptions

def prod(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
    promote_integers: builtins.bool = ...,
) -> Array: ...

promote_types = _np.promote_types

def ptp(
    a: ArrayLike, axis: _Axis = ..., out: None = ..., keepdims: builtins.bool = ...
) -> Array: ...
def put(
    a: ArrayLike,
    ind: ArrayLike,
    v: ArrayLike,
    mode: str | None = ...,
    *,
    inplace: builtins.bool = ...,
) -> Array: ...
def put_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    values: ArrayLike,
    axis: int | None,
    inplace: bool = True,
    *,
    mode: str | None = None,
) -> Array: ...
def quantile(
    a: ArrayLike,
    q: ArrayLike,
    axis: int | tuple[int, ...] | None = ...,
    out: None = ...,
    overwrite_input: builtins.bool = ...,
    method: str = ...,
    keepdims: builtins.bool = ...,
    *,
    interpolation: DeprecatedArg | str = ...,
) -> Array: ...

r_: _RClass

@overload
def rad2deg(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def rad2deg(x: ArrayLike, /) -> Array: ...
@overload
def radians(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def radians(x: ArrayLike, /) -> Array: ...
def ravel(
    a: ArrayLike, order: str = ..., *, out_sharding: NamedSharding | P | None = ...
) -> Array: ...
def ravel_multi_index(
    multi_index: Sequence[ArrayLike],
    dims: Sequence[int],
    mode: str = ...,
    order: str = ...,
) -> Array: ...
@overload
def real(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def real(x: ArrayLike, /) -> Array: ...
@overload
def reciprocal(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def reciprocal(x: ArrayLike, /) -> Array: ...
@overload
def remainder(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def remainder(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def remainder(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def remainder(x: ArrayLike, y: ArrayLike, /) -> Array: ...
def repeat(
    a: ArrayLike,
    repeats: ArrayLike,
    axis: int | None = ...,
    *,
    total_repeat_length: int | None = ...,
    out_sharding: NamedSharding | P | None = None,
) -> Array: ...
def reshape(
    a: ArrayLike,
    shape: DimSize | Shape,
    order: str = ...,
    *,
    copy: bool | None = ...,
    out_sharding: NamedSharding | P | None = ...,
) -> Array: ...
def resize(a: ArrayLike, new_shape: Shape) -> Array: ...
def result_type(*args: Any) -> DType: ...
@overload
def right_shift(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def right_shift(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def right_shift(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def right_shift(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def rint(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def rint(x: ArrayLike, /) -> Array: ...
def roll(
    a: ArrayLike,
    shift: ArrayLike | Sequence[int],
    axis: int | Sequence[int] | None = ...,
) -> Array: ...
def rollaxis(a: ArrayLike, axis: int, start: int = 0) -> Array: ...
def roots(p: ArrayLike, *, strip_zeros: builtins.bool = ...) -> Array: ...
def rot90(m: ArrayLike, k: int = ..., axes: tuple[int, int] = ...) -> Array: ...
def round(a: ArrayLike, decimals: int = ..., out: None = ...) -> Array: ...

s_ = _np.s_
save = _np.save
savez = _np.savez

def searchsorted(
    a: ArrayLike,
    v: ArrayLike,
    side: str = ...,
    sorter: ArrayLike | None = ...,
    *,
    method: str = ...,
) -> Array: ...
def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = ...,
) -> Array: ...

set_printoptions = _np.set_printoptions

def setdiff1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: builtins.bool = ...,
    *,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
) -> Array: ...
def setxor1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    assume_unique: builtins.bool = ...,
    *,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
) -> Array: ...
def shape(a: ArrayLike | SupportsShape) -> tuple[int, ...]: ...
@overload
def sign(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def sign(x: ArrayLike, /) -> Array: ...
@overload
def signbit(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def signbit(x: ArrayLike, /) -> Array: ...

signedinteger = _np.signedinteger

@overload
def sin(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def sin(x: ArrayLike, /) -> Array: ...
@overload
def sinc(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def sinc(x: ArrayLike, /) -> Array: ...

single: Any

@overload
def sinh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def sinh(x: ArrayLike, /) -> Array: ...
def size(a: ArrayLike | SupportsSize, axis: int | None = None) -> int: ...
def sort(
    a: ArrayLike,
    axis: int | None = ...,
    *,
    stable: builtins.bool = ...,
    descending: builtins.bool = ...,
    kind: str | None = ...,
    order: None = ...,
) -> Array: ...
def sort_complex(a: ArrayLike) -> Array: ...
@overload
def spacing(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def spacing(x: ArrayLike, /) -> Array: ...
def split(
    ary: ArrayLike,
    indices_or_sections: int | Sequence[int] | ArrayLike,
    axis: int = ...,
) -> list[Array]: ...
@overload
def sqrt(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def sqrt(x: ArrayLike, /) -> Array: ...
@overload
def square(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def square(x: ArrayLike, /) -> Array: ...
def squeeze(a: ArrayLike, axis: int | Sequence[int] | None = ...) -> Array: ...
def stack(
    arrays: _np.ndarray | Array | Sequence[ArrayLike],
    axis: int = ...,
    out: None = ...,
    dtype: DTypeLike | None = ...,
) -> Array: ...
def std(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: builtins.bool = ...,
    *,
    where: ArrayLike | None = ...,
    correction: int | float | None = ...,
) -> Array: ...

subtract: BinaryUfunc

def sum(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    keepdims: builtins.bool = ...,
    initial: ArrayLike | None = ...,
    where: ArrayLike | None = ...,
    promote_integers: builtins.bool = ...,
) -> Array: ...
def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> Array: ...
def take(
    a: ArrayLike,
    indices: ArrayLike,
    axis: int | None = ...,
    out: None = ...,
    mode: str | None = ...,
    unique_indices: builtins.bool = ...,
    indices_are_sorted: builtins.bool = ...,
    fill_value: StaticScalar | None = ...,
) -> Array: ...
def take_along_axis(
    arr: ArrayLike,
    indices: ArrayLike,
    axis: int | None,
    mode: str | GatherScatterMode | None = ...,
    fill_value: StaticScalar | None = None,
) -> Array: ...
@overload
def tan(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def tan(x: ArrayLike, /) -> Array: ...
@overload
def tanh(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def tanh(x: ArrayLike, /) -> Array: ...
def tensordot(
    a: ArrayLike,
    b: ArrayLike,
    axes: int | Sequence[int] | Sequence[Sequence[int]] = ...,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def tile(A: ArrayLike, reps: DimSize | Sequence[DimSize]) -> Array: ...
def trace(
    a: ArrayLike,
    offset: int | ArrayLike = ...,
    axis1: int = ...,
    axis2: int = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
) -> Array: ...
def transpose(a: ArrayLike, axes: Sequence[int] | None = ...) -> Array: ...
def trapezoid(
    y: ArrayLike, x: ArrayLike | None = None, dx: ArrayLike = ..., axis: int = ...
) -> Array: ...
def tri(
    N: int, M: int | None = ..., k: int = ..., dtype: DTypeLike | None = ...
) -> Array: ...
def tril(m: ArrayLike, k: int = ...) -> Array: ...
def tril_indices(n: int, k: int = ..., m: int | None = ...) -> tuple[Array, Array]: ...
def tril_indices_from(
    arr: ArrayLike | SupportsShape, k: int = ...
) -> tuple[Array, Array]: ...
def fill_diagonal(
    a: ArrayLike,
    val: ArrayLike,
    wrap: builtins.bool = ...,
    *,
    inplace: builtins.bool = ...,
) -> Array: ...
def trim_zeros(filt: ArrayLike, trim: str = ...) -> Array: ...
def triu(m: ArrayLike, k: int = ...) -> Array: ...
def triu_indices(n: int, k: int = ..., m: int | None = ...) -> tuple[Array, Array]: ...
def triu_indices_from(
    arr: ArrayLike | SupportsShape, k: int = ...
) -> tuple[Array, Array]: ...
@overload
def true_divide(x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
@overload
def true_divide(x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def true_divide(x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def true_divide(x: ArrayLike, y: ArrayLike, /) -> Array: ...
@overload
def trunc(x: _ArrayValueT, /) -> _ArrayValueT: ...
@overload
def trunc(x: ArrayLike, /) -> Array: ...

uint: Any
uint16: Any
uint32: Any
uint4: Any
uint64: Any
uint8: Any

def union1d(
    ar1: ArrayLike,
    ar2: ArrayLike,
    *,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
) -> Array: ...

class _UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array

class _UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array

class _UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array

def unique(
    ar: ArrayLike,
    return_index: builtins.bool = ...,
    return_inverse: builtins.bool = ...,
    return_counts: builtins.bool = ...,
    axis: int | None = ...,
    *,
    equal_nan: builtins.bool = ...,
    size: int | None = ...,
    fill_value: ArrayLike | None = ...,
    sorted: bool = ...,
): ...
def unique_all(
    x: ArrayLike, /, *, size: int | None = ..., fill_value: ArrayLike | None = ...
) -> _UniqueAllResult: ...
def unique_counts(
    x: ArrayLike, /, *, size: int | None = ..., fill_value: ArrayLike | None = ...
) -> _UniqueCountsResult: ...
def unique_inverse(
    x: ArrayLike, /, *, size: int | None = ..., fill_value: ArrayLike | None = ...
) -> _UniqueInverseResult: ...
@overload
def unique_values(
    x: _ArrayValueT, /, *, size: int | None = ..., fill_value: ArrayLike | None = ...
) -> _ArrayValueT: ...
@overload
def unique_values(
    x: ArrayLike, /, *, size: int | None = ..., fill_value: ArrayLike | None = ...
) -> Array: ...
def unpackbits(
    a: ArrayLike,
    axis: int | None = ...,
    count: ArrayLike | None = ...,
    bitorder: str = ...,
) -> Array: ...
def unravel_index(indices: ArrayLike, shape: Shape) -> tuple[Array, ...]: ...

unsignedinteger = _np.unsignedinteger

def unstack(x: ArrayLike, /, *, axis: int = ...) -> tuple[Array, ...]: ...
def unwrap(
    p: ArrayLike,
    discont: ArrayLike | None = ...,
    axis: int = ...,
    period: ArrayLike = ...,
) -> Array: ...
def vander(
    x: ArrayLike, N: int | None = ..., increasing: builtins.bool = ...
) -> Array: ...
def var(
    a: ArrayLike,
    axis: _Axis = ...,
    dtype: DTypeLike | None = ...,
    out: None = ...,
    ddof: int = ...,
    keepdims: builtins.bool = ...,
    *,
    where: ArrayLike | None = ...,
    correction: int | float | None = ...,
) -> Array: ...
def vdot(
    a: ArrayLike,
    b: ArrayLike,
    *,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def vecdot(
    x1: ArrayLike,
    x2: ArrayLike,
    /,
    *,
    axis: int = ...,
    precision: PrecisionLike = ...,
    preferred_element_type: DTypeLike | None = ...,
) -> Array: ...
def vecmat(x1: ArrayLike, x2: ArrayLike, /) -> Array: ...
def vsplit(ary: ArrayLike, indices_or_sections: int | ArrayLike) -> list[Array]: ...
def vstack(
    tup: _np.ndarray | Array | Sequence[ArrayLike], dtype: DTypeLike | None = ...
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: Literal[None] = ...,
    y: Literal[None] = ...,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> tuple[Array, ...]: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> Array: ...
@overload
def where(
    condition: ArrayLike,
    x: ArrayLike | None = ...,
    y: ArrayLike | None = ...,
    /,
    *,
    size: int | None = ...,
    fill_value: None | ArrayLike | tuple[ArrayLike, ...] = ...,
) -> Array | tuple[Array, ...]: ...
def zeros(
    shape: Any,
    dtype: DTypeLike | None = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def zeros_like(
    a: ArrayLike | DuckTypedArray,
    dtype: DTypeLike | None = ...,
    shape: Any = ...,
    *,
    device: _Device | _Sharding | None = ...,
) -> Array: ...
def vectorize(pyfunc, *, excluded=..., signature=...) -> Callable: ...
