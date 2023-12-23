__all__ = ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]


from jax.experimental import array_api
from jax.experimental.array_api._data_type_functions import FInfo, IInfo
from quax import Value

from ._types import DType
from ._utils import quaxify


@quaxify
def astype(x: Value, dtype: DType, /, *, copy: bool = True) -> Value:
    return array_api.astype(x, dtype, copy=copy)


@quaxify
def can_cast(from_: DType | Value, to: DType, /) -> bool:
    return array_api.can_cast(from_, to)


@quaxify
def finfo(type: DType | Value, /) -> FInfo:
    return array_api.finfo(type)


@quaxify
def iinfo(type: DType | Value, /) -> IInfo:
    return array_api.iinfo(type)


@quaxify
def isdtype(dtype: DType, kind: DType | str | tuple[DType | str, ...]) -> bool:
    return array_api.isdtype(dtype, kind)


@quaxify
def result_type(*arrays_and_dtypes: Value | DType) -> DType:
    return array_api.result_type(*arrays_and_dtypes)
