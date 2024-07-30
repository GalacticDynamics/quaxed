__all__ = ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]


from jax.experimental import array_api
from jaxtyping import ArrayLike

from quaxed._types import DType
from quaxed._utils import JAX_VERSION, quaxify

if JAX_VERSION < (0, 4, 31):
    from jax.experimental.array_api._data_type_functions import FInfo
else:
    from numpy import finfo as FInfo  # noqa: N812


if JAX_VERSION < (0, 4, 27):
    from jax.experimental.array_api._data_type_functions import IInfo
else:
    from jax.experimental.array_api import iinfo as IInfo  # noqa: N812


@quaxify
def astype(x: ArrayLike, dtype: DType, /, *, copy: bool = True) -> ArrayLike:
    return array_api.astype(x, dtype, copy=copy)


@quaxify
def can_cast(from_: DType | ArrayLike, to: DType, /) -> bool:
    return array_api.can_cast(from_, to)


@quaxify
def finfo(type: DType | ArrayLike, /) -> FInfo:
    return array_api.finfo(type)


@quaxify
def iinfo(type: DType | ArrayLike, /) -> IInfo:
    return array_api.iinfo(type)


isdtype = quaxify(array_api.isdtype)


@quaxify
def result_type(*arrays_and_dtypes: ArrayLike | DType) -> DType:
    return array_api.result_type(*arrays_and_dtypes)
