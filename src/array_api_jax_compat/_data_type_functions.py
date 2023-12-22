__all__ = ["astype", "can_cast", "finfo", "iinfo", "isdtype", "result_type"]


import jax
from jax import Device
from quax import Value

from ._types import DType
from ._utils import quaxify


@quaxify
def astype(
    x: Value,
    dtype: DType,
    /,
    *,
    copy: bool = True,  # TODO: support  # pylint: disable=unused-argument
    device: Device | None = None,
) -> Value:
    out = jax.lax.convert_element_type(x, dtype)
    return jax.device_put(out, device=device)


@quaxify
def can_cast(from_: DType | Value, to: DType, /) -> bool:
    return jax.numpy.can_cast(from_, to)


@quaxify
def finfo(type: DType | Value, /) -> jax.numpy.finfo:
    return jax.numpy.finfo(type)


@quaxify
def iinfo(type: DType | Value, /) -> jax.numpy.iinfo:
    return jax.numpy.iinfo(type)


@quaxify
def isdtype(dtype: DType, kind: DType | str | tuple[DType | str, ...]) -> bool:
    raise NotImplementedError


@quaxify
def result_type(*arrays_and_dtypes: Value | DType) -> DType:
    return jax.numpy.result_type(*arrays_and_dtypes)
