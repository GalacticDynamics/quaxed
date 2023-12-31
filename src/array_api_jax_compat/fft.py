"""FFT functions."""

__all__ = [
    "fft",
    "ifft",
    "fftn",
    "ifftn",
    "rfft",
    "irfft",
    "rfftn",
    "irfftn",
    "hfft",
    "ihfft",
    "fftfreq",
    "rfftfreq",
    "fftshift",
    "ifftshift",
]

from collections.abc import Sequence
from typing import Literal

from jax import Device
from jax.experimental.array_api import fft as _jax_fft
from quax import Value

from ._utils import quaxify


@quaxify
def fft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.fft(x, n=n, axis=axis, norm=norm)


@quaxify
def ifft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.ifft(x, n=n, axis=axis, norm=norm)


@quaxify
def fftn(
    x: Value,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.fftn(x, s=s, axes=axes, norm=norm)


@quaxify
def ifftn(
    x: Value,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.ifftn(x, s=s, axes=axes, norm=norm)


@quaxify
def rfft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.rfft(x, n=n, axis=axis, norm=norm)


@quaxify
def irfft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.irfft(x, n=n, axis=axis, norm=norm)


@quaxify
def rfftn(
    x: Value,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.rfftn(x, s=s, axes=axes, norm=norm)


@quaxify
def irfftn(
    x: Value,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.irfftn(x, s=s, axes=axes, norm=norm)


@quaxify
def hfft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.hfft(x, n=n, axis=axis, norm=norm)


@quaxify
def ihfft(
    x: Value,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.ihfft(x, n=n, axis=axis, norm=norm)


@quaxify
def fftfreq(n: int, /, *, d: float = 1.0, device: Device | None = None) -> Value:
    return _jax_fft.fftfreq(n, d=d, device=device)


@quaxify
def rfftfreq(n: int, /, *, d: float = 1.0, device: Device | None = None) -> Value:
    return _jax_fft.rfftfreq(n, d=d, device=device)


@quaxify
def fftshift(x: Value, /, *, axes: int | Sequence[int] | None = None) -> Value:
    return _jax_fft.fftshift(x, axes=axes)


@quaxify
def ifftshift(x: Value, /, *, axes: int | Sequence[int] | None = None) -> Value:
    return _jax_fft.ifftshift(x, axes=axes)
