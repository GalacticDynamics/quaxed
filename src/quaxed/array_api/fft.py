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
from jaxtyping import ArrayLike
from quax import Value

from quaxed._utils import quaxify


@quaxify
def fft(
    x: ArrayLike,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.fft(x, n=n, axis=axis, norm=norm)


@quaxify
def ifft(
    x: ArrayLike,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.ifft(x, n=n, axis=axis, norm=norm)


@quaxify
def fftn(
    x: ArrayLike,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.fftn(x, s=s, axes=axes, norm=norm)


@quaxify
def ifftn(
    x: ArrayLike,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.ifftn(x, s=s, axes=axes, norm=norm)


@quaxify
def rfft(
    x: ArrayLike,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.rfft(x, n=n, axis=axis, norm=norm)


@quaxify
def irfft(
    x: ArrayLike,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.irfft(x, n=n, axis=axis, norm=norm)


@quaxify
def rfftn(
    x: ArrayLike,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.rfftn(x, s=s, axes=axes, norm=norm)


@quaxify
def irfftn(
    x: ArrayLike,
    /,
    *,
    s: Sequence[int] | None = None,
    axes: Sequence[int] | None = None,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.irfftn(x, s=s, axes=axes, norm=norm)


@quaxify
def hfft(
    x: ArrayLike,
    /,
    *,
    n: int | None = None,
    axis: int = -1,
    norm: Literal["backward", "ortho", "forward"] = "backward",
) -> Value:
    return _jax_fft.hfft(x, n=n, axis=axis, norm=norm)


@quaxify
def ihfft(
    x: ArrayLike,
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
def fftshift(x: ArrayLike, /, *, axes: int | Sequence[int] | None = None) -> Value:
    return _jax_fft.fftshift(x, axes=axes)


@quaxify
def ifftshift(x: ArrayLike, /, *, axes: int | Sequence[int] | None = None) -> Value:
    return _jax_fft.ifftshift(x, axes=axes)
