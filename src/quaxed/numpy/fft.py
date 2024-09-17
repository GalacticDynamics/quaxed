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


from jax.numpy import fft as _jax_fft

from quaxed._utils import quaxify

fft = quaxify(_jax_fft.fft)
ifft = quaxify(_jax_fft.ifft)
fftn = quaxify(_jax_fft.fftn)
ifftn = quaxify(_jax_fft.ifftn)
rfft = quaxify(_jax_fft.rfft)
irfft = quaxify(_jax_fft.irfft)
rfftn = quaxify(_jax_fft.rfftn)
irfftn = quaxify(_jax_fft.irfftn)
hfft = quaxify(_jax_fft.hfft)
ihfft = quaxify(_jax_fft.ihfft)
fftfreq = quaxify(_jax_fft.fftfreq)
rfftfreq = quaxify(_jax_fft.rfftfreq)
fftshift = quaxify(_jax_fft.fftshift)
ifftshift = quaxify(_jax_fft.ifftshift)
