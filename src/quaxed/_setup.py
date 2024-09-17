"""Setup file for the Quaxed package."""

from importlib.metadata import version

JAX_VERSION: tuple[int, ...] = tuple(map(int, version("jax").split(".")))
