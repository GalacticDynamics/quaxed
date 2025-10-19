"""Doctest configuration."""

from optional_dependencies import OptionalDependencyEnum, auto


class OptDeps(OptionalDependencyEnum):
    """Dependencies for `quaxed`."""

    JAX = auto()
