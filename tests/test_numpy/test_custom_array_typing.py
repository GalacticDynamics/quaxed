"""Tests for typing of custom array types in quaxed.numpy."""

from dataclasses import dataclass
from typing import assert_type

import jax
import jax.numpy as jnp
import quax

import quaxed.numpy as qnp


@dataclass
class SimpleArray(quax.ArrayValue):
    """A simple :class:`quax.ArrayValue` subclass for testing typing."""

    array: jax.Array

    def materialise(self) -> jax.Array:  # noqa: D102
        return self.array

    def aval(self) -> jax.core.ShapedArray:  # noqa: D102
        return jax.core.ShapedArray(self.array.shape, self.array.dtype)


def test_simple_array_typing():
    result: SimpleArray = qnp.sin(SimpleArray(jnp.array([1.4])))
    assert_type(result, SimpleArray)

    assert_type(
        qnp.add(SimpleArray(jnp.array([1.0])), SimpleArray(jnp.array([2.0]))),
        SimpleArray,
    )
    assert_type(qnp.add(SimpleArray(jnp.array([2.0])), jnp.array([1.0])), SimpleArray)
