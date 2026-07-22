"""CodSpeed performance benchmarks for :mod:`quaxed`.

These benchmarks exercise the core value proposition of ``quaxed``: applying
:func:`quax.quaxify` to JAX functions so they operate on custom array-ish
objects. We measure the overhead of the quaxed wrappers on a representative
custom array type (:class:`MyArray`), covering elementwise ops, reductions,
autodiff transforms, and the top-level lazy attribute forwarding.

The benchmarks are collected by ``pytest`` and run under CodSpeed via
``pytest-codspeed``. Each benchmarked call performs a single, representative
unit of work; CodSpeed handles warmup and repetition.
"""

import jax.numpy as jnp
import pytest

import quaxed
import quaxed.numpy as qnp
from tests.myarray import MyArray

# Representative inputs. Small arrays keep JAX tracing/dispatch overhead
# dominant, which is exactly what quaxed adds on top of JAX.
x = MyArray(jnp.arange(64.0).reshape(8, 8))
y = MyArray(jnp.arange(64.0).reshape(8, 8) + 1.0)
vec = MyArray(jnp.linspace(0.1, 1.0, 64))


def test_numpy_add(benchmark):
    """Elementwise addition of two custom arrays via quaxed.numpy."""
    result = benchmark(qnp.add, x, y)
    assert isinstance(result, MyArray)


def test_numpy_multiply(benchmark):
    """Elementwise multiplication of two custom arrays."""
    result = benchmark(qnp.multiply, x, y)
    assert isinstance(result, MyArray)


def test_numpy_sin(benchmark):
    """Unary transcendental function over a custom array."""
    result = benchmark(qnp.sin, vec)
    assert isinstance(result, MyArray)


def test_numpy_sum(benchmark):
    """Reduction over a custom array."""
    result = benchmark(qnp.sum, x)
    assert result is not None


def test_numpy_matmul(benchmark):
    """Matrix multiplication over custom arrays."""
    result = benchmark(qnp.matmul, x, y)
    assert isinstance(result, MyArray)


def test_numpy_stack(benchmark):
    """Stacking a list of custom arrays."""
    result = benchmark(qnp.stack, [x, y])
    assert isinstance(result, MyArray)


def test_grad(benchmark):
    """Autodiff via the quaxed ``grad`` transform."""

    def f(v):
        return qnp.sum(qnp.square(v))

    grad_f = quaxed.grad(f)
    result = benchmark(grad_f, vec)
    assert result is not None


def test_value_and_grad(benchmark):
    """Combined value-and-grad transform through quaxed."""

    def f(v):
        return qnp.sum(qnp.square(v))

    vg = quaxed.value_and_grad(f)
    result = benchmark(vg, vec)
    assert result is not None


@pytest.mark.parametrize("name", ["abs", "exp", "sqrt", "tanh"])
def test_numpy_unary_ops(benchmark, name):
    """A spread of quaxed.numpy unary ops on a custom array."""
    func = getattr(qnp, name)
    result = benchmark(func, vec)
    assert isinstance(result, MyArray)
