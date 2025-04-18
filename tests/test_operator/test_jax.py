"""Test with JAX inputs."""

import operator as ops

import jax.numpy as jnp
import pytest

import quaxed.operator as qops

x = jnp.array([[1, 2], [3, 4]], dtype=float)
y = jnp.array([[5, 6], [7, 8]], dtype=float)
xbit = jnp.array([[1, 0], [0, 1]], dtype=int)


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("abs", (x,), {}),
        ("add", (x, y), {}),
        ("and_", (xbit, xbit), {}),
        ("concat", (x, y), {}),  # this is a+b
        pytest.param(
            "contains", (x, 1), {}, marks=pytest.mark.xfail(reason="array truth value")
        ),
        pytest.param(
            "countOf", (x, 1), {}, marks=pytest.mark.xfail(reason="array truth value")
        ),
        ("eq", (x, y), {}),
        ("floordiv", (x, y), {}),
        ("ge", (x, y), {}),
        ("getitem", (x, slice(0, 1)), {}),
        ("gt", (x, y), {}),
        pytest.param(
            "indexOf", (x, 1), {}, marks=pytest.mark.xfail(reason="array truth value")
        ),
        ("inv", (xbit,), {}),
        ("invert", (xbit,), {}),
        ("is_", (x, x), {}),
        ("is_not", (x, y), {}),
        ("le", (x, y), {}),
        ("lshift", (xbit, 1), {}),
        ("lt", (x, y), {}),
        ("matmul", (x, y), {}),
        ("mod", (x, y), {}),
        ("mul", (x, y), {}),
        ("ne", (x, y), {}),
        ("neg", (x,), {}),
        pytest.param(
            "not_", (xbit,), {}, marks=pytest.mark.xfail(reason="array truth value")
        ),
        ("or_", (xbit, xbit), {}),
        ("pos", (x,), {}),
        ("pow", (x, 2), {}),
        ("rshift", (xbit, 1), {}),
        ("sub", (x, y), {}),
        ("truediv", (x, y), {}),
        ("xor", (xbit, xbit), {}),
    ],
)
def test_lax_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    got = getattr(qops, func_name)(*args, **kw)
    exp = getattr(ops, func_name)(*args, **kw)
    assert jnp.array_equal(got, exp)
