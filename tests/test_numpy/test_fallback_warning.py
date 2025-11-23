"""Test fallback warnings for numpy functions not in quaxed.numpy.__all__."""

import warnings
from operator import itemgetter

import jax.numpy as jnp
import jax.random as jr
import pytest

import quaxed.numpy as qnp


@pytest.fixture
def key(request):
    """Provide a JAX random key seeded by the test function name."""
    seed = hash(request.node.name) % (2**32)
    return jr.key(seed)


def test_missing_functions_raise_warning(key):
    """Test missing quaxed.numpy functions raise UserWarning, fall back to numpy."""
    # Get all callable functions from numpy that are in numpy.__all__
    numpy_functions = {
        name for name in dir(jnp) if hasattr(jnp, name) and callable(getattr(jnp, name))
    }

    # Get all functions available in quaxed.numpy
    quaxed_functions = set(qnp.__all__)  # type: ignore[attr-defined]

    # Find functions that are in numpy but not in quaxed
    missing_functions = numpy_functions - quaxed_functions

    # Test a sample of missing functions
    key, subkey = jr.split(key)
    idx = jr.choice(subkey, len(missing_functions), (5,)).tolist()
    sample_functions = (
        itemgetter(*idx)(sorted(missing_functions)) if missing_functions else []
    )

    for func_name in sample_functions:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Access the function through quaxed.numpy
            func = getattr(qnp, func_name)

            # Verify a warning was raised
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert f"Missing `quaxed.numpy.{func_name}`" in str(w[0].message)
            assert f"Falling back to `jax.numpy.{func_name}`" in str(w[0].message)

            # Verify it actually returns the numpy function
            assert func is getattr(jnp, func_name)


def test_available_functions_no_warning():
    """Test that functions in quaxed.numpy.__all__ don't raise warnings."""
    # Test a few functions that should be available
    test_functions = ["array", "zeros", "ones"]

    for func_name in test_functions:
        if func_name in qnp.__all__:  # type: ignore[attr-defined]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")

                # Access the function
                _ = getattr(qnp, func_name)

                # Verify no warning was raised
                assert len(w) == 0


def test_nonexistent_attribute_raises_error():
    """Test that truly nonexistent attributes raise AttributeError."""
    with pytest.raises(AttributeError, match="has no attribute 'nonexistent_function'"):
        _ = qnp.nonexistent_function  # type: ignore[attr-defined]
