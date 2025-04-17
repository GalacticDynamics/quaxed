"""Test with JAX inputs."""

import jax.numpy as jnp
import pytest
from jax import lax

import quaxed.lax as qlax

x = jnp.array([[1, 2], [3, 4]], dtype=float)
y = jnp.array([[5, 6], [7, 8]], dtype=float)
x1225 = jnp.array([[1, 2], [2, 5]], dtype=float)
xtrig = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
xtrig2 = jnp.array([[0.5, 0.6], [0.7, 0.8]], dtype=float)
xbit = jnp.array([[1, 0], [0, 1]], dtype=int)
xcomplex = jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=complex)
xround = jnp.array([[1.1, 2.2], [3.3, 4.4]])
conv_kernel = jnp.array([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=float)
xcomp = jnp.array([[5, 2], [7, 2]], dtype=float)


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("abs", (x,), {}),
        ("acos", (xtrig,), {}),
        ("acosh", (x,), {}),
        ("add", (x, y), {}),
        pytest.param("after_all", (), {}, marks=pytest.mark.skip),
        ("approx_max_k", (x, 2), {}),
        ("approx_min_k", (x, 2), {}),
        ("argmax", (x,), {"axis": 0, "index_dtype": int}),
        ("argmin", (x,), {"axis": 0, "index_dtype": int}),
        ("asin", (xtrig,), {}),
        ("asinh", (xtrig,), {}),
        ("atan", (xtrig,), {}),
        ("atan2", (x, y), {}),
        ("atanh", (xtrig,), {}),
        (
            "batch_matmul",
            (
                jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float),
                jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=float),
            ),
            {},
        ),
        ("bessel_i0e", (x,), {}),
        ("bessel_i1e", (x,), {}),
        ("betainc", (1.0, xtrig, xtrig2), {}),
        ("bitcast_convert_type", (x, jnp.int32), {}),
        ("bitwise_and", (xbit, xbit), {}),
        ("bitwise_not", (xbit,), {}),
        ("bitwise_or", (xbit, xbit), {}),
        ("bitwise_xor", (xbit, xbit), {}),
        ("broadcast", (x, (1, 1)), {}),
        ("broadcast_in_dim", (x, (1, 1, 2, 2), (2, 3)), {}),
        ("broadcast_shapes", ((2, 3), (1, 3)), {}),
        ("broadcast_to_rank", (x,), {"rank": 3}),
        pytest.param("broadcasted_iota", (), {}, marks=pytest.mark.skip),
        ("cbrt", (x,), {}),
        ("ceil", (xround,), {}),
        ("clamp", (2.0, x, 3.0), {}),
        ("clz", (xbit,), {}),
        ("collapse", (x, 1), {}),
        ("concatenate", ((x, y), 0), {}),
        ("conj", (xcomplex,), {}),
        (
            "conv",
            (jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), conv_kernel),
            {"window_strides": (1, 1), "padding": "SAME"},
        ),
        ("convert_element_type", (x, jnp.int32), {}),
        (
            "conv_dimension_numbers",
            ((1, 4, 4, 1), (2, 2, 1, 1), ("NHWC", "HWIO", "NHWC")),
            {},
        ),
        (
            "conv_general_dilated",
            (jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), conv_kernel),
            {"window_strides": (1, 1), "padding": "SAME"},
        ),
        pytest.param("conv_general_dilated_local", (), {}, marks=pytest.mark.skip),
        (
            "conv_general_dilated_patches",
            (jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)),),
            {"filter_shape": (2, 2), "window_strides": (1, 1), "padding": "VALID"},
        ),
        (
            "conv_transpose",
            (jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4)), conv_kernel),
            {
                "strides": (2, 2),
                "padding": "SAME",
                "dimension_numbers": ("NCHW", "OIHW", "NCHW"),
            },
        ),
        pytest.param("conv_with_general_padding", (), {}, marks=pytest.mark.skip),
        ("cos", (x,), {}),
        ("cosh", (x,), {}),
        ("cumlogsumexp", (x,), {"axis": 0}),
        ("cummax", (x,), {"axis": 0}),
        ("cummin", (x,), {"axis": 0}),
        ("cumprod", (x,), {"axis": 0}),
        ("cumsum", (x,), {"axis": 0}),
        ("digamma", (xtrig,), {}),
        ("div", (x, y), {}),
        ("dot", (x, y), {}),
        pytest.param("dot_general", (), {}, marks=pytest.mark.skip),
        pytest.param("dynamic_index_in_dim", (), {}, marks=pytest.mark.skip),
        ("dynamic_slice", (x, (0, 0), (2, 2)), {}),
        pytest.param("dynamic_slice_in_dim", (), {}, marks=pytest.mark.skip),
        pytest.param("dynamic_update_index_in_dim", (), {}, marks=pytest.mark.skip),
        ("dynamic_update_slice", (x, y, (0, 0)), {}),
        ("dynamic_update_slice_in_dim", (x, y, 0, 0), {}),
        ("eq", (x, x), {}),
        ("erf", (xtrig,), {}),
        ("erfc", (xtrig,), {}),
        ("erf_inv", (xtrig,), {}),
        ("exp", (x,), {}),
        ("exp2", (x,), {}),
        ("expand_dims", (x, (0,)), {}),
        ("expm1", (x,), {}),
        ("fft", (x,), {"fft_type": "fft", "fft_lengths": (2, 2)}),
        ("floor", (xround,), {}),
        ("full", ((2, 2), 1.0), {}),
        ("full_like", (x, 1.0), {}),
        pytest.param("gather", (), {}, marks=pytest.mark.skip),
        ("ge", (x, xcomp), {}),
        ("gt", (x, xcomp), {}),
        ("igamma", (1.0, xtrig), {}),
        ("igammac", (1.0, xtrig), {}),
        ("imag", (xcomplex,), {}),
        ("index_in_dim", (x, 0, 0), {}),
        pytest.param("index_take", (), {}, marks=pytest.mark.skip),
        ("integer_pow", (x, 2), {}),
        pytest.param("iota", (), {}, marks=pytest.mark.skip),
        ("is_finite", (x,), {}),
        ("le", (x, xcomp), {}),
        ("lgamma", (x,), {}),
        ("log", (x,), {}),
        ("log1p", (x,), {}),
        ("logistic", (x,), {}),
        ("lt", (x, jnp.array([[5, 1], [7, 2]], dtype=float)), {}),
        ("max", (x, y), {}),
        ("min", (x, y), {}),
        ("mul", (x, y), {}),
        ("ne", (x, xcomp), {}),
        ("neg", (x,), {}),
        ("nextafter", (x, y), {}),
        pytest.param("pad", (), {}, marks=pytest.mark.skip),
        ("polygamma", (1.0, xtrig), {}),
        ("population_count", (xbit,), {}),
        ("pow", (x, y), {}),
        ("random_gamma_grad", (1.0, x), {}),
        ("real", (xcomplex,), {}),
        ("reciprocal", (x,), {}),
        pytest.param("reduce", (), {}, marks=pytest.mark.skip),
        pytest.param("reduce_precision", (), {}, marks=pytest.mark.skip),
        pytest.param("reduce_window", (), {}, marks=pytest.mark.skip),
        ("rem", (x, y), {}),
        ("reshape", (x, (1, 4)), {}),
        ("rev", (x,), {"dimensions": (0,)}),
        pytest.param("rng_bit_generator", (), {}, marks=pytest.mark.skip),
        ("rng_uniform", (0, 1, (2, 3)), {}),
        ("round", (xround,), {}),
        ("rsqrt", (x,), {}),
        pytest.param("scatter", (), {}, marks=pytest.mark.skip),
        pytest.param("scatter_apply", (), {}, marks=pytest.mark.skip),
        pytest.param("scatter_max", (), {}, marks=pytest.mark.skip),
        pytest.param("scatter_min", (), {}, marks=pytest.mark.skip),
        pytest.param("scatter_mul", (), {}, marks=pytest.mark.skip),
        ("shift_left", (xbit, 1), {}),
        ("shift_right_arithmetic", (xbit, 1), {}),
        ("shift_right_logical", (xbit, 1), {}),
        ("sign", (x,), {}),
        ("sin", (x,), {}),
        ("sinh", (x,), {}),
        ("slice", (x, (0, 0), (2, 2)), {}),
        ("slice_in_dim", (x, 0, 0, 2), {}),
        ("sort", (x,), {}),
        pytest.param("sort_key_val", (), {}, marks=pytest.mark.skip),
        ("sqrt", (x,), {}),
        ("square", (x,), {}),
        ("sub", (x, y), {}),
        ("tan", (x,), {}),
        ("tanh", (x,), {}),
        ("top_k", (x, 1), {}),
        ("transpose", (x, (1, 0)), {}),
        ("zeros_like_array", (x,), {}),
        ("zeta", (x, 2.0), {}),
        pytest.param("associative_scan", (), {}, marks=pytest.mark.skip),
        ("cond", (True, lambda: x, lambda: y), {}),
        pytest.param("fori_loop", (), {}, marks=pytest.mark.skip),
        ("map", (lambda x: x + 1, x), {}),
        pytest.param("scan", (), {}, marks=pytest.mark.skip),
        (
            "select",
            (jnp.array([[True, False], [True, False]], dtype=bool), x, y),
            {},
        ),
        pytest.param("select_n", (), {}, marks=pytest.mark.skip),
        pytest.param("switch", (), {}, marks=pytest.mark.skip),
        ("while_loop", (lambda x: jnp.all(x < 10), lambda x: x + 1, x), {}),
        ("stop_gradient", (x,), {}),
        pytest.param("custom_linear_solve", (), {}, marks=pytest.mark.skip),
        pytest.param("custom_root", (), {}, marks=pytest.mark.skip),
        pytest.param("all_gather", (), {}, marks=pytest.mark.skip),
        pytest.param("all_to_all", (), {}, marks=pytest.mark.skip),
        pytest.param("psum", (), {}, marks=pytest.mark.skip),
        pytest.param("psum_scatter", (), {}, marks=pytest.mark.skip),
        pytest.param("pmax", (), {}, marks=pytest.mark.skip),
        pytest.param("pmin", (), {}, marks=pytest.mark.skip),
        pytest.param("pmean", (), {}, marks=pytest.mark.skip),
        pytest.param("ppermute", (), {}, marks=pytest.mark.skip),
        pytest.param("pshuffle", (), {}, marks=pytest.mark.skip),
        pytest.param("pswapaxes", (), {}, marks=pytest.mark.skip),
        pytest.param("axis_index", (), {}, marks=pytest.mark.skip),
        # --- Sharding-related operators ---
        pytest.param("with_sharding_constraint", (), {}, marks=pytest.mark.skip),
    ],
)
def test_lax_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    got = getattr(qlax, func_name)(*args, **kw)
    exp = getattr(lax, func_name)(*args, **kw)
    assert jnp.array_equal(got, exp)


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("cholesky", (x1225,), {}),
        ("eig", (x1225,), {}),
        ("eigh", (x1225,), {}),
        ("hessenberg", (x1225,), {}),
        ("lu", (x1225,), {}),
        (
            "householder_product",
            lax.linalg.hessenberg(jnp.array([[1.0, 2], [2, 5]])),
            {},
        ),
        ("qdwh", (x1225,), {}),
        ("qr", (x1225,), {}),
        ("schur", (x1225,), {}),
        ("svd", (x1225,), {}),
        ("tridiagonal", (x1225,), {}),
        pytest.param("tridiagonal_solve", (), {}, marks=pytest.mark.skip),
    ],
)
def test_lax_linalg_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    got = getattr(qlax.linalg, func_name)(*args, **kw)
    exp = getattr(lax.linalg, func_name)(*args, **kw)

    got = got if isinstance(got, tuple | list) else (got,)
    exp = exp if isinstance(exp, tuple | list) else (exp,)
    assert all(jnp.array_equal(g, e) for g, e in zip(got, exp, strict=False))
