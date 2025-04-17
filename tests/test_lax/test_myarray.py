"""Test with JAX inputs."""

import jax.numpy as jnp
import jax.tree as jtu
import pytest
from jax import lax

import quaxed.lax as qlax

from ..myarray import MyArray

x = MyArray(jnp.array([[1, 2], [3, 4]], dtype=float))
y = MyArray(jnp.array([[5, 6], [7, 8]], dtype=float))
x1225 = MyArray(jnp.array([[1, 2], [2, 5]], dtype=float))
xtrig = MyArray(jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float))
xtrig2 = MyArray(jnp.array([[0.5, 0.6], [0.7, 0.8]], dtype=float))
xbit = MyArray(jnp.array([[1, 0], [0, 1]], dtype=int))
xcomplex = MyArray(jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]], dtype=complex))
xround = MyArray(jnp.array([[1.1, 2.2], [3.3, 4.4]]))
conv_kernel = MyArray(jnp.array([[[[1.0, 0.0], [0.0, -1.0]]]], dtype=float))
xcomp = MyArray(jnp.array([[5, 2], [7, 2]], dtype=float))


@pytest.mark.parametrize(
    ("func_name", "args", "kw", "expect_myarray"),
    [
        ("abs", (x,), {}, True),
        ("acos", (xtrig,), {}, True),
        ("acosh", (x,), {}, True),
        ("add", (x, y), {}, True),
        pytest.param("after_all", (), {}, True, marks=pytest.mark.skip),
        ("approx_max_k", (x, 2), {}, (True, True)),
        ("approx_min_k", (x, 2), {}, True),
        ("argmax", (x,), {"axis": 0, "index_dtype": int}, True),
        ("argmin", (x,), {"axis": 0, "index_dtype": int}, True),
        ("asin", (xtrig,), {}, True),
        ("asinh", (xtrig,), {}, True),
        ("atan", (xtrig,), {}, True),
        ("atan2", (x, y), {}, True),
        ("atanh", (xtrig,), {}, True),
        (
            "batch_matmul",
            (
                MyArray(jnp.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)),
                MyArray(
                    jnp.array([[[9, 10], [11, 12]], [[13, 14], [15, 16]]], dtype=float)
                ),
            ),
            {},
            True,
        ),
        ("bessel_i0e", (x,), {}, True),
        ("bessel_i1e", (x,), {}, True),
        ("betainc", (1.0, xtrig, xtrig2), {}, True),
        ("bitcast_convert_type", (x, jnp.int32), {}, True),
        ("bitwise_and", (xbit, xbit), {}, True),
        ("bitwise_not", (xbit,), {}, True),
        ("bitwise_or", (xbit, xbit), {}, True),
        ("bitwise_xor", (xbit, xbit), {}, True),
        ("broadcast", (x, (1, 1)), {}, True),
        ("broadcast_in_dim", (x, (1, 1, 2, 2), (2, 3)), {}, True),
        ("broadcast_shapes", ((2, 3), (1, 3)), {}, False),
        ("broadcast_to_rank", (x,), {"rank": 3}, True),
        pytest.param("broadcasted_iota", (), {}, True, marks=pytest.mark.skip),
        ("cbrt", (x,), {}, True),
        ("ceil", (xround,), {}, True),
        ("clamp", (2.0, x, 3.0), {}, True),
        ("clz", (xbit,), {}, True),
        ("collapse", (x, 1), {}, True),
        ("concatenate", ((x, y), 0), {}, True),
        ("conj", (xcomplex,), {}, True),
        (
            "conv",
            (
                MyArray(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4))),
                conv_kernel,
            ),
            {"window_strides": (1, 1), "padding": "SAME"},
            True,
        ),
        ("convert_element_type", (x, jnp.int32), {}, True),
        (
            "conv_dimension_numbers",
            ((1, 4, 4, 1), (2, 2, 1, 1), ("NHWC", "HWIO", "NHWC")),
            {},
            False,
        ),
        (
            "conv_general_dilated",
            (
                MyArray(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4))),
                conv_kernel,
            ),
            {"window_strides": (1, 1), "padding": "SAME"},
            True,
        ),
        pytest.param(
            "conv_general_dilated_local", (), {}, True, marks=pytest.mark.skip
        ),
        (
            "conv_general_dilated_patches",
            (MyArray(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4))),),
            {"filter_shape": (2, 2), "window_strides": (1, 1), "padding": "VALID"},
            True,
        ),
        (
            "conv_transpose",
            (
                MyArray(jnp.arange(1, 17, dtype=float).reshape((1, 1, 4, 4))),
                conv_kernel,
            ),
            {
                "strides": (2, 2),
                "padding": "SAME",
                "dimension_numbers": ("NCHW", "OIHW", "NCHW"),
            },
            True,
        ),
        pytest.param("conv_with_general_padding", (), {}, True, marks=pytest.mark.skip),
        ("cos", (x,), {}, True),
        ("cosh", (x,), {}, True),
        ("cumlogsumexp", (x,), {"axis": 0}, True),
        ("cummax", (x,), {"axis": 0}, True),
        ("cummin", (x,), {"axis": 0}, True),
        ("cumprod", (x,), {"axis": 0}, True),
        ("cumsum", (x,), {"axis": 0}, True),
        ("digamma", (xtrig,), {}, True),
        ("div", (x, y), {}, True),
        ("dot", (x, y), {}, True),
        pytest.param("dot_general", (), {}, True, marks=pytest.mark.skip),
        pytest.param("dynamic_index_in_dim", (), {}, True, marks=pytest.mark.skip),
        ("dynamic_slice", (x, (0, 0), (2, 2)), {}, True),
        pytest.param("dynamic_slice_in_dim", (), {}, True, marks=pytest.mark.skip),
        pytest.param(
            "dynamic_update_index_in_dim", (), {}, True, marks=pytest.mark.skip
        ),
        ("dynamic_update_slice", (x, y, (0, 0)), {}, True),
        ("dynamic_update_slice_in_dim", (x, y, 0, 0), {}, True),
        ("eq", (x, x), {}, True),
        ("erf", (xtrig,), {}, True),
        ("erfc", (xtrig,), {}, True),
        ("erf_inv", (xtrig,), {}, True),
        ("exp", (x,), {}, True),
        ("exp2", (x,), {}, True),
        ("expand_dims", (x, (0,)), {}, True),
        ("expm1", (x,), {}, True),
        ("fft", (x,), {"fft_type": "fft", "fft_lengths": (2, 2)}, True),
        ("floor", (xround,), {}, True),
        ("full", ((2, 2), 1.0), {}, False),
        ("full_like", (x, 1.0), {}, False),
        pytest.param("gather", (), {}, True, marks=pytest.mark.skip),
        ("ge", (x, xcomp), {}, True),
        ("gt", (x, xcomp), {}, True),
        ("igamma", (1.0, xtrig), {}, True),
        ("igammac", (1.0, xtrig), {}, True),
        ("imag", (xcomplex,), {}, True),
        ("index_in_dim", (x, 0, 0), {}, True),
        pytest.param("index_take", (), {}, True, marks=pytest.mark.skip),
        ("integer_pow", (x, 2), {}, True),
        pytest.param("iota", (), {}, True, marks=pytest.mark.skip),
        ("is_finite", (x,), {}, True),
        ("le", (x, xcomp), {}, True),
        ("lgamma", (x,), {}, True),
        ("log", (x,), {}, True),
        ("log1p", (x,), {}, True),
        ("logistic", (x,), {}, True),
        ("lt", (x, jnp.array([[5, 1], [7, 2]], dtype=float)), {}, True),
        ("max", (x, y), {}, True),
        ("min", (x, y), {}, True),
        ("mul", (x, y), {}, True),
        ("ne", (x, xcomp), {}, True),
        ("neg", (x,), {}, True),
        ("nextafter", (x, y), {}, True),
        pytest.param("pad", (), {}, True, marks=pytest.mark.skip),
        ("polygamma", (1.0, xtrig), {}, True),
        ("population_count", (xbit,), {}, True),
        ("pow", (x, y), {}, True),
        ("random_gamma_grad", (1.0, x), {}, True),
        ("real", (xcomplex,), {}, True),
        ("reciprocal", (x,), {}, True),
        pytest.param("reduce", (), {}, True, marks=pytest.mark.skip),
        pytest.param("reduce_precision", (), {}, True, marks=pytest.mark.skip),
        pytest.param("reduce_window", (), {}, True, marks=pytest.mark.skip),
        ("rem", (x, y), {}, True),
        ("reshape", (x, (1, 4)), {}, True),
        ("rev", (x,), {"dimensions": (0,)}, True),
        pytest.param("rng_bit_generator", (), {}, True, marks=pytest.mark.skip),
        ("rng_uniform", (0, 1, (2, 3)), {}, False),
        ("round", (xround,), {}, True),
        ("rsqrt", (x,), {}, True),
        pytest.param("scatter", (), {}, True, marks=pytest.mark.skip),
        pytest.param("scatter_apply", (), {}, True, marks=pytest.mark.skip),
        pytest.param("scatter_max", (), {}, True, marks=pytest.mark.skip),
        pytest.param("scatter_min", (), {}, True, marks=pytest.mark.skip),
        pytest.param("scatter_mul", (), {}, True, marks=pytest.mark.skip),
        ("shift_left", (xbit, 1), {}, True),
        ("shift_right_arithmetic", (xbit, 1), {}, True),
        ("shift_right_logical", (xbit, 1), {}, True),
        ("sign", (x,), {}, True),
        ("sin", (x,), {}, True),
        ("sinh", (x,), {}, True),
        ("slice", (x, (0, 0), (2, 2)), {}, True),
        ("slice_in_dim", (x, 0, 0, 2), {}, True),
        ("sort", (x,), {}, True),
        pytest.param("sort_key_val", (), {}, True, marks=pytest.mark.skip),
        ("sqrt", (x,), {}, True),
        ("square", (x,), {}, True),
        ("sub", (x, y), {}, True),
        ("tan", (x,), {}, True),
        ("tanh", (x,), {}, True),
        ("top_k", (x, 1), {}, True),
        ("transpose", (x, (1, 0)), {}, True),
        ("zeros_like_array", (x,), {}, False),
        ("zeta", (x, 2.0), {}, True),
        pytest.param("associative_scan", (), {}, True, marks=pytest.mark.skip),
        pytest.param("fori_loop", (), {}, True, marks=pytest.mark.skip),
        pytest.param("scan", (), {}, True, marks=pytest.mark.skip),
        (
            "select",
            (jnp.array([[True, False], [True, False]], dtype=bool), x, y),
            {},
            True,
        ),
        pytest.param("select_n", (), {}, True, marks=pytest.mark.skip),
        pytest.param("switch", (), {}, True, marks=pytest.mark.skip),
        ("while_loop", (lambda x: jnp.all(x < 10), lambda x: x + 1, x), {}, True),
        ("stop_gradient", (x,), {}, True),
        pytest.param("custom_linear_solve", (), {}, True, marks=pytest.mark.skip),
        pytest.param("custom_root", (), {}, True, marks=pytest.mark.skip),
        pytest.param("all_gather", (), {}, True, marks=pytest.mark.skip),
        pytest.param("all_to_all", (), {}, True, marks=pytest.mark.skip),
        pytest.param("psum", (), {}, True, marks=pytest.mark.skip),
        pytest.param("psum_scatter", (), {}, True, marks=pytest.mark.skip),
        pytest.param("pmax", (), {}, True, marks=pytest.mark.skip),
        pytest.param("pmin", (), {}, True, marks=pytest.mark.skip),
        pytest.param("pmean", (), {}, True, marks=pytest.mark.skip),
        pytest.param("ppermute", (), {}, True, marks=pytest.mark.skip),
        pytest.param("pshuffle", (), {}, True, marks=pytest.mark.skip),
        pytest.param("pswapaxes", (), {}, True, marks=pytest.mark.skip),
        pytest.param("axis_index", (), {}, True, marks=pytest.mark.skip),
        # --- Sharding-related operators ---
        pytest.param("with_sharding_constraint", (), {}, True, marks=pytest.mark.skip),
    ],
)
def test_lax_functions(func_name, args, kw, expect_myarray):
    """Test lax vs qlax functions."""
    # Jax version
    jax_args, jax_kw = jtu.map(
        lambda x: x.array if isinstance(x, MyArray) else x,
        (args, kw),
        is_leaf=lambda x: isinstance(x, MyArray),
    )
    exp = getattr(lax, func_name)(*jax_args, **jax_kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed version
    got = getattr(qlax, func_name)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)
    got_ = []
    expect_myarray = (
        expect_myarray
        if isinstance(expect_myarray, tuple)
        else (expect_myarray,) * len(got)
    )
    for i, (g, exp_ma) in enumerate(zip(got, expect_myarray, strict=True)):
        if exp_ma:
            assert isinstance(g, MyArray), f"{func_name} return {i} is not MyArray"
        got_.append(g.array if isinstance(g, MyArray) else g)

    assert all([jnp.array_equal(g, e) for g, e in zip(got_, exp, strict=False)])


def test_cond() -> None:
    """Test lax.cond vs qlax.cond with MyArray."""
    exp = lax.cond(True, lambda: x.array, lambda: y.array)  # noqa: FBT003
    got = qlax.cond(True, lambda: x, lambda: y)  # noqa: FBT003

    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, exp)


def test_map() -> None:
    """Test lax.map vs qlax.map with MyArray."""
    exp = lax.map(lambda x: x + 1, x.array)
    got = qlax.map(lambda x: x + 1, x)
    assert isinstance(got, MyArray)
    assert jnp.array_equal(got.array, exp)


@pytest.mark.parametrize(
    ("func_name", "args", "kw", "expect_myarray"),
    [
        ("cholesky", (x1225,), {}, True),
        ("eig", (x1225,), {}, True),
        ("eigh", (x1225,), {}, True),
        ("hessenberg", (x1225,), {}, True),
        ("lu", (x1225,), {}, True),
        (
            "householder_product",
            jtu.map(MyArray, lax.linalg.hessenberg(jnp.array([[1.0, 2], [2, 5]]))),
            {},
            True,
        ),
        ("qdwh", (x1225,), {}, (True, True, False, True)),
        ("qr", (x1225,), {}, True),
        ("schur", (x1225,), {}, True),
        ("svd", (x1225,), {}, True),
        ("tridiagonal", (x1225,), {}, True),
        pytest.param("tridiagonal_solve", (), {}, True, marks=pytest.mark.skip),
    ],
)
def test_lax_linalg_functions(func_name, args, kw, expect_myarray):
    """Test lax vs qlax functions."""
    # Jax version
    jax_args, jax_kw = jtu.map(
        lambda x: x.array if isinstance(x, MyArray) else x,
        (args, kw),
        is_leaf=lambda x: isinstance(x, MyArray),
    )
    exp = getattr(lax.linalg, func_name)(*jax_args, **jax_kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed version
    got = getattr(qlax.linalg, func_name)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)
    got_ = []
    expect_myarray = (
        expect_myarray
        if isinstance(expect_myarray, tuple)
        else (expect_myarray,) * len(got)
    )
    for i, (g, exp_ma) in enumerate(zip(got, expect_myarray, strict=True)):
        if exp_ma:
            assert isinstance(g, MyArray), f"{func_name} return {i} is not MyArray"
        got_.append(g.array if isinstance(g, MyArray) else g)

    assert all(jnp.array_equal(g, e) for g, e in zip(got_, exp, strict=False))
