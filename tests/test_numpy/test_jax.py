"""Test with JAX inputs."""

import tempfile

import jax.numpy as jnp
import jax.random as jr
import jax.tree as jtu
import jax.tree_util as jt
import numpy as np
import pytest

import quaxed.numpy as qnp

xfail_quax58 = pytest.mark.xfail(
    reason="https://github.com/patrick-kidger/quax/issues/58"
)
mark_todo = pytest.mark.skip("TODO")

x = jnp.array([[1, 2], [3, 4]], dtype=float)
y = jnp.array([[5, 6], [7, 8]], dtype=float)
xtrig = jnp.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)
xbool = jnp.array([True, False, True], dtype=bool)


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("abs", (x,), {}),
        ("absolute", (x,), {}),
        ("acos", (xtrig,), {}),
        ("acosh", (x,), {}),
        ("add", (x, y), {}),
        ("all", (xbool,), {}),
        *[("allclose", (x, x), {}), ("allclose", (x, y), {})],
        ("amax", (x,), {}),
        ("amin", (x,), {}),
        ("angle", (x,), {}),
        ("any", (xbool,), {}),
        ("append", (x, 4), {}),
        ("apply_along_axis", (lambda x: x + 1, 0, x), {}),
        ("apply_over_axes", (lambda x, _: x + 1, x, [0, 1]), {}),
        ("arange", (3,), {}),
        ("arccos", (xtrig,), {}),
        ("arccosh", (x,), {}),
        ("arcsin", (xtrig,), {}),
        ("arcsinh", (xtrig,), {}),
        ("arctan", (xtrig,), {}),
        ("arctan2", (x, y), {}),
        ("arctanh", (xtrig,), {}),
        ("argmax", (x,), {}),
        ("argmin", (x,), {}),
        ("argpartition", (x, 1), {}),
        ("argsort", (x,), {}),
        *(
            pytest.param("argwhere", (x,), {}, marks=xfail_quax58),
            ("argwhere", (x,), {"size": x.size}),  # TODO: not need static size
        ),
        ("around", (x,), {}),
        ("array", (x,), {}),
        ("array_equal", (x, y), {}),
        ("array_equiv", (x, y), {}),
        pytest.param("array_str", (x,), {}, marks=mark_todo),
        ("asarray", (x,), {}),
        ("asin", (xtrig,), {}),
        ("asinh", (xtrig,), {}),
        ("astype", (x, int), {}),
        ("atan", (xtrig,), {}),
        ("atan2", (x, y), {}),
        ("atanh", (xtrig,), {}),
        ("atleast_1d", (x,), {}),
        ("atleast_2d", (x,), {}),
        ("atleast_3d", (x,), {}),
        ("average", (x,), {}),
        ("bartlett", (3,), {}),
        *(
            pytest.param(
                "bincount", (jnp.asarray([0, 1, 1, 2, 2, 2]),), {}, marks=xfail_quax58
            ),
            ("bincount", (jnp.asarray([0, 1, 1, 2, 2, 2]),), {"length": 3}),
        ),
        ("bitwise_and", (xbool, xbool), {}),
        ("bitwise_count", (xbool,), {}),
        ("bitwise_invert", (xbool,), {}),
        ("bitwise_left_shift", (xbool, 1), {}),
        ("bitwise_not", (xbool,), {}),
        ("bitwise_or", (xbool, xbool), {}),
        ("bitwise_right_shift", (xbool, 1), {}),
        ("bitwise_xor", (xbool, xbool), {}),
        ("blackman", (3,), {}),
        ("block", ([x, x],), {}),
        ("bool", (x,), {}),
        ("bool_", (x,), {}),
        ("broadcast_arrays", (x, y), {}),
        ("broadcast_shapes", (x.shape, y.shape), {}),
        ("broadcast_to", (x, (2, 2)), {}),
        ("can_cast", (x, int), {}),
        ("cbrt", (x,), {}),
        ("ceil", (x,), {}),
        pytest.param("choose", (0, [x, x]), {}, marks=xfail_quax58),
        ("clip", (x, 1, 2), {}),
        ("column_stack", ([x, x],), {}),
        pytest.param("complex128", (1,), {}, marks=pytest.mark.xfail),
        ("complex64", (1,), {}),
        pytest.param("complex_", (1,), {}, marks=pytest.mark.xfail),
        pytest.param("complexfloating", (1,), {}, marks=pytest.mark.xfail),
        pytest.param("compress", (xbool, x), {}, marks=xfail_quax58),
        ("concat", ([x, x],), {}),
        ("concatenate", ([x, x],), {}),
        ("conj", (x,), {}),
        ("conjugate", (x,), {}),
        ("convolve", (x[:, 0], y[:, 0]), {}),
        ("copy", (x,), {}),
        ("copysign", (x, y), {}),
        ("corrcoef", (x,), {}),
        ("correlate", (x[:, 0], y[:, 0]), {}),
        ("cos", (x,), {}),
        ("cosh", (x,), {}),
        ("count_nonzero", (x,), {}),
        ("cov", (x,), {}),
        ("cross", (x, y), {}),
        ("csingle", (1,), {}),
        ("cumprod", (x,), {}),
        ("cumsum", (x,), {}),
        ("deg2rad", (x,), {}),
        ("degrees", (x,), {}),
        ("delete", (x, 1), {}),
        ("diag", (x,), {}),
        ("diag_indices", (5,), {}),
        ("diag_indices_from", (jnp.eye(4),), {}),
        ("diagflat", (x,), {}),
        ("diagonal", (jnp.eye(4),), {}),
        ("diff", (x,), {}),
        ("digitize", (x, jnp.asarray([0, 2, 4])), {}),
        ("divide", (x, y), {}),
        ("divmod", (x, 2), {}),
        ("dot", (x, y), {}),
        pytest.param("double", (x,), {}, marks=pytest.mark.xfail),
        ("dsplit", (jnp.arange(16.0).reshape(2, 2, 4), 2), {}),
        ("dstack", (x,), {}),
        ("ediff1d", (x,), {}),
        ("einsum", ("ij,jk->ik", x, y), {}),
        # ("einsum_path", ("ij,jk,kl->il", rand1, rand2, rand3), {"optimize": "greedy"}),  # TODO: replace independent test with this  # noqa: E501
        ("empty", (4,), {}),
        ("empty_like", (x,), {}),
        ("equal", (x, y), {}),
        ("exp", (x,), {}),
        ("exp2", (x,), {}),
        ("expand_dims", (x, 0), {}),
        ("expm1", (x,), {}),
        pytest.param("extract", (jnp.array([True]), x), {}, marks=xfail_quax58),
        ("eye", (2,), {}),
        ("fabs", (x,), {}),
        ("fill_diagonal", (jnp.eye(3), 2), {"inplace": False}),
        ("fix", (x,), {}),
        *(
            pytest.param("flatnonzero", (x,), {}, marks=xfail_quax58),
            ("flatnonzero", (x,), {"size": x.size}),
        ),
        ("flip", (x,), {}),
        ("fliplr", (jnp.arange(4).reshape(2, 2),), {}),
        ("flipud", (x,), {}),
        ("float16", (x,), {}),
        ("float32", (x,), {}),
        pytest.param("float64", (x,), {}, marks=mark_todo),
        ("float8_e4m3b11fnuz", (x,), {}),
        ("float8_e4m3fn", (x,), {}),
        ("float8_e4m3fnuz", (x,), {}),
        ("float8_e5m2", (x,), {}),
        ("float8_e5m2fnuz", (x,), {}),
        pytest.param(
            "float_",
            (x,),
            {},
            marks=pytest.mark.skipif(hasattr(jnp, "float_"), reason="not available"),
        ),
        ("float_power", (x, y), {}),
        ("floor", (x,), {}),
        ("floor_divide", (x, y), {}),
        ("fmax", (x, y), {}),
        ("fmin", (x, y), {}),
        ("fmod", (x, y), {}),
        ("frexp", (x,), {}),
        ("from_dlpack", (x,), {}),
        ("frombuffer", (b"\x01\x02",), {"dtype": jnp.uint8}),
        ("fromfunction", ((lambda i, _: i), (2, 2)), {"dtype": float}),
        ("fromstring", ("1 2",), {"dtype": int, "sep": " "}),
        ("full", ((2, 2), 4.0), {}),
        ("full_like", (x, 4.0), {}),
        ("gcd", (jnp.array([12, 8, 32]), jnp.array([4, 4, 4])), {}),
        ("geomspace", (1.0, 100.0), {}),
        ("gradient", (x,), {}),
        ("greater", (x, y), {}),
        ("greater_equal", (x, y), {}),
        ("hamming", (2,), {}),
        ("hanning", (2,), {}),
        ("heaviside", (x, y), {}),
        ("histogram", (x,), {}),
        ("histogram2d", (x[:, 0], x[:, 1]), {}),
        ("histogram_bin_edges", (x,), {"bins": 3}),
        ("histogramdd", (x,), {"bins": 3}),
        ("hsplit", (x, 2), {}),
        ("hstack", ([x, x],), {}),
        ("hypot", (x, y), {}),
        ("i0", (x,), {}),
        ("identity", (4,), {}),
        ("imag", (x,), {}),
        ("inner", (x, y), {}),
        ("insert", (x, 1, 2.0), {}),
        ("interp", (x[:, 0], x[:, 0], 2 * x[:, 0]), {}),
        *(
            pytest.param("intersect1d", (x[:, 0], x[:, 0]), {}, marks=xfail_quax58),
            ("intersect1d", (x[:, 0], x[:, 0]), {"size": x[:, 0].size}),
        ),
        ("invert", (x.astype(int),), {}),
        ("isclose", (x, y), {}),
        ("iscomplex", (x,), {}),
        ("iscomplexobj", (x,), {}),
        ("isdtype", (x, "real floating"), {}),
        ("isfinite", (x,), {}),
        ("isin", (2 * jnp.arange(4).reshape((2, 2)), jnp.asarray([1, 2, 4, 8])), {}),
        ("isinf", (x,), {}),
        ("isnan", (x,), {}),
        ("isneginf", (x,), {}),
        ("isposinf", (x,), {}),
        ("isreal", (x,), {}),
        ("isrealobj", (x,), {}),
        ("isscalar", (x,), {}),
        ("issubdtype", (x.dtype, jnp.float32), {}),
        ("iterable", (x,), {}),
        ("ix_", (x[:, 0], y[:, 0]), {}),
        ("kaiser", (4, jnp.asarray(3.0)), {}),
        ("kron", (x, x), {}),
        ("lcm", (x.astype(int), y.astype(int)), {}),
        ("ldexp", (x, y.astype(int)), {}),
        ("left_shift", (x.astype(int), y.astype(int)), {}),
        ("less", (x, y), {}),
        ("less_equal", (x, y), {}),
        ("lexsort", (x,), {}),
        ("linspace", (2.0, 3.0, 5), {}),
        ("log", (x,), {}),
        ("log10", (x,), {}),
        ("log1p", (x,), {}),
        ("log2", (x,), {}),
        ("logaddexp", (x, y), {}),
        ("logaddexp2", (x, y), {}),
        ("logical_and", (x, y), {}),
        ("logical_not", (x,), {}),
        ("logical_or", (x, y), {}),
        ("logical_xor", (x, y), {}),
        ("logspace", (0.0, 10.0), {}),
        ("matmul", (x, y), {}),
        ("matrix_transpose", (x,), {}),
        ("max", (x,), {}),
        ("maximum", (x, y), {}),
        ("mean", (x,), {}),
        ("median", (x,), {}),
        ("meshgrid", (x[:, 0], y[:, 0]), {}),
        ("min", (x,), {}),
        ("minimum", (x, y), {}),
        ("mod", (x, 2), {}),
        ("modf", (x,), {}),
        ("moveaxis", (x[None], 0, 1), {}),
        ("multiply", (x, y), {}),
        ("nan_to_num", (x,), {}),
        ("nanargmax", (x,), {}),
        ("nanargmin", (x,), {}),
        ("nancumprod", (x,), {}),
        ("nancumsum", (x,), {}),
        ("nanmax", (x,), {}),
        ("nanmean", (x,), {}),
        ("nanmedian", (x,), {}),
        ("nanmin", (x,), {}),
        ("nanpercentile", (x, 50), {}),
        ("nanprod", (x,), {}),
        ("nanquantile", (x, 0.5), {}),
        ("nanstd", (x,), {}),
        ("nansum", (x,), {}),
        ("nanvar", (x,), {}),
        ("ndim", (x,), {}),
        ("negative", (x,), {}),
        ("nextafter", (x, y), {}),
        *(
            pytest.param("nonzero", (x,), {}, marks=xfail_quax58),
            ("nonzero", (x,), {"size": x.size}),
        ),
        ("not_equal", (x, y), {}),
        ("ones", (5,), {}),
        ("ones_like", (x,), {}),
        ("outer", (x, y), {}),
        ("packbits", (jnp.array([[[1, 0, 1], [0, 1, 0]]], dtype=jnp.uint8),), {}),
        ("pad", (x, 20), {}),
        ("partition", (x, 1), {}),
        ("percentile", (x, 50), {}),
        ("permute_dims", (x, (0, 1)), {}),
        ("piecewise", (x, [x < 0, x >= 0], [-1, 1]), {}),
        ("place", (x, x > jnp.mean(x), 0), {"inplace": False}),
        ("poly", (x,), {}),
        pytest.param("polyadd", (x,), {}, marks=mark_todo),
        pytest.param("polyder", (x,), {}, marks=mark_todo),
        pytest.param("polydiv", (x,), {}, marks=mark_todo),
        pytest.param("polyfit", (x,), {}, marks=mark_todo),
        pytest.param("polyint", (x,), {}, marks=mark_todo),
        pytest.param("polymul", (x,), {}, marks=mark_todo),
        pytest.param("polysub", (x,), {}, marks=mark_todo),
        pytest.param("polyval", (x,), {}, marks=mark_todo),
        ("positive", (x,), {}),
        ("pow", (x, 2), {}),
        ("power", (x, 2.5), {}),
        ("prod", (x,), {}),
        ("ptp", (x,), {}),
        ("put", (x, 0, 2), {"inplace": False}),
        ("quantile", (x,), {"q": 0.5}),
        ("rad2deg", (x,), {}),
        ("radians", (x,), {}),
        ("ravel", (x,), {}),
        pytest.param(
            "ravel_multi_index",
            (jnp.array([[0, 1], [0, 1]]), (2, 2)),
            {},
            marks=xfail_quax58,
        ),
        ("real", (x,), {}),
        ("reciprocal", (x,), {}),
        ("remainder", (x, y), {}),
        ("repeat", (x, 3), {}),
        ("reshape", (x, (1, -1)), {}),
        ("resize", (x, (1, len(x))), {}),
        # ("result_type", (3, x), {}),
        ("right_shift", (xbool, xbool), {}),
        ("rint", (x,), {}),
        ("roll", (x, 4), {}),
        ("rollaxis", (x, -1), {}),
        pytest.param("roots", (x[:, 0],), {}, marks=xfail_quax58),
        ("rot90", (x,), {}),
        ("round", (x,), {}),
        # pytest.param("round_", (x,), {}, marks=pytest.mark.deprecated),
        pytest.param("save", (x,), {}, marks=pytest.mark.xfail),
        pytest.param("savez", (x,), {}, marks=pytest.mark.xfail),
        pytest.param("searchsorted", (x,), {}, marks=mark_todo),
        pytest.param("select", (x,), {}, marks=mark_todo),
        *(
            pytest.param("setdiff1d", (x[:, 0], y[:, 0]), {}, marks=xfail_quax58),
            ("setdiff1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}),
        ),
        *(
            pytest.param("setxor1d", (x[:, 0], y[:, 0]), {}, marks=xfail_quax58),
            ("setxor1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}),
        ),
        ("shape", (x,), {}),
        ("sign", (x,), {}),
        ("signbit", (x,), {}),
        ("sin", (x,), {}),
        ("sinc", (x,), {}),
        ("sinh", (x,), {}),
        ("size", (x,), {}),
        ("sort", (x,), {}),
        ("sort_complex", (x,), {}),
        ("split", (x, 2), {}),
        ("sqrt", (x,), {}),
        ("square", (x,), {}),
        ("squeeze", (x[None],), {}),
        ("stack", ([x, x],), {}),
        ("std", (x,), {}),
        ("subtract", (x, y), {}),
        ("sum", (x,), {}),
        ("swapaxes", (x, 0, 1), {}),
        pytest.param("take", (), {}, marks=mark_todo),
        pytest.param("take_along_axis", (), {}, marks=mark_todo),
        ("tan", (x,), {}),
        ("tanh", (x,), {}),
        ("tensordot", (x, y), {}),
        ("tile", (x, 3), {}),
        ("trace", (x,), {}),
        ("transpose", (x,), {}),
        ("tril", (jnp.eye(4),), {}),
        ("tril_indices_from", (jnp.eye(4),), {}),
        pytest.param(
            "trim_zeros",
            (
                jnp.concatenate(
                    (np.array([0.0, 0, 0]), x[:, 0], jnp.array([0.0, 0, 0]))
                ),
            ),
            {},
            marks=xfail_quax58,
        ),
        ("triu", (x,), {}),
        ("triu_indices_from", (x,), {}),
        ("true_divide", (x, y), {}),
        ("trunc", (x,), {}),
        pytest.param("ufunc", (x,), {}, marks=mark_todo),
        *(
            pytest.param("union1d", (x[:, 0], y[:, 0]), {}, marks=xfail_quax58),
            ("union1d", (x[:, 0], y[:, 0]), {"size": x[:, 0].size}),
        ),
        *(
            pytest.param("unique", (x,), {}, marks=xfail_quax58),
            ("unique", (x,), {"size": x.size}),
        ),
        *(
            pytest.param("unique_all", (x,), {}, marks=xfail_quax58),
            ("unique_all", (x,), {"size": x.size}),
        ),
        *(
            pytest.param("unique_counts", (x,), {}, marks=xfail_quax58),
            ("unique_counts", (x,), {"size": x.size}),
        ),
        *(
            pytest.param("unique_inverse", (x,), {}, marks=xfail_quax58),
            ("unique_inverse", (x,), {"size": x.size}),
        ),
        *(
            pytest.param("unique_values", (x,), {}, marks=xfail_quax58),
            ("unique_values", (x,), {"size": x.size}),
        ),
        pytest.param(
            "unpackbits",
            (jnp.array([[[1, 0, 1], [0, 1, 0]]], dtype=jnp.uint8),),
            {},
            marks=mark_todo,
        ),
        ("unravel_index", (x,), {"shape": (1, 4)}),
        ("unwrap", (x,), {}),
        pytest.param("vander", (x[:, 0],), {}, marks=mark_todo),
        ("var", (x,), {}),
        ("vdot", (x, y), {}),
        ("vecdot", (x, y), {}),
        ("vsplit", (x, 2), {}),
        ("vstack", ([x, y],), {}),
        ("where", (jnp.ones_like(x, dtype=bool), x, y), {}),
        ("zeros", (4,), {}),
        ("zeros_like", (x,), {}),
    ],
)
def test_lax_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    # Jax
    exp = getattr(jnp, func_name)(*args, **kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed
    got = getattr(qnp, func_name)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)

    assert jtu.all(jtu.map(jnp.allclose, got, exp))


###############################################################################


def test_c_():
    """Test `quaxed.numpy.c_`."""
    assert jnp.all(qnp.c_[1:3, 4:6] == jnp.c_[1:3, 4:6])


def test_dtype():
    """Test `quaxed.numpy.dtype`."""
    assert jnp.all(qnp.dtype(x) == jnp.dtype(x))


def test_einsum_path():
    """Test `quaxed.numpy.einsum_path`."""
    key1, key2, key3 = jr.split(jr.PRNGKey(0), 3)
    a = jr.uniform(key1, (2, 2))
    b = jr.uniform(key2, (2, 5))
    c = jr.uniform(key3, (5, 2))

    got = qnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")
    exp = jnp.einsum_path("ij,jk,kl->il", a, b, c, optimize="greedy")

    assert jnp.array_equal(got[0], exp[0])


def test_euler_gamma():
    """Test `quaxed.numpy.euler_gamma`."""
    assert qnp.euler_gamma == jnp.euler_gamma


def test_finfo():
    """Test `quaxed.numpy.finfo`."""
    assert jnp.all(qnp.finfo(x) == jnp.finfo(x))


@pytest.mark.xfail(reason="JAX doesn't implement fromfile.")
def test_fromfile():
    """Test `quaxed.numpy.fromfile`."""
    dt = np.dtype([("time", [("min", np.int64), ("sec", np.int64)]), ("temp", float)])
    x = np.zeros((1,), dtype=dt)
    x["time"]["min"] = 10
    x["temp"] = 98.25

    fname = tempfile.mkstemp()[1]
    x.tofile(fname)

    assert jnp.all(qnp.fromfile(x, dtype=dt) == jnp.fromfile(x, dtype=dt))


def test_frompyfunc():
    """Test `quaxed.numpy.frompyfunc`."""
    assert jnp.all(
        qnp.frompyfunc(lambda x: x, 1, 1)(x) == jnp.frompyfunc(lambda x: x, 1, 1)(x)
    )


def test_get_printoptions():
    """Test `quaxed.numpy.get_printoptions`."""
    assert jnp.all(qnp.get_printoptions() == jnp.get_printoptions())


def test_iinfo():
    """Test `quaxed.numpy.iinfo`."""
    got = qnp.iinfo(x.astype(int))
    expect = jnp.iinfo(x.astype(int))
    assert jt.tree_structure(got) == jt.tree_structure(expect)
    assert all(getattr(got, k) == getattr(expect, k) for k in ("min", "max", "dtype"))


def test_load(tmp_path):
    """Test `quaxed.numpy.load`."""
    path = tmp_path / "test.npy"
    jnp.save(path, x)
    assert jnp.array_equal(qnp.load(path), jnp.load(path))


def test_r_():
    """Test `quaxed.numpy.r_`."""
    assert jnp.all(qnp.r_[x, y] == jnp.r_[x, y])


def test_result_type():
    """Test `quaxed.numpy.result_type`."""
    assert qnp.result_type(3, x) == jnp.result_type(3, x)


def test_vectorize():
    """Test `quaxed.numpy.vectorize`."""

    @qnp.vectorize
    def f(x):
        return x + 1

    assert jnp.all(f(x) == jnp.vectorize(lambda x: x + 1)(x))


###############################################################################
# Linalg

x1225 = jnp.array([[1, 2], [2, 5]], dtype=float)
xN3 = jnp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)


@pytest.mark.parametrize(
    ("func_name", "args", "kw"),
    [
        ("cholesky", (x1225,), {}),
        ("cross", (xN3, xN3), {}),
        ("det", (x1225,), {}),
        ("diagonal", (xN3,), {}),
        ("eig", (xN3,), {}),
        ("eigvals", (xN3,), {}),
        ("eigh", (xN3,), {}),
        ("eigvalsh", (xN3,), {}),
        ("inv", (x1225,), {}),
        ("matmul", (xN3, xN3), {}),
        ("matrix_norm", (xN3,), {"ord": 2}),
        ("matrix_power", (xN3, 2), {}),
        ("matrix_rank", (xN3,), {}),
        ("matrix_transpose", (xN3,), {}),
        ("outer", (xN3[:, 0], xN3[:, 1]), {}),
        ("pinv", (xN3,), {}),
        ("qr", (xN3,), {}),
        ("slogdet", (x1225,), {}),
        ("solve", (x1225, jnp.array([1, 2])), {}),
        ("svd", (xN3,), {}),
        ("svdvals", (xN3,), {}),
        ("tensordot", (xN3, xN3), {"axes": 1}),
        ("trace", (xN3,), {}),
        ("vecdot", (xN3[:, 0], xN3[:, 1]), {}),
        ("vector_norm", (xN3,), {"ord": 2}),
        ("norm", (xN3,), {"ord": 2}),
    ],
)
def test_linalg_functions(func_name, args, kw):
    """Test lax vs qlax functions."""
    # Jax
    exp = getattr(jnp.linalg, func_name)(*args, **kw)
    exp = exp if isinstance(exp, tuple | list) else (exp,)

    # Quaxed
    got = getattr(qnp.linalg, func_name)(*args, **kw)
    got = got if isinstance(got, tuple | list) else (got,)

    assert jtu.all(jtu.map(jnp.allclose, got, exp))
