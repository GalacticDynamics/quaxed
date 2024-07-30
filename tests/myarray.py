"""Test with :class:`MyArray` inputs."""

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import equinox as eqx
import jax
import jax.experimental.array_api as jax_xp
from jax import Device, lax
from jax._src.lax.lax import DotDimensionNumbers, PrecisionLike
from jax._src.lax.slicing import GatherDimensionNumbers, GatherScatterMode
from jax._src.typing import DTypeLike, Shape
from jaxtyping import ArrayLike
from quax import ArrayValue, register

from quaxed._types import DType
from quaxed.array_api._dispatch import dispatcher


class MyArray(ArrayValue):
    """A :class:`quax.ArrayValue` that is dense.

    This is different from :class:`quax.MyArray` only in that
    `quax` will not attempt to convert it to a JAX array.
    """

    array: jax.Array = eqx.field(converter=jax_xp.asarray)

    def materialise(self) -> jax.Array:
        """Convert to a JAX array."""
        raise NotImplementedError

    def aval(self) -> jax.core.ShapedArray:
        """Return the ShapedArray."""
        return jax.core.get_aval(self.array)


# ==============================================================================


@register(lax.abs_p)
def _abs_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.abs(x.array))


# ==============================================================================


@register(lax.acos_p)
def _acos_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.acos(x.array))


# ==============================================================================


@register(lax.acosh_p)
def _acosh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.acosh(x.array))


# ==============================================================================


@register(lax.add_p)
def _add_p_qq(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.add(x.array, y))


@register(lax.add_p)
def _add_p_qq(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.add(x.array, y.array))


# ==============================================================================


@register(lax.after_all_p)
def _after_all_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_gather_p)
def _all_gather_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.all_to_all_p)
def _all_to_all_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.and_p)
def _and_p(x1: MyArray, x2: MyArray, /) -> MyArray:
    return MyArray(x1.array & x2.array)


# ==============================================================================


@register(lax.approx_top_k_p)
def _approx_top_k_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.argmax_p)
def _argmax_p(operand: MyArray, *, axes: Any, index_dtype: Any) -> MyArray:
    return replace(operand, array=lax.argmax(operand.array, axes[0], index_dtype))


# ==============================================================================


@register(lax.argmin_p)
def _argmin_p(operand: MyArray, *, axes: Any, index_dtype: Any) -> MyArray:
    return replace(operand, array=lax.argmin(operand.array, axes[0], index_dtype))


# ==============================================================================


@register(lax.asin_p)
def _asin_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.asin(x.array))


# ==============================================================================


@register(lax.asinh_p)
def _asinh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.asinh(x.array))


# ==============================================================================


@register(lax.atan2_p)
def _atan2_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.atan2(x.array, y.array))


# ==============================================================================


@register(lax.atan_p)
def _atan_p(x: MyArray) -> MyArray:
    return MyArray(lax.atan(x.array))


# ==============================================================================


@register(lax.atanh_p)
def _atanh_p(x: MyArray) -> MyArray:
    return MyArray(lax.atanh(x.array))


# ==============================================================================


@register(lax.axis_index_p)
def _axis_index_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i0e_p)
def _bessel_i0e_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.bessel_i1e_p)
def _bessel_i1e_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.bitcast_convert_type_p)
def _bitcast_convert_type_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.broadcast_in_dim_p)
def _broadcast_in_dim_p(
    operand: MyArray,
    *,
    shape: Any,
    broadcast_dimensions: Any,
) -> MyArray:
    return replace(
        operand,
        array=lax.broadcast_in_dim(operand.array, shape, broadcast_dimensions),
    )


# ==============================================================================


@register(lax.cbrt_p)
def _cbrt_p(x: MyArray) -> MyArray:
    return MyArray(lax.cbrt(x.array))


# ==============================================================================


@register(lax.ceil_p)
def _ceil_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.ceil(x.array))


# ==============================================================================


@register(lax.clamp_p)
def _clamp_p(min: MyArray, x: MyArray, max: MyArray) -> MyArray:
    return replace(x, array=lax.clamp(min.array, x.array, max.array))


# ==============================================================================


@register(lax.clz_p)
def _clz_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.complex_p)
def _complex_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.complex(x.array, y.array))


# ==============================================================================


@register(lax.concatenate_p)
def _concatenate_p(
    operand0: MyArray,
    *operands: MyArray,
    **kwargs: Any,
) -> MyArray:
    return MyArray(
        lax.concatenate([operand0.array] + [op.array for op in operands], **kwargs)
    )


@register(lax.concatenate_p)
def _concatenate_p(operand0: ArrayLike, operand1: MyArray, /, **kwargs: Any) -> MyArray:
    return MyArray(lax.concatenate_p.bind(operand0, operand1.array, **kwargs))


# ==============================================================================


@register(lax.cond_p)  # TODO: implement
def _cond_p(index, consts) -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.conj_p)
def _conj_p(x: MyArray, *, input_dtype: Any) -> MyArray:
    del input_dtype  # TODO: use this?
    return replace(x, array=lax.conj(x.array))


# ==============================================================================


@register(lax.conv_general_dilated_p)
def _conv_general_dilated_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.convert_element_type_p)
def _convert_element_type_p(operand: MyArray, **kwargs: Any) -> MyArray:
    return replace(
        operand,
        array=lax.convert_element_type_p.bind(operand.array, **kwargs),
    )


# ==============================================================================


@register(lax.copy_p)
def _copy_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.copy_p.bind(x.array))


# ==============================================================================


@register(lax.cos_p)
def _cos_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.cos(x.array))


# ==============================================================================


@register(lax.cosh_p)
def _cosh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.cosh(x.array))


# ==============================================================================


@register(lax.create_token_p)
def _create_token_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.cumlogsumexp_p)
def _cumlogsumexp_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    # TODO: double check units make sense here.
    return replace(
        operand,
        array=lax.cumlogsumexp(operand.array, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cummax_p)
def _cummax_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cummax(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cummin_p)
def _cummin_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cummin(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.cumprod_p)
def _cumprod_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(
        operand,
        array=lax.cumprod(operand.array, axis=axis, reverse=reverse),
    )


# ==============================================================================


@register(lax.cumsum_p)
def _cumsum_p(operand: MyArray, *, axis: Any, reverse: Any) -> MyArray:
    return replace(operand, array=lax.cumsum(operand.array, axis=axis, reverse=reverse))


# ==============================================================================


@register(lax.device_put_p)
def _device_put_p(x: MyArray, **kwargs: Any) -> MyArray:
    return replace(x, array=jax.device_put(x.array, **kwargs))


# ==============================================================================


@register(lax.digamma_p)
def _digamma_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.digamma(x.array))


# ==============================================================================


@register(lax.div_p)
def _div_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.div(x.array, y.array))


@register(lax.div_p)
def _div_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.div(x.array, y))


# ==============================================================================


@register(lax.dot_general_p)  # TODO: implement
def _dot_general_p(
    lhs: MyArray,
    rhs: MyArray,
    *,
    dimension_numbers: DotDimensionNumbers,
    precision: PrecisionLike = None,
    preferred_element_type: DTypeLike | None = None,
) -> MyArray:
    return MyArray(
        lax.dot_general_p.bind(
            lhs.array,
            rhs.array,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        ),
    )


# ==============================================================================


@register(lax.dynamic_slice_p)
def _dynamic_slice_p(
    operand: MyArray,
    start_indices: ArrayLike,
    dynamic_sizes: ArrayLike,
    *,
    slice_sizes: Any,
) -> MyArray:
    raise NotImplementedError  # TODO: implement


# ==============================================================================


@register(lax.dynamic_update_slice_p)
def _dynamic_update_slice_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.eq_p)
def _eq_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.eq(x.array, y.array))


@register(lax.eq_p)
def _eq_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.eq(x.array, y))


# ==============================================================================


@register(lax.eq_to_p)
def _eq_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.erf_inv_p)
def _erf_inv_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erf_inv(x.array))


# ==============================================================================


@register(lax.erf_p)
def _erf_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erf(x.array))


# ==============================================================================


@register(lax.erfc_p)
def _erfc_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.erfc(x.array))


# ==============================================================================


@register(lax.exp2_p)
def _exp2_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.exp2(x.array))


# ==============================================================================


@register(lax.exp_p)
def _exp_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.exp(x.array))


# ==============================================================================


@register(lax.expm1_p)
def _expm1_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.expm1(x.array))


# ==============================================================================


@register(lax.fft_p)
def _fft_p(x: MyArray, *, fft_type: Any, fft_lengths: Any) -> MyArray:
    return replace(x, array=lax.fft(x.array, fft_type, fft_lengths))


# ==============================================================================


@register(lax.floor_p)
def _floor_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.floor(x.array))


# ==============================================================================


@register(lax.gather_p)
def _gather_p(
    operand: MyArray,
    start_indices: MyArray,
    *,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str | GatherScatterMode | None = None,
    fill_value: Any = None,
) -> MyArray:
    return MyArray(
        lax.gather(
            operand.array,
            start_indices.array,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            mode=mode,
            fill_value=fill_value,
        ),
    )


@register(lax.gather_p)
def _gather_p(
    operand: MyArray,
    start_indices: ArrayLike,
    *,
    dimension_numbers: GatherDimensionNumbers,
    slice_sizes: Shape,
    unique_indices: bool,
    indices_are_sorted: bool,
    mode: str | GatherScatterMode | None = None,
    fill_value: Any = None,
) -> MyArray:
    return MyArray(
        lax.gather(
            operand.array,
            start_indices,
            dimension_numbers=dimension_numbers,
            slice_sizes=slice_sizes,
            unique_indices=unique_indices,
            indices_are_sorted=indices_are_sorted,
            mode=mode,
            fill_value=fill_value,
        ),
    )


# ==============================================================================


@register(lax.ge_p)
def _ge_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.ge(x.array, y.array))


# ==============================================================================


@register(lax.gt_p)
def _gt_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.gt(x.array, y.array))


# ==============================================================================


@register(lax.igamma_grad_a_p)
def _igamma_grad_a_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.igamma_p)
def _igamma_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.igammac_p)
def _igammac_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.imag_p)
def _imag_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.imag(x.array))


# ==============================================================================


@register(lax.infeed_p)
def _infeed_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.integer_pow_p)
def _integer_pow_p(x: MyArray, *, y: Any) -> MyArray:
    return replace(x, array=lax.integer_pow(x.array, y))


# ==============================================================================


# @register(lax.iota_p)
# def _iota_p(dtype: MyArray) -> MyArray:
#     raise NotImplementedError


# ==============================================================================


@register(lax.is_finite_p)
def _is_finite_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.is_finite(x.array))


# ==============================================================================


@register(lax.le_p)
def _le_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.le(x.array, y.array))


# ==============================================================================


@register(lax.le_to_p)
def _le_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.lgamma_p)
def _lgamma_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.lgamma(x.array))


# ==============================================================================


@register(lax.linear_solve_p)
def _linear_solve_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.log1p_p)
def _log1p_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.log1p(x.array))


# ==============================================================================


@register(lax.log_p)
def _log_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.log(x.array))


# ==============================================================================


@register(lax.logistic_p)
def _logistic_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.logistic(x.array))


# ==============================================================================


@register(lax.lt_p)
def _lt_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.lt(x.array, y.array))


@register(lax.lt_p)
def _lt_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.lt(x.array, y))


# ==============================================================================


@register(lax.lt_to_p)
def _lt_to_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.max_p)
def _max_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.max(x.array, y.array))


@register(lax.max_p)
def _max_p_d1(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.max(x, y.array))


@register(lax.max_p)
def _max_p_d2(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.max(x.array, y))


# ==============================================================================


@register(lax.min_p)
def _min_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.min(x.array, y.array))


# ==============================================================================
# Multiplication


@register(lax.mul_p)
def _mul_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.mul(x.array, y.array))


# ==============================================================================


@register(lax.ne_p)
def _ne_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.ne(x.array, y))


@register(lax.ne_p)
def _ne_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.ne(x.array, y.array))


# ==============================================================================


@register(lax.neg_p)
def _neg_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.neg(x.array))


# ==============================================================================


@register(lax.nextafter_p)
def _nextafter_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.not_p)
def _not_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.bitwise_not(x.array))


# ==============================================================================


@register(lax.or_p)
def _or_p(x: MyArray, y: MyArray) -> MyArray:
    return replace(x, array=lax.bitwise_or(x.array, y.array))


# ==============================================================================


@register(lax.outfeed_p)
def _outfeed_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.pad_p)
def _pad_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmax_p)
def _pmax_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.pmin_p)
def _pmin_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.polygamma_p)
def _polygamma_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.population_count_p)
def _population_count_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.pow_p)
def _pow_p_qq(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(array=lax.pow(x.array, y.array))


# ==============================================================================


@register(lax.ppermute_p)
def _ppermute_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.psum_p)
def _psum_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.random_gamma_grad_p)
def _random_gamma_grad_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.real_p)
def _real_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.real(x.array))


# ==============================================================================


@register(lax.reduce_and_p)
def _reduce_and_p(
    operand: MyArray,
    *,
    axes: Sequence[int],
) -> Any:
    return lax.reduce_and_p.bind(operand.array, axes=tuple(axes))


# ==============================================================================


@register(lax.reduce_max_p)
def _reduce_max_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_min_p)
def _reduce_min_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_or_p)
def _reduce_or_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_p)
def _reduce_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_precision_p)
def _reduce_precision_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_prod_p)
def _reduce_prod_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_sum_p)
def _reduce_sum_p(operand: MyArray, *, axes: tuple[int, ...]) -> MyArray:
    return MyArray(lax.reduce_sum_p.bind(operand.array, axes=axes))


# ==============================================================================


@register(lax.reduce_window_max_p)
def _reduce_window_max_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_min_p)
def _reduce_window_min_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_p)
def _reduce_window_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_window_sum_p)
def _reduce_window_sum_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.reduce_xor_p)
def _reduce_xor_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.regularized_incomplete_beta_p)
def _regularized_incomplete_beta_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.rem_p)
def _rem_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.rem(x.array, y.array))


@register(lax.rem_p)
def _rem_p_d1(x: ArrayLike, y: MyArray) -> MyArray:
    return MyArray(lax.rem(x, y.array))


@register(lax.rem_p)
def _rem_p_d1(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.rem(x.array, y))


# ==============================================================================


@register(lax.reshape_p)
def _reshape_p(operand: MyArray, *, new_sizes: Any, dimensions: Any) -> MyArray:
    return replace(operand, array=lax.reshape(operand.array, new_sizes, dimensions))


# ==============================================================================


@register(lax.rev_p)
def _rev_p(operand: MyArray, *, dimensions: Any) -> MyArray:
    return replace(operand, array=lax.rev(operand.array, dimensions))


# ==============================================================================


@register(lax.rng_bit_generator_p)
def _rng_bit_generator_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.rng_uniform_p)
def _rng_uniform_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.round_p)
def _round_p(x: MyArray, *, rounding_method: Any) -> MyArray:
    return replace(x, array=lax.round(x.array, rounding_method))


# ==============================================================================


@register(lax.rsqrt_p)
def _rsqrt_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.rsqrt(x.array) ** (-1 / 2))


# ==============================================================================


@register(lax.scan_p)
def _scan_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_add_p)
def _scatter_add_p(
    operand: MyArray,
    scatter_indices: MyArray,
    updates: MyArray,
    *,
    update_jaxpr: Any,
    update_consts: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str | GatherScatterMode | None = None,
) -> MyArray:
    return MyArray(
        lax.scatter_add_p.bind(
            operand.array,
            scatter_indices.array,
            updates.array,
            update_jaxpr=update_jaxpr,
            update_consts=update_consts,
            dimension_numbers=dimension_numbers,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        ),
    )


@register(lax.scatter_add_p)
def _scatter_add_p(
    operand: MyArray,
    scatter_indices: ArrayLike,
    updates: ArrayLike,
    *,
    update_jaxpr: Any,
    update_consts: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str | GatherScatterMode | None = None,
) -> MyArray:
    return MyArray(
        lax.scatter_add_p.bind(
            operand.array,
            scatter_indices,
            updates,
            update_jaxpr=update_jaxpr,
            update_consts=update_consts,
            dimension_numbers=dimension_numbers,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        ),
    )


@register(lax.scatter_add_p)
def _scatter_add_p(
    operand: ArrayLike,
    scatter_indices: MyArray,
    updates: ArrayLike,
    *,
    update_jaxpr: Any,
    update_consts: Any,
    dimension_numbers: Any,
    indices_are_sorted: bool,
    unique_indices: bool,
    mode: str | GatherScatterMode | None = None,
) -> MyArray:
    return MyArray(
        lax.scatter_add_p.bind(
            operand,
            scatter_indices.array,
            updates,
            update_jaxpr=update_jaxpr,
            update_consts=update_consts,
            dimension_numbers=dimension_numbers,
            indices_are_sorted=indices_are_sorted,
            unique_indices=unique_indices,
            mode=mode,
        ),
    )


# @register(lax.scatter_add_p)
# def _scatter_add_p(
#     operand: Zero,
#     scatter_indices: MyArray,
#     updates: MyArray,
#     *,
#     update_jaxpr: Any,
#     update_consts: Any,
#     dimension_numbers: Any,
#     indices_are_sorted: bool,
#     unique_indices: bool,
#     mode: str | GatherScatterMode | None = None,
# ) -> MyArray:
#     return MyArray(
#         lax.scatter_add_p.bind(
#             jax_xp.zeros_like(operand),
#             scatter_indices.array,
#             updates.array,
#             update_jaxpr=update_jaxpr,
#             update_consts=update_consts,
#             dimension_numbers=dimension_numbers,
#             indices_are_sorted=indices_are_sorted,
#             unique_indices=unique_indices,
#             mode=mode,
#         ),
#     )


# ==============================================================================


@register(lax.scatter_max_p)
def _scatter_max_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_min_p)
def _scatter_min_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_mul_p)
def _scatter_mul_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.scatter_p)
def _scatter_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_gather_add_p)
def _select_and_gather_add_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_add_p)
def _select_and_scatter_add_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_and_scatter_p)
def _select_and_scatter_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.select_n_p)
def _select_n_p(which: MyArray, *cases: MyArray) -> MyArray:
    # Process the cases, replacing Zero and MyArray with a materialised array.
    return MyArray(lax.select_n(which.array, *(case.array for case in cases)))


@register(lax.select_n_p)
def _select_n_p(which: ArrayLike, case0: ArrayLike, case1: MyArray) -> MyArray:
    return MyArray(lax.select_n(which, case0, case1.array))


@register(lax.select_n_p)
def _select_n_p(which: ArrayLike, case0: MyArray, case1: ArrayLike) -> MyArray:
    return MyArray(lax.select_n(which, case0.array, case1))


# ==============================================================================


@register(lax.sharding_constraint_p)
def _sharding_constraint_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.shift_left_p)
def _shift_left_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.shift_left(x.array, y.array))


# ==============================================================================


@register(lax.shift_right_arithmetic_p)
def _shift_right_arithmetic_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.shift_right_arithmetic(x.array, y.array))


# ==============================================================================


@register(lax.shift_right_logical_p)
def _shift_right_logical_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.sign_p)
def _sign_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sign(x.array))


# ==============================================================================


@register(lax.sin_p)
def _sin_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sin(x.array))


# ==============================================================================


@register(lax.sinh_p)
def _sinh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sinh(x.array))


# ==============================================================================


@register(lax.slice_p)
def _slice_p(
    operand: MyArray,
    *,
    start_indices: Any,
    limit_indices: Any,
    strides: Any,
) -> MyArray:
    return replace(
        operand,
        array=lax.slice_p.bind(
            operand.array,
            start_indices=start_indices,
            limit_indices=limit_indices,
            strides=strides,
        ),
    )


# ==============================================================================


@register(lax.sort_p)
def _sort_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.sqrt_p)
def _sqrt_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.sqrt(x.array))


# ==============================================================================


@register(lax.squeeze_p)
def _squeeze_p(x: MyArray, *, dimensions: Any) -> MyArray:
    return replace(x, array=lax.squeeze(x.array, dimensions))


# ==============================================================================


@register(lax.stop_gradient_p)
def _stop_gradient_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.stop_gradient(x.array))


# ==============================================================================
# Subtraction


@register(lax.sub_p)
def _sub_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.sub(x.array, y.array))


@register(lax.sub_p)
def _sub_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.sub(x.array, y))


# ==============================================================================


@register(lax.tan_p)
def _tan_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.tan(x.array))


# ==============================================================================


@register(lax.tanh_p)
def _tanh_p(x: MyArray) -> MyArray:
    return replace(x, array=lax.tanh(x.array))


# ==============================================================================


@register(lax.top_k_p)
def _top_k_p(operand: MyArray, k: int) -> MyArray:
    raise replace(operand, array=lax.top_k(operand.array, k))


# ==============================================================================


@register(lax.transpose_p)
def _transpose_p(operand: MyArray, *, permutation: Any) -> MyArray:
    return replace(operand, array=lax.transpose(operand.array, permutation))


# ==============================================================================


@register(lax.while_p)
def _while_p() -> MyArray:
    raise NotImplementedError


# ==============================================================================


@register(lax.xor_p)
def _xor_p(x: MyArray, y: MyArray) -> MyArray:
    return MyArray(lax.bitwise_xor(x.array, y.array))


@register(lax.xor_p)
def _xor_p(x: MyArray, y: ArrayLike) -> MyArray:
    return MyArray(lax.bitwise_xor(x.array, y))


# ==============================================================================


@register(lax.zeta_p)
def _zeta_p() -> MyArray:
    raise NotImplementedError


###############################################################################


@dispatcher
def arange(
    start: MyArray,
    stop: MyArray | None = None,
    step: MyArray | None = None,
    *,
    dtype: Any = None,
    device: Any = None,
) -> MyArray:
    return MyArray(
        jax_xp.arange(
            start.array,
            stop=stop.array if stop is not None else None,
            step=step.array if step is not None else None,
            dtype=dtype,
            device=device,
        ),
    )


@dispatcher  # type: ignore[misc]
def empty_like(
    x: MyArray,
    /,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> MyArray:
    return MyArray(jax_xp.empty_like(x.array, dtype=dtype, device=device))


@dispatcher
def full_like(
    x: MyArray,
    /,
    fill_value: bool | int | float | complex | MyArray,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
) -> MyArray:
    return MyArray(
        jax_xp.full_like(x.array, fill_value, dtype=dtype, device=device),
    )


@dispatcher
def linspace(
    start: MyArray,
    stop: MyArray,
    num: int,
    *,
    dtype: DType | None = None,
    device: Device | None = None,
    endpoint: bool = True,
) -> MyArray:
    return MyArray(
        jax_xp.linspace(
            start.array,
            stop.array,
            num=num,
            dtype=dtype,
            device=device,
            endpoint=endpoint,
        ),
    )


@dispatcher
def ones_like(
    x: MyArray,
    /,
    dtype: DType | None = None,
    device: Device | None = None,
) -> MyArray:
    return MyArray(jax_xp.ones_like(x.array, dtype=dtype, device=device))


@dispatcher
def zeros_like(
    x: MyArray,
    /,
    dtype: DType | None = None,
    device: Device | None = None,
) -> MyArray:
    return MyArray(jax_xp.zeros_like(x.array, dtype=dtype, device=device))
