"""Building blocks for Array-ish objects.

!!! warning "Experimental"
    This module is experimental and subject to change. Use with caution.

"""

# fmt: off
__all__ = [
    ################################################
    "LaxComparisonMixin", "NumpyComparisonMixin",  # rich comparison
    # ----------
    "LaxEqMixin", "NumpyEqMixin",  # `__eq__`
    "LaxNeMixin", "NumpyNeMixin",  # `__ne__`
    "LaxLtMixin", "NumpyLtMixin",  # `__lt__`
    "LaxLeMixin", "NumpyLeMixin",  # `__le__`
    "LaxGtMixin", "NumpyGtMixin",  # `__gt__`
    "LaxGeMixin", "NumpyGeMixin",  # `__ge__`
    ##################################################
    "LaxBinaryOpsMixin", "NumpyBinaryOpsMixin",
    # ========== Float Operations ==========
    "LaxMathMixin", "NumpyMathMixin",
    # ----- add -----
    "LaxBothAddMixin", "NumpyBothAddMixin",
    "LaxAddMixin", "NumpyAddMixin",  # __add__
    "LaxRAddMixin", "NumpyRAddMixin",  # __radd__
    # ----- sub -----
    "LaxBothSubMixin", "NumpyBothSubMixin",
    "LaxSubMixin", "NumpySubMixin",  # __sub__
    "LaxRSubMixin", "NumpyRSubMixin",  # __rsub__
    # ----- mul -----
    "LaxBothMulMixin", "NumpyBothMulMixin",
    "LaxMulMixin", "NumpyMulMixin",  # __mul__
    "LaxRMulMixin", "NumpyRMulMixin",  # __rmul__
    # ---- matmul -----
    "LaxMatMulMixin", "NumpyMatMulMixin",  # __matmul__
    # "LaxRMatMulMixin", "NumpyRMatMulMixin",  # __rmatmul__
    # ----- truediv -----
    "LaxBothTrueDivMixin", "NumpyBothTrueDivMixin",
    "LaxTrueDivMixin", "NumpyTrueDivMixin",  # __truediv__
    "LaxRTrueDivMixin", "NumpyRTrueDivMixin",  # __rtruediv__
    # ----- floordiv -----
    "LaxBothFloorDivMixin", "NumpyBothFloorDivMixin",
    "LaxFloorDivMixin", "NumpyFloorDivMixin",  # __floordiv__
    "LaxRFloorDivMixin", "NumpyRFloorDivMixin",  # __rfloordiv__
    # ----- mod -----
    "LaxBothModMixin", "NumpyBothModMixin",
    "LaxModMixin", "NumpyModMixin",  # __mod__
    "LaxRModMixin", "NumpyRModMixin",  # __rmod__
    # ----- divmod -----
    "NumpyBothDivModMixin",
    # "LaxDivModMixin",
    "NumpyDivModMixin",
    # "LaxRDivModMixin",
    "NumpyRDivModMixin",
    # ----- pow -----
    "LaxBothPowMixin", "NumpyBothPowMixin",
    "LaxPowMixin", "NumpyPowMixin",  # __pow__
    "LaxRPowMixin", "NumpyRPowMixin",  # __rpow__
    # ========== Bitwise Operations ==========
    "LaxBitwiseMixin", "NumpyBitwiseMixin",
    # ----- lshift -----
    "LaxBothLShiftMixin", "NumpyBothLShiftMixin",
    "LaxLShiftMixin", "NumpyLShiftMixin",  # __lshift__
    "LaxRLShiftMixin", "NumpyRLShiftMixin",  # __rlshift__
    # ----- rshift -----
    "LaxBothRShiftMixin", "NumpyBothRShiftMixin",
    "LaxRShiftMixin", "NumpyRShiftMixin",  # __rshift__
    "LaxRRShiftMixin", "NumpyRRShiftMixin",  # __rrshift__
    # ----- and -----
    "LaxBothAndMixin", "NumpyBothAndMixin",
    "LaxAndMixin", "NumpyAndMixin",  # __and__
    "LaxRAndMixin", "NumpyRAndMixin",  # __rand__
    # ----- xor -----
    "LaxBothXorMixin", "NumpyBothXorMixin",
    "LaxXorMixin", "NumpyXorMixin",  # __xor__
    "LaxRXorMixin", "NumpyRXorMixin",  # __rxor__
    # ----- or -----
    "LaxBothOrMixin", "NumpyBothOrMixin",
    "LaxOrMixin", "NumpyOrMixin",  # __or__
    "LaxROrMixin", "NumpyROrMixin",  # __ror__
    ##################################################
    "LaxUnaryMixin", "NumpyUnaryMixin",
    # ----------
    "LaxPosMixin", "NumpyPosMixin",  # __pos__
    "LaxNegMixin", "NumpyNegMixin",  # __neg__
    "NumpyInvertMixin",              # __invert__
    "LaxAbsMixin", "NumpyAbsMixin",  # __abs__
    #################################################
    "LaxRoundMixin", "NumpyRoundMixin",  # __round__
    "LaxTruncMixin", "NumpyTruncMixin",  # __trunc__
    "LaxFloorMixin", "NumpyFloorMixin",  # __floor__
    "LaxCeilMixin", "NumpyCeilMixin",  # __ceil__
    ##################################################
    "LaxLenMixin", "NumpyLenMixin",  # __len__
    "LaxLengthHintMixin", "NumpyLengthHintMixin",  # __length_hint__
    #################################################
    "NumpyCopyMixin",  # __copy__
    "NumpyDeepCopyMixin",  # __deepcopy__
]
# fmt: on

from ._arrayish.binary import (
    LaxAddMixin,
    LaxAndMixin,
    LaxBinaryOpsMixin,
    LaxBitwiseMixin,
    LaxBothAddMixin,
    LaxBothAndMixin,
    LaxBothFloorDivMixin,
    LaxBothLShiftMixin,
    LaxBothModMixin,
    LaxBothMulMixin,
    LaxBothOrMixin,
    LaxBothPowMixin,
    LaxBothRShiftMixin,
    LaxBothSubMixin,
    LaxBothTrueDivMixin,
    LaxBothXorMixin,
    LaxFloorDivMixin,
    LaxLShiftMixin,
    LaxMathMixin,
    LaxMatMulMixin,
    LaxModMixin,
    LaxMulMixin,
    LaxOrMixin,
    LaxPowMixin,
    LaxRAddMixin,
    LaxRAndMixin,
    LaxRFloorDivMixin,
    LaxRLShiftMixin,
    LaxRModMixin,
    LaxRMulMixin,
    LaxROrMixin,
    LaxRPowMixin,
    LaxRRShiftMixin,
    LaxRShiftMixin,
    LaxRSubMixin,
    LaxRTrueDivMixin,
    LaxRXorMixin,
    LaxSubMixin,
    LaxTrueDivMixin,
    LaxXorMixin,
    NumpyAddMixin,
    NumpyAndMixin,
    NumpyBinaryOpsMixin,
    NumpyBitwiseMixin,
    NumpyBothAddMixin,
    NumpyBothAndMixin,
    NumpyBothDivModMixin,
    NumpyBothFloorDivMixin,
    NumpyBothLShiftMixin,
    NumpyBothModMixin,
    NumpyBothMulMixin,
    NumpyBothOrMixin,
    NumpyBothPowMixin,
    NumpyBothRShiftMixin,
    NumpyBothSubMixin,
    NumpyBothTrueDivMixin,
    NumpyBothXorMixin,
    NumpyDivModMixin,
    NumpyFloorDivMixin,
    NumpyLShiftMixin,
    NumpyMathMixin,
    NumpyMatMulMixin,
    NumpyModMixin,
    NumpyMulMixin,
    NumpyOrMixin,
    NumpyPowMixin,
    NumpyRAddMixin,
    NumpyRAndMixin,
    NumpyRDivModMixin,
    NumpyRFloorDivMixin,
    NumpyRLShiftMixin,
    NumpyRModMixin,
    NumpyRMulMixin,
    NumpyROrMixin,
    NumpyRPowMixin,
    NumpyRRShiftMixin,
    NumpyRShiftMixin,
    NumpyRSubMixin,
    NumpyRTrueDivMixin,
    NumpyRXorMixin,
    NumpySubMixin,
    NumpyTrueDivMixin,
    NumpyXorMixin,
)
from ._arrayish.container import (
    LaxLengthHintMixin,
    LaxLenMixin,
    NumpyLengthHintMixin,
    NumpyLenMixin,
)
from ._arrayish.copy import (
    NumpyCopyMixin,
    NumpyDeepCopyMixin,
)
from ._arrayish.rich import (
    LaxComparisonMixin,
    LaxEqMixin,
    LaxGeMixin,
    LaxGtMixin,
    LaxLeMixin,
    LaxLtMixin,
    LaxNeMixin,
    NumpyComparisonMixin,
    NumpyEqMixin,
    NumpyGeMixin,
    NumpyGtMixin,
    NumpyLeMixin,
    NumpyLtMixin,
    NumpyNeMixin,
)
from ._arrayish.round import (
    LaxCeilMixin,
    LaxFloorMixin,
    LaxRoundMixin,
    LaxTruncMixin,
    NumpyCeilMixin,
    NumpyFloorMixin,
    NumpyRoundMixin,
    NumpyTruncMixin,
)
from ._arrayish.unary import (
    LaxAbsMixin,
    LaxNegMixin,
    LaxPosMixin,
    LaxUnaryMixin,
    NumpyAbsMixin,
    NumpyInvertMixin,
    NumpyNegMixin,
    NumpyPosMixin,
    NumpyUnaryMixin,
)
