"""Copyright (c) 2023 Nathaniel Starkman. All rights reserved.

quaxed: Array-API JAX compatibility
"""

# pylint: disable=redefined-builtin

from jax.experimental.array_api import __array_api_version__
from jaxtyping import install_import_hook

with install_import_hook("quaxed", None):
    from . import (
        _constants,
        _creation_functions,
        _data_type_functions,
        _dispatch,
        _elementwise_functions,
        _indexing_functions,
        _linear_algebra_functions,
        _manipulation_functions,
        _searching_functions,
        _set_functions,
        _sorting_functions,
        _statistical_functions,
        _utility_functions,
        fft,
        linalg,
    )
    from ._constants import *
    from ._creation_functions import *
    from ._data_type_functions import *
    from ._dispatch import *
    from ._elementwise_functions import *
    from ._indexing_functions import *
    from ._linear_algebra_functions import *
    from ._manipulation_functions import *
    from ._searching_functions import *
    from ._set_functions import *
    from ._sorting_functions import *
    from ._statistical_functions import *
    from ._utility_functions import *

__all__ = ["__array_api_version__", "fft", "linalg"]
__all__ += _constants.__all__
__all__ += _creation_functions.__all__
__all__ += _data_type_functions.__all__
__all__ += _elementwise_functions.__all__
__all__ += _indexing_functions.__all__
__all__ += _linear_algebra_functions.__all__
__all__ += _manipulation_functions.__all__
__all__ += _searching_functions.__all__
__all__ += _set_functions.__all__
__all__ += _sorting_functions.__all__
__all__ += _statistical_functions.__all__
__all__ += _utility_functions.__all__
__all__ += _dispatch.__all__
