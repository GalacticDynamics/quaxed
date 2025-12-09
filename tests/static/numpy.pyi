# Static type tests for quaxed.numpy with MyArray.
#
# This module contains static type tests that verify MyArray works correctly
# with quaxed.numpy functions. Type checkers (mypy, pyright) should verify
# these without runtime execution.

import quaxed.numpy as qnp
from tests.myarray import MyArray

# Define test arrays
arr000 = MyArray([1.0, 2.0, 3.0])
arr001 = MyArray([4.0, 5.0, 6.0])
arr002 = MyArray([[1.0, 2.0], [3.0, 4.0]])
scalar000: float = 2.0
lst000: list[float] = [1.0, 2.0, 3.0]

# Test that unary ufuncs preserve MyArray type
result000: MyArray = qnp.sin(arr000)
result001: MyArray = qnp.cos(arr000)
result002: MyArray = qnp.exp(arr000)
result003: MyArray = qnp.log(arr000)
result004: MyArray = qnp.abs(arr000)
result005: MyArray = qnp.sqrt(arr000)

# Test that binary ufuncs preserve MyArray type
result006: MyArray = qnp.add(arr000, arr001)
result007: MyArray = qnp.subtract(arr000, arr001)
result008: MyArray = qnp.multiply(arr000, arr001)
result009: MyArray = qnp.divide(arr000, arr001)
result010: MyArray = qnp.maximum(arr000, arr001)
result011: MyArray = qnp.minimum(arr000, arr001)

# Test that binary ufuncs with mixed types preserve MyArray
result012: MyArray = qnp.add(arr000, scalar000)
result013: MyArray = qnp.add(scalar000, arr000)
result014: MyArray = qnp.multiply(arr000, scalar000)
result015: MyArray = qnp.multiply(scalar000, arr000)

# MyArray with list should return MyArray
result016: MyArray = qnp.add(arr000, lst000)
result017: MyArray = qnp.add(lst000, arr000)

# Test reduction operations with MyArray
result018 = qnp.sum(arr002)
result019 = qnp.mean(arr002)
result020 = qnp.max(arr002)
result021 = qnp.min(arr002)

# Test array creation functions with MyArray
result022 = qnp.stack([arr000, arr000])
result023 = qnp.concatenate([arr000, arr000])
result024 = qnp.reshape(arr000, (3, 1))
