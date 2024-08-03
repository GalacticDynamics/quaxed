# Quaxed

pre-`quaxified` JAX functions.

`quax` enables JAX + multiple dispatch + custom array-ish objects. `quaxed`
means you don't have to wrap every function in `quax.quaxify` wrappers every
time.

## Installation

```bash
pip install quaxed
```

## Getting started

Import whatever library you need as a drop-in replacement for its JAX
counterpart.

To see the API check out the [`quaxed`](./api/quaxed.md) in the left bar.

```python
import quaxed.numpy as jnp

x = jnp.linspace(0.0, 1.0, num=3)

print(jnp.cos(x))
# [1.         0.87758255 0.5403023 ]
```

The advantage of `quaxed` over plain `JAX` is that every function is `quaxify`'d
and will work with properly formulated array-ish objects.

For this example we use [unxt](https://github.com/GalacticDyanamics/quax)'s
`Quantity` for unitful calculations.

```python
from unxt import Quantity

x = Quantity(jnp.linspace(0.0, 1.0, num=3), "deg")

print(jnp.cos(x))
# Quantity['dimensionless'](Array([1.       , 0.9999619, 0.9998477], dtype=float32), unit='')
```

## See also: other libraries in the Quax ecosystem

[Quax](https://github.com/patrick-kidger/quax): the base library.

[unxt](https://github.com/GalacticDyanamics/quax): Units and Quantities in Jax.

[coordinax](https://github.com/GalacticDyanamics/coordinax): Vector
representations (built on `unxt`).

[galax](https://github.com/GalacticDyanamics/galax): Galactic dynamics in Jax
(built on `unxt` and `coordinax`).
