<h1 align='center'> quaxed </h1>
<h3 align="center">Pre-<code>Quaxify</code>'ed <code>JAX</code></h3>

`Quaxed` wraps [jax](https://jax.readthedocs.io/en/latest/) libraries (using
[`quax`](https://docs.kidger.site/quax/)) to enable using those libraries with
custom array-ish objects, not only jax arrays.

## Installation

[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

<!-- [![Conda-Forge][conda-badge]][conda-link] -->

```bash
pip install quaxed
```

## Documentation

[![Documentation Status][rtd-badge]][rtd-link]

## Quick Start

To understand how `quax` works it's magic, see
[`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) and the
[tutorials](https://docs.kidger.site/quax/examples/custom_rules/).

To use this library, it's as simple as:

```pycon
# Import pre-quaxified library
>>> import quaxed.numpy as jnp  # this is quaxify(jax.numpy)

# As an example, let's import an array-ish object
>>> from unxt import Quantity
>>> x = Quantity(2, "km")
>>> jnp.square(w)
Quantity['area'](Array(4, dtype=int64, weak_type=True), unit='km2')
```

## Development

[![Actions Status][actions-badge]][actions-link]
[![Codecov][codecov-badge]][codecov-link]

We welcome contributions!

## Citation

[![DOI][zenodo-badge]][zenodo-link]

If you found this library to be useful and want to support the development and
maintenance of lower-level utility libraries for the scientific community,
please consider citing this work.

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/quaxed/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/quaxed/actions
[codecov-badge]:            https://codecov.io/gh/GalacticDynamics/quaxed/graph/badge.svg?token=9G19ONVD3U
[codecov-link]:             https://codecov.io/gh/GalacticDynamics/quaxed
[pypi-link]:                https://pypi.org/project/quaxed/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/quaxed
[pypi-version]:             https://img.shields.io/pypi/v/quaxed
[rtd-badge]:                https://readthedocs.org/projects/quaxed/badge/?version=latest
[rtd-link]:                 https://quaxed.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/732262318.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850521

<!-- prettier-ignore-end -->
