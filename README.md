<h1 align='center'> quaxed </h1>
<h3 align="center">Pre-<code>Quaxify</code>'ed <code>JAX</code></h3>

<p align="center">
    <a href="https://pypi.org/project/quaxed/"> <img alt="PyPI: quaxed" src="https://img.shields.io/pypi/v/quaxed?style=flat" /> </a>
    <a href="https://pypi.org/project/quaxed/"> <img alt="PyPI versions: quaxed" src="https://img.shields.io/pypi/pyversions/quaxed" /> </a>
    <a href="https://quaxed.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://img.shields.io/badge/read_docs-here-orange" /> </a>
    <a href="https://pypi.org/project/quaxed/"> <img alt="quaxed license" src="https://img.shields.io/github/license/GalacticDynamics/quaxed" /> </a>
</p>
<p align="center">
    <a href="https://github.com/GalacticDynamics/quaxed/actions/workflows/ci.yml"> <img alt="CI status" src="https://github.com/GalacticDynamics/quaxed/actions/workflows/ci.yml/badge.svg?branch=main" /> </a>
    <a href="https://quaxed.readthedocs.io/en/"> <img alt="ReadTheDocs" src="https://readthedocs.org/projects/quaxed/badge/?version=latest" /> </a>
    <a href="https://codecov.io/gh/GalacticDynamics/quaxed"> <img alt="codecov" src="https://codecov.io/gh/GalacticDynamics/quaxed/graph/badge.svg" /> </a>
    <a href="https://scientific-python.org/specs/spec-0000/"> <img alt="ruff" src="https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038" /> </a>
    <a href="https://docs.astral.sh/ruff/"> <img alt="ruff" src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json" /> </a>
    <a href="https://pre-commit.com"> <img alt="pre-commit" src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit" /> </a>
</p>

---

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

[![Read The Docs](https://img.shields.io/badge/read_docs-here-orange)](https://unxt.readthedocs.io/en/)

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
[![Documentation Status][rtd-badge]][rtd-link]
[![codecov][codecov-badge]][codecov-link]
[![SPEC 0 — Minimum Supported Dependencies][spec0-badge]][spec0-link]
[![pre-commit][pre-commit-badge]][pre-commit-link]
[![ruff][ruff-badge]][ruff-link]

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
[pre-commit-badge]:         https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
[pre-commit-link]:          https://pre-commit.com
[pypi-link]:                https://pypi.org/project/quaxed/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/quaxed
[pypi-version]:             https://img.shields.io/pypi/v/quaxed
[rtd-badge]:                https://readthedocs.org/projects/quaxed/badge/?version=latest
[rtd-link]:                 https://quaxed.readthedocs.io/en/latest/?badge=latest
[ruff-badge]:               https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json
[ruff-link]:                https://docs.astral.sh/ruff/
[spec0-badge]:              https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038
[spec0-link]:               https://scientific-python.org/specs/spec-0000/
[zenodo-badge]:             https://zenodo.org/badge/732262318.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850521

<!-- prettier-ignore-end -->
