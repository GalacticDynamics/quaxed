# quaxed

[![Actions Status][actions-badge]][actions-link]
[![Documentation Status][rtd-badge]][rtd-link]

[![PyPI version][pypi-version]][pypi-link]
[![Conda-Forge][conda-badge]][conda-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

[![DOI][zenodo-badge]][zenodo-link]

<!-- [![GitHub Discussion][github-discussions-badge]][github-discussions-link] -->

<!-- SPHINX-START -->

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/GalacticDynamics/quaxed/workflows/CI/badge.svg
[actions-link]:             https://github.com/GalacticDynamics/quaxed/actions
[conda-badge]:              https://img.shields.io/conda/vn/conda-forge/quaxed
[conda-link]:               https://github.com/conda-forge/quaxed-feedstock
<!-- [github-discussions-badge]: https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github
[github-discussions-link]:  https://github.com/GalacticDynamics/quaxed/discussions -->
[pypi-link]:                https://pypi.org/project/quaxed/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/quaxed
[pypi-version]:             https://img.shields.io/pypi/v/quaxed
[rtd-badge]:                https://readthedocs.org/projects/quaxed/badge/?version=latest
[rtd-link]:                 https://quaxed.readthedocs.io/en/latest/?badge=latest
[zenodo-badge]:             https://zenodo.org/badge/732262318.svg
[zenodo-link]:              https://zenodo.org/doi/10.5281/zenodo.10850521


<!-- prettier-ignore-end -->

`Quaxed` wraps [jax](https://jax.readthedocs.io/en/latest/) libraries (using
[`quax`](https://docs.kidger.site/quax/)) to enable using those libraries with
custom array-ish objects, not only jax arrays.

To understand how `quax` works it's magic, see
[`quax.quaxify`](https://docs.kidger.site/quax/api/quax/#quax.quaxify) and the
[tutorials](https://docs.kidger.site/quax/examples/custom_rules/).

To use this library, it's as simple as:

```pycon
# Import pre-quaxified library
>>> import quaxed.numpy as qnp  # this is quaxify(jax.numpy)

# As an example, let's import an array-ish object
>>> from unxt import Quantity
>>> x = Quantity(2, "km")
>>> qnp.square(w)
Quantity['area'](Array(4, dtype=int64, weak_type=True), unit='km2')
```
