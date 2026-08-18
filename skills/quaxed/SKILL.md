---
name: quaxed
description:
  Use when writing, reviewing, or debugging code that imports quaxed
  (`quaxed.numpy`, `quaxed.lax`, `quaxed.scipy`, `quaxed.operator`), or that
  uses a quax array-ish type — unxt Quantity, coordinax vectors, galax — through
  JAX functions. Also use when a `quaxed.numpy` fallback UserWarning appears,
  when a custom array-ish type raises "requires ndarray or scalar arguments" or
  comes back as a bare Array, when a function is missing from
  `quaxed.numpy`/`quaxed.lax`, when registering `plum.dispatch` methods on
  `quaxed.numpy` creation functions, or when `quaxed` type stubs are stale after
  a JAX upgrade.
---

# Using Quaxed Effectively

`quaxed` is `quax.quaxify` pre-applied to JAX. `quaxed.numpy.sin` is
`quaxify(jax.numpy.sin)`, resolved lazily by a module-level `__getattr__`. That
is nearly the whole library — a few hundred lines of real code plus a large
`__all__`.

Resolution is two hops, which matters when reading tracebacks:
`quaxed.numpy.__getattr__` delegates to `quaxed.numpy._core`, and `_core`'s own
`__getattr__` does the quaxify and caches the result **on `_core`** via
`setattr`. Nothing is ever cached on the `quaxed.numpy` package module, so every
access keeps going through `quaxed.numpy.__getattr__` — but only the first pays
for the quaxify.

Two exceptions live in `_core`: names in `_DIRECT_TRANSFER` (dtypes and
constants — `bool`, `float32`, `dtype`, `e`, `euler_gamma`, ...) are handed over
**unquaxified**, and `_FILTER_SPEC` gives a few functions a per-argument
`filter_spec` (`kaiser`, `swapaxes`). So "everything in `quaxed.numpy` is
quaxified" is not quite true.

**Read the
[`quax` agent skill](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md)
first for anything about dispatch itself** (it lives in the `quax` repo, not
this one): the `Value`/`ArrayValue` contract, `aval()`/`materialise()`, writing
`quax.register` rules, `Value.default`, plum ambiguity, and the
quaxify-plus-`jax.jit` performance rule (bare `quaxify` is 50–100x slower). This
skill covers only what is specific to `quaxed`, and does not restate any of it.

Checked against quaxed 0.10.5, jax 0.10.0, quax 0.4.1, plum-dispatch 2.5.7,
equinox 0.13.2, Python >=3.11. Docs:
<https://galacticdynamics.github.io/quaxed/>

## Quick start

Import the quaxed module in place of its JAX counterpart. Nothing else changes.

```python
import quaxed.numpy as jnp
from unxt import Quantity

jnp.square(Quantity(2, "km"))
# Quantity(Array(4, dtype=int32, weak_type=True), unit='km2')
```

Available: `quaxed.numpy` (plus `.linalg`, `.fft`), `quaxed.lax` (plus
`.linalg`), `quaxed.scipy` (`.linalg`, `.special`), `quaxed.operator`, and
`quaxed.grad`/`hessian`/`jacfwd`/`jacrev`/`value_and_grad`/`device_put` at the
top level.

The convention across the ecosystem (unxt, coordinax, galax) is
`import quaxed.numpy as jnp`, shadowing the `jax.numpy` name deliberately so
that ordinary-looking JAX code is quax-aware. Some files use `qnp`/`qlax`
instead when both are needed.

## The fallback is the main hazard

`quaxed.numpy` and `quaxed.lax` handle a missing name differently, and the
difference depends on whether the name exists upstream:

| Access                                                     | Behavior                                                                 |
| ---------------------------------------------------------- | ------------------------------------------------------------------------ |
| `quaxed.numpy.<name>`, not in `__all__` but in `jax.numpy` | Returns **bare, unquaxified `jax.numpy.<name>`** after a `UserWarning`   |
| `quaxed.numpy.<name>`, not in `jax.numpy` either           | Raises `AttributeError: module 'quaxed.numpy' has no attribute '<name>'` |
| `quaxed.lax.<name>`, not in `__all__`                      | Raises `AttributeError: Cannot get <name> from quaxed.lax.`              |

The fallback only triggers for names that exist upstream in `jax.numpy`; a
genuine typo still raises. That is what makes the fallback easy to miss — it
fires precisely on the real function you meant to call.

So `quaxed.numpy` is not a complete cover of `jax.numpy`, and it does not tell
you loudly when you have stepped outside it:

```py
import quaxed.numpy as jnp
from unxt import Quantity

jnp.trapezoid(Quantity([1.0, 2.0, 3.0], "m"))
# UserWarning: Missing `quaxed.numpy.trapezoid`. Falling back to `jax.numpy.trapezoid`.
# TypeError: trapezoid requires ndarray or scalar arguments,
#            got <class 'unxt...Quantity[...]'> at position 0.
```

**The failure is misattributed, not silent.** The `TypeError` names your custom
type and reads as though the type is broken. It isn't — quaxed simply has no
`trapezoid`, so bare JAX got the object and rejected it. Diagnose in this order:

1. Is the function in `quaxed.numpy.__all__`? If not, that is the bug.
2. Only then look at your type's registered primitives.

The `UserWarning` that tells you this is emitted on **every** access (the
fallback path does not cache), but a warning is trivially lost in test output.
If your project depends on quax-awareness, promote it:

```toml
# pyproject.toml
[tool.pytest.ini_options]
filterwarnings = ["error:Missing `quaxed:UserWarning"]
```

```ini
# pytest.ini
[pytest]
filterwarnings =
    error:Missing `quaxed:UserWarning
```

The message part is a regex matched against the start of the warning, so
``Missing `quaxed`` catches every fallback and nothing else.

When a function really is missing, file an issue upstream. Locally, either add a
`quax.register` rule for the primitives it lowers to, or write a `plum.dispatch`
method (see "Extending quaxed" below) — do not silently keep using the fallback.

## What is not a plain quaxify

Four places deviate. Knowing they exist prevents "fixing" them.

| Where                          | What                                                                                                     | Why                                                                                                                                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `numpy/_creation_functions.py` | `arange`, `full`, `full_like`, `linspace`, `empty_like`, `ones_like`, `zeros_like` are `plum.Function`s  | Their leading arguments are shapes/dtypes, not arrays, so quaxify has nothing to trace on. plum dispatch routes on the argument types instead. |
| `numpy/_higher_order.py`       | `vectorize` is a reimplementation of `jnp.vectorize`, not `quaxify(jnp.vectorize)`                       | quaxify makes objects _look like_ arrays inside the traced function; this version passes the real object through.                              |
| `_jax.py`                      | `grad`, `hessian`, `jacfwd`, `jacrev`, `value_and_grad` are hand-written and take an extra `filter_spec` | Forwarded to `quaxify` so integer arguments work under `allow_int=True`.                                                                       |
| `lax/_patch.py`                | `quax.register(lax.scan_p)` rebinding to `lax.scan_p.bind`                                               | Works around quax's handling of `scan`.                                                                                                        |

The `vectorize` difference is observable — the inner function sees the real
type, not a proxy:

```python
import jax.numpy as jraw, quax, quaxed.numpy as jnp
from unxt import Quantity


def probe(a):
    return jnp.multiply(a, 2) if hasattr(a, "unit") else jnp.multiply(a, -999)


x = Quantity(jraw.array([1.0, 2.0, 3.0]), "m")

jnp.vectorize(probe, signature="()->()")(x)
# Quantity(Array([2., 4., 6.], dtype=float32), unit='m')          <- sees Quantity

quax.quaxify(jraw.vectorize(probe, signature="()->()"))(x)
# Quantity(Array([-999., -1998., -2997.], dtype=float32), unit='m')  <- took the wrong branch
```

Always reach for `quaxed.numpy.vectorize`, never `quaxify(jnp.vectorize)`.

## Top-level `quaxed.<anything>` has no guard

`quaxed/__init__.py`'s `__getattr__` quaxifies **any** `jax` attribute on demand
and caches it. There is no `__all__` check, and (per a TODO in the source) no
distinction between functions and modules. So every `jax.*` name appears to
exist:

```python
import quaxed

type(quaxed.grad)  # <class 'function'>   — hand-written in _jax.py, correct
type(quaxed.jit)  # <class 'quax._quaxify._Quaxify'>  — quaxify(jax.jit)
type(quaxed.vmap)  # <class 'quax._quaxify._Quaxify'>  — quaxify(jax.vmap)
```

`quaxed.__all__` has ten deliberate names: the four submodules (`lax`, `numpy`,
`scipy`, `experimental`) and six JAX helpers (`device_put`, `grad`, `hessian`,
`jacfwd`, `jacrev`, `value_and_grad`). Anything else you can reach is not API.
`quaxed.jit` and `quaxed.vmap` are accidents of the `__getattr__` and are the
transform mistake the quax skill warns about — quaxifying a transform rather
than the function it transforms. **Use `jax.jit` and `jax.vmap` directly; wrap
the function, not the transform.**

`quaxed.grad` and friends are the exception because they are written by hand to
apply `quaxify` in the right order:

```python
import quaxed, quaxed.numpy as jnp
from unxt import Quantity

quaxed.grad(lambda q: jnp.sum(jnp.square(q)))(Quantity([1.0, 2.0, 3.0], "m"))
# Quantity(Array([2., 4., 6.], dtype=float32), unit='m')
```

Pass `filter_spec` to these when differentiating integer inputs alongside
`allow_int=True`; it goes straight to `quax.quaxify`.

## Extending quaxed for your own array-ish type

Most work is `quax.register` on `lax` primitives — that is quax's job, and
covered by its skill. Everything in `quaxed.numpy` that lowers to primitives
works automatically once those rules exist.

The `quaxed`-specific part is the creation functions, which lower to nothing
useful because their arguments are shapes and dtypes. They split two ways:

| Extend with `@plum.dispatch`                                                       | Cannot be extended — go through primitives |
| ---------------------------------------------------------------------------------- | ------------------------------------------ |
| `arange`, `full`, `full_like`, `linspace`, `empty_like`, `ones_like`, `zeros_like` | `asarray`, `meshgrid`, `tril`, `triu`      |

The right column is `quaxify`'d, so it has no dispatch table to add to.

For the left column, register a method on the same plum function. plum resolves
by qualified name, so defining `arange` under `@plum.dispatch` in your own
module adds a method to `quaxed.numpy.arange`. The canonical example is unxt's
`_src/quantity/register_dispatches.py`:

```py
import plum
from jaxtyping import ArrayLike


@plum.dispatch
def arange(
    start: AbstractQuantity,
    stop: AbstractQuantity | None = None,
    step: AbstractQuantity | None = None,
    **kwargs,
) -> AbstractQuantity:
    unit = start.unit
    return plum.type_unparametrized(start)(
        jax.numpy.arange(
            start.value,
            stop=ustrip(unit, stop) if stop is not None else None,
            step=ustrip(unit, step) if step is not None else None,
            **kwargs,
        ),
        unit=unit,
    )
```

Which then works through the quaxed name:

```python
import quaxed.numpy as jnp
from unxt import Quantity

jnp.arange(Quantity(5, "m"))
# Quantity(Array([0, 1, 2, 3, 4], dtype=int32), unit='m')

jnp.linspace(Quantity(0, "m"), Quantity(1, "m"), 3)
# Quantity(Array([0. , 0.5, 1. ], dtype=float32), unit='m')
```

Notes that save time:

- Import for side effects. The dispatch registration only happens if the module
  defining it is imported; downstream packages import their `register_*` modules
  from their `__init__`.
- Dispatch on your own type, never on `ArrayLike` — quaxed already owns those
  methods and you will create an ambiguity.
- Check your method landed with `len(jnp.arange.methods)` before debugging
  anything else — it is 4 with quaxed alone and 7 once unxt is imported.
  `quaxed.numpy.arange is unxt...register_dispatches.arange` is `True`.
- Use the ordinary `jax.numpy` function inside the body (as above), not the
  quaxed one, or you will recurse.
- **Do not annotate a dispatched parameter with a parametrized builtin**
  (`tuple[int, ...]`, `list[str]`, ...). plum cannot cache such a signature —
  matching has to inspect the elements — and a single uncacheable overload
  disables method caching for the _whole_ function, including quaxed's own
  overloads. Use the bare `tuple`/`list`. `quaxed.numpy.full` carries a comment
  to this effect for exactly this reason; the parametrized spelling cost ~10% on
  warmed calls. Non-dispatched keyword-only parameters are unaffected.

## Typing and stubs

`quaxed` ships `.pyi` stubs because the `__getattr__` indirection erases every
signature. They are not all the same kind of file:

| Stub                                                                      | Origin                                                                                                                                                                 |
| ------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `numpy/__init__.pyi`                                                      | **Generated** at build time by a hatch custom hook (`_tools/make_numpy_stub.py`, entry point `quaxed-make-stubs`). Gitignored — absent from a fresh clone until built. |
| `lax/__init__.pyi`, `lax/linalg.pyi`, `operator.pyi`, `scipy/special.pyi` | Hand-written, tracked in git. Edit them yourself.                                                                                                                      |
| `_version.pyi`                                                            | Tracked; pairs with the VCS-generated `_version.py`.                                                                                                                   |

Only the numpy stub is regenerated by `--remake-stubs`. It is built against the
_installed_ JAX, so after a JAX upgrade it can disagree with reality — giving
type errors on valid code, or silence on invalid code. Regenerate:

```bash
nox -s lint -- --remake-stubs
```

If stub drift is causing trouble, pin `quaxed` and `jax` together with `uv`.

## Troubleshooting

| Symptom                                                               | Cause / fix                                                                                                                                            |
| --------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `UserWarning: Missing quaxed.numpy.<f>. Falling back...`              | `<f>` is not in `quaxed.numpy.__all__`; you are now calling bare JAX. Add a dispatch/register rule or file an issue. Not cached — it warns every call. |
| `TypeError: <f> requires ndarray or scalar arguments, got <YourType>` | Almost always the fallback above, not a defect in your type. Check `__all__` first.                                                                    |
| `AttributeError: Cannot get <f> from quaxed.lax.`                     | `quaxed.lax` has no fallback by design. `__all__` in `lax/__init__.py` is the sole gate; add the name there.                                           |
| Custom type comes back as a bare `Array`                              | A `quax.register` rule is missing and `materialise()` was used, or a fallback path ran. See the quax skill for the dispatch resolution order.          |
| `jnp.arange(YourType(...))` raises or returns the wrong type          | No `@plum.dispatch` method registered; the registering module was never imported.                                                                      |
| plum ambiguity on a creation function                                 | You dispatched on `ArrayLike`; narrow to your own type.                                                                                                |
| Everything works but is very slow                                     | Bare `quaxify` without `jax.jit`. See the quax skill — this is the expensive, invisible mistake.                                                       |
| `quaxed.jit` / `quaxed.vmap` behaves oddly                            | They are unintended `__getattr__` products. Use `jax.jit` / `jax.vmap`.                                                                                |
| Type checker disagrees with runtime                                   | Stale stubs after a JAX bump. `nox -s lint -- --remake-stubs`.                                                                                         |

## Version notes

`_setup.py::JAX_VERSION` is a plain tuple parsed from the installed JAX. Its
only current use is in `lax/__init__.py`, adding `zeros_like` to `__all__` for
JAX >= 0.7.0 — so `quaxed.lax.zeros_like` does not exist on older JAX.

quaxed follows [SPEC 0](https://scientific-python.org/specs/spec-0000/) for
minimum supported dependencies. `quax` is developed at
<https://github.com/nstarman/quax> (the older `patrick-kidger/quax` URL
redirects there) and is on 0.4.x; prior knowledge of quax is likely a version
behind.
