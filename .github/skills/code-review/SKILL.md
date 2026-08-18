---
name: code-review
description:
  Use when reviewing a pull request or diff in the quaxed repository. Covers the
  quaxed-specific defects that generic review misses — a name in `__all__` with
  the wrong wrapping strategy, a plum-dispatched creation function with a
  parametrized-generic annotation that silently disables caching, a JAX version
  gate at the wrong boundary, the numpy-vs-lax fallback asymmetry, and stub
  drift after a JAX upgrade.
---

# Reviewing quaxed changes

Quaxed is `quax.quaxify` pre-applied to JAX: `quaxed.numpy.sin` is
`quaxify(jax.numpy.sin)`, resolved lazily through a module `__getattr__`. Two
properties follow, and they generate nearly every real defect in this
repository:

- **A name only exists if it is in `__all__`.** Adding a function without adding
  its name does nothing; the wrapping machinery is generic, `__all__` is the
  gate.
- **The wrapping strategy is chosen per name, not uniform.** Most names go
  through lazy `quaxify`, but dtypes/constants must bypass it
  (`_DIRECT_TRANSFER`), creation functions must dispatch on shape/dtype instead
  of tracing (`plum.dispatch`), and a few functions are hand-written because
  `quaxify` would do the wrong thing (`vectorize`). Picking the wrong strategy
  for a new or changed name is the most common real bug here.

## Scope of this review

Leave these alone — they are already gated elsewhere:

- Formatting, import order, and lint (`ruff`), plus `pylint`, `mypy`, and
  `pyright`. All of these run in CI via `nox -s lint`
  ([noxfile.py](../../../noxfile.py)); `prek` only runs the faster subset (ruff,
  taplo, codespell, prettier, commit-message format) on each commit. Do not
  restate a lint or type error the CI job will already catch.
- Generic security checklists. There is no user input, no network, no
  serialisation of untrusted data. Injection and XSS questions do not apply.
- Value-level correctness of a `jax.lax`/`jax.numpy` primitive itself — that's
  JAX's problem, not quaxed's. Quaxed only has to route the call correctly.

Spend the review on the sections below instead — one exception to the "pyright
already checks types" rule is called out in
[Creation functions](#creation-functions): a type-correct annotation can still
be the wrong one for `plum`'s caching, and no type checker catches that.

## What changed → what to check

| Change                                                                            | Check                                                             |
| --------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| `__all__` edited in `numpy/_core.py`, `lax/__init__.py`, `operator.py`, `scipy/*` | [Lazy wrapping and `__all__`](#lazy-wrapping-and-__all__)         |
| `numpy/_creation_functions.py`                                                    | [Creation functions](#creation-functions)                         |
| `numpy/_higher_order.py`, `_jax.py`                                               | [Hand-written reimplementations](#hand-written-reimplementations) |
| `lax/_patch.py`                                                                   | [`quax.register` patches](#quaxregister-patches)                  |
| `_setup.py`, any `JAX_VERSION` use                                                | [Version gates](#version-gates)                                   |
| `_tools/make_numpy_stub.py`, any `*.pyi`                                          | [Stubs](#stubs)                                                   |
| A new extension namespace or submodule proposal                                   | [Scope discipline](#scope-discipline)                             |
| Anything under `tests/` or `benchmarks/`                                          | [Tests and benchmarks](#tests-and-benchmarks)                     |

## Lazy wrapping and `__all__`

`numpy/_core.py`, `lax/__init__.py`, `operator.py`, and `scipy/*` wrap `jax.*`
via a module `__getattr__` that quaxifies on first access and caches the result.
A name added to `__all__` gets this treatment **unless** it is carved out — and
the carve-outs are the recurring bug:

- **Dtypes and constants must go in `_DIRECT_TRANSFER`** (`numpy/_core.py`).
  Quaxifying a dtype/constant is wrong — it isn't a function you trace through.
  Two shipped bugs were exactly this omission: `bool`/`bool_` (fix in #107) and
  `float16` (hotfix in #114). A new dtype, constant, or other non-callable added
  to `__all__` without a matching `_DIRECT_TRANSFER` entry is the same bug
  again.
- **`_FILTER_SPEC` needs a new entry when a function's non-array leading
  argument would otherwise get traced.** `kaiser` and `swapaxes` are the
  existing cases; a new function with a similar shape (a size, an axis, a flag
  before the array args) likely needs one too.
- **The numpy-vs-lax fallback asymmetry is deliberate — do not harmonise it.**
  `quaxed.numpy.<missing>` warns and falls back to bare `jax.numpy.<missing>` if
  the name exists upstream; `quaxed.lax.<missing>` always raises
  `AttributeError`. This is called out explicitly in
  [AGENTS.md](../../../AGENTS.md#missing-name-behavior-differs-by-module--preserve-it).
  A PR that makes `lax` warn-and-fall-back like `numpy`, or vice versa, needs a
  stated reason, not just consistency for its own sake.
- **The top-level `quaxed.__getattr__` has no `__all__` guard at all** — it
  quaxifies any `jax` attribute on demand. `quaxed.jit` / `quaxed.vmap` are
  accidents of this, not API (wrapping a transform instead of the function it
  transforms is the mistake the `quax` skill warns about). A PR that starts
  relying on an unlisted top-level name, or that "fixes" this by adding a guard,
  is touching intentional-but-undocumented behaviour — flag it either way.

## Creation functions

`arange`, `full`, `full_like`, `linspace`, `empty_like`, `ones_like`, and
`zeros_like` live in `numpy/_creation_functions.py` as `plum.dispatch`
functions, not lazy `quaxify`, because their leading arguments are shapes/dtypes
rather than arrays — there's nothing for `quaxify` to trace.

The one footgun here is invisible to pyright: **a parametrized generic
(`tuple[int, ...]`, `list[str]`, ...) on a _dispatched_ (positional) parameter
disables plum's method cache for the entire function**, including every other
overload and every downstream package's registered method. This is exactly what
happened to `full` — both overloads annotated `shape` as
`tuple[int, ...] | int`, and `full` silently re-resolved its dispatch on every
call until #200 fixed it to the bare `tuple | int`. The type is _correct_ either
way; only the unparametrized spelling is _fast_. Read the current
`full`/`full_like` code and their `NB:` comment for the pattern to match.

This only applies to parameters plum actually dispatches on:

- **Positional parameters are dispatched** — never parametrize their type.
- **Keyword-only parameters that aren't part of dispatch are unaffected** — e.g.
  `empty_like`'s/`ones_like`'s/`zeros_like`'s keyword-only
  `shape: tuple[int, ...] | None` is fine as written, because dispatch resolves
  on `prototype`/`x`, not on `shape`.

So: does the changed parameter participate in dispatch (positional, or a
`plum.dispatch`-annotated keyword)? If yes and its annotation is a parametrized
generic, that's a caching regression — ask for the bare container, matching
`full`'s comment. If the repo doesn't already have a benchmark exercising the
changed function, check whether [benchmarks/](../../../benchmarks/) should gain
one (see [Tests and benchmarks](#tests-and-benchmarks)).

Two more rules for this file, both already stated in
[skills/quaxed/SKILL.md](../../../skills/quaxed/SKILL.md#extending-quaxed-for-your-own-array-ish-type):

- A new method should dispatch on the caller's own type, never on `ArrayLike` —
  quaxed's own methods already own that, and a second `ArrayLike` method is an
  ambiguity, not an addition.
- The function body should call the plain `jax.numpy`/`jax.lax` version
  internally, not the `quaxed` one, or it recurses.

## Hand-written reimplementations

`numpy/_higher_order.py::vectorize` and the five functions in `_jax.py` (`grad`,
`hessian`, `jacfwd`, `jacrev`, `value_and_grad`) are hand-written because plain
`quaxify` would do the wrong thing:

- **`vectorize` must never become `quaxify(jnp.vectorize)`.** `quaxify` makes
  objects _look like_ arrays inside the traced function; `vectorize` needs the
  _real_ object to reach the inner function (so it can check
  `hasattr(a, "unit")`, dispatch on its type, etc). The docstring in
  `_higher_order.py` has a runnable example of the divergence — if a PR touches
  this file, confirm it still passes the real object through, not a proxy. Prior
  fixes here (#92, #141) were both this class of regression.
- **The `_jax.py` five all take an extra `filter_spec` forwarded to `quaxify`**,
  to support integer arguments under `allow_int=True`. A new function in this
  style that drops `filter_spec`, or forwards it to the wrong call, silently
  breaks integer-argument differentiation.

## `quax.register` patches

`lax/_patch.py` currently has one entry, `scan_p`, rebinding to
`lax.scan_p.bind` to work around quax's default `scan` handling. Any new
`quax.register` patch in this file is a workaround for a specific upstream
behavior — the PR should say which primitive, what quax's default does wrong,
and ideally link the quax issue/PR it works around. A patch with no comment
explaining _why_ it's needed is a maintenance trap: nobody will know when it's
safe to delete after quax fixes the underlying issue.

## Version gates

`_setup.py::JAX_VERSION` is a plain tuple parsed from the installed JAX; its one
current use gates `quaxed.lax.zeros_like` on `>= (0, 7, 0)`. The general rule,
same as it is for any version-gated dependency: **gate on the version where the
upstream API actually changed**, verified against JAX's own changelog/release,
not against whatever flag happens to be nearby. Past fixes here were exactly
boundary corrections: #146 (bumping the `quax` floor for a JAX 0.7.2+
requirement), #100 and #64 (adapting to specific JAX point-release removals).

The sharpest version-boundary risk in this repo isn't even a gate in the source
— it's the **stub generator's implicit assumption that installed JAX's `.pyi`
shape is stable**. #205 is the canonical example: a same-day JAX 0.11.1 release
changed `jax/numpy/__init__.pyi` in two ways the generator didn't expect, and
every PR (including one already in CI) started failing on `Invalid syntax` in
the generated stub. A PR touching `_tools/make_numpy_stub.py` should be read
against this precedent — is it handling one JAX `.pyi` shape, or the general
pattern the shape came from?

## Stubs

Only `numpy/__init__.pyi` is generated (`_tools/make_numpy_stub.py`, a hatch
build hook) — it's gitignored and absent until the package is built, and it's
built against _whatever JAX happens to be installed_, so it can silently
disagree with a different JAX version (#153, #205 are the two times this broke
CI). `lax/__init__.pyi`, `lax/linalg.pyi`, `operator.pyi`, and
`scipy/special.pyi` are hand-written and tracked in git — **these do not
regenerate themselves**. A PR that adds a name to `lax/__init__.py`'s `__all__`
(or `operator.py`'s, or `scipy/special.py`'s) without a matching edit to its
hand-written `.pyi` has shipped a stub that silently disagrees with the runtime.
`tests/static/numpy.pyi`, checked by `nox -s mypy_test`, is how stub correctness
against real usage is asserted for the generated stub — check whether a change
to `numpy/_core.py`'s `__all__` needs a corresponding line there.

## Scope discipline

Quaxed's extension namespaces are `fft` and `linalg` — the only two extensions
the
[Python Array API standard](https://data-apis.org/array-api/latest/extensions/index.html)
defines. A proposal to add another namespace (e.g. `lib`, mirroring `numpy.lib`)
was investigated and deliberately rejected: `jax.numpy` has no such submodule to
wrap, and `numpy.lib`'s contents aren't Array-API surface. If a PR adds a new
top-level namespace, the bar is a concrete downstream call site needing a
specific attribute — not speculative completeness with `numpy`.

## Tests and benchmarks

- `tests/myarray.py` is the canonical `quax.ArrayValue` fixture with
  `quax.register` rules for the primitives it's tested against — extend it
  rather than inventing a new fixture type, matching
  [AGENTS.md](../../../AGENTS.md#testing).
- **Doctests are real tests, but only for `.py`/`.rst`.** The root
  [conftest.py](../../../conftest.py) runs sybil over those two patterns; a
  `>>>` example inside a docstring is exercised by the normal test session.
- **`skills/quaxed/SKILL.md`'s ` ```python ` blocks run in a separate, unlocked
  session**,
  [tests/test_skill_examples.py](../../../tests/test_skill_examples.py) via
  `nox -s skill_examples` — not the default `nox -s test`. It needs `unxt`,
  which can't be added to `uv.lock` (unxt depends on quaxed; uv can't resolve a
  project against a differently-sourced package sharing its own name), so this
  session installs into its own venv instead. A block meant to illustrate a
  failure or pseudocode, not to run, is fenced ` ```py ` instead of
  ` ```python ` so the test skips it — check that a genuinely-runnable new
  example isn't accidentally fenced that way, and that a PR touching the skill's
  examples mentions whether `nox -s skill_examples` was run, since it won't show
  up in the normal `nox -s test` output.
- A new function added per [AGENTS.md](../../../AGENTS.md#adding-a-function)
  should come with a doctest (step 4) and, if it touches the dispatch/hot path,
  a benchmark. `benchmarks/` (CodSpeed, added in #196) exists because a caching
  regression like #200 is invisible to correctness tests — they still pass, just
  slower. A PR changing `numpy/_core.py`'s `__getattr__`, anything in
  `numpy/_creation_functions.py`, or `_setup.py` should say whether the
  benchmarks were run, not just that pytest passed.

## Repo conventions

- **`uv run`/`nox -s ...` for everything** — never bare `python` or `pytest`.
  See [AGENTS.md](../../../AGENTS.md#essential-commands) for the session list.
- Commits use gitmoji plus conventional commits (`.czrc`,
  cz-conventional-gitmoji) — match the existing log style.
- `--remake-stubs` regenerates the numpy stub before `test`/`lint`/`mypy_test`/
  `pyright_lint`; a PR whose CI is failing only on the generated stub likely
  needs this, not a source change.

## Further reading

- [skills/quaxed/SKILL.md](../../../skills/quaxed/SKILL.md) — the four wrapping
  strategies, the fallback hazard, extending quaxed for a new array-ish type,
  and a troubleshooting table.
- [AGENTS.md](../../../AGENTS.md) — architecture table, essential commands, and
  the pitfalls list this skill draws from.
- the
  [`quax` agent skill](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md)
  — the dispatch foundation quaxed is built on (upstream, not in this repo).
