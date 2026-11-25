# Quaxed — Agent Instructions

Quaxed is `quax.quaxify` pre-applied to JAX and related libraries, so custom
array-ish objects work with functions that were never written to accept them.

For using or extending quaxed from _outside_ this repo, read
[skills/quaxed/SKILL.md](skills/quaxed/SKILL.md). This file is for working
inside the repo.

## Code graph

This repository has a committed graphify knowledge graph at
`graphify-out/graph.json`: every symbol, the links between them, the
architectural hubs, and which subsystem each symbol belongs to. It is built from
the source by tree-sitter, so it is as current as the last commit.

Querying it is cheaper and more accurate than grepping. The rules:

- For any question about this codebase, query the graph **before** your first
  grep, glob or source-file read:
  - `graphify query "<question>"` — a scoped subgraph for a plain-language
    question
  - `graphify explain "<symbol>"` — one symbol and its neighbours
  - `graphify path "<A>" "<B>"` — how two things connect
  - `graphify-out/GRAPH_REPORT.md` — hubs and subsystems, for broad architecture
    review only

  Read source files for the lines a query has already located, not to find them.

- Put this rule in every subagent prompt you write that explores the code. A
  subagent that greps from scratch undoes the point of the graph.
- After changing code, refresh it: `PYTHONHASHSEED=0 graphify update . --force`
  — tree-sitter only, no model call, no network. `--force` keeps your rebuild
  equal to the one the commit hook and the drift workflow produce; without it
  the rebuild is incremental and drifts.
- Ignore `built_at_commit` in the graph, and `GRAPH_REPORT.md`'s "built from
  commit … compare to check if the graph is stale". `update` does not rewrite
  that field, so it names an old commit for ever and is not a staleness signal.
  Do not rebuild on the strength of it.
- Use the pinned release — the `graphifyy==` pin in `.pre-commit-config.yaml` —
  so your rebuild clusters the same way the hook does:
  `uvx --from graphifyy==<pinned version> graphify ...`.
- `graphify-out/graph.json` is generated. Do not read it directly, do not review
  it in a diff, and do not hand-resolve a conflict in it: rebuild it instead.
- Skip the graph only when the task is about the graph tooling itself being
  stale or wrong, or when asked to. An uncommitted or just-rebuilt graph file is
  normal and is not a reason to skip it.
- `.graphifyignore` fixes what is indexed. Markdown, images and YAML are out, so
  the graph answers questions about the Python only.

Scope and merge-driver setup, and what to do when a rebuild blocks your commit,
are in [.github/CONTRIBUTING.md](.github/CONTRIBUTING.md#the-code-graph).

## Essential Commands

```bash
uv tool install --with nox-uv nox   # one-time
nox -s test                         # pytest (incl. doctests) + mypy on tests/static
nox -s lint                         # prek + pylint + mypy + pyright
nox -s docs                         # build docs
nox -s pytest -- tests/unit/test_numpy   # a subset
nox -s skill_examples                    # skills/quaxed/SKILL.md examples (needs unxt; not part of `test`)
```

Pass `--remake-stubs` to `test`, `lint`, `mypy_test`, or `pyright_lint` to
regenerate the numpy stub first:

```bash
nox -s lint -- --remake-stubs
```

## Architecture

Every public name is wrapped by one of four strategies. Identifying which one
applies is the first step in any change.

| Strategy                                | Where                                                                                                       | Notes                                                                                                                                                                                                                                                                                                                    |
| --------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Lazy `quaxify` via module `__getattr__` | `numpy/_core.py`, `lax/__init__.py`, `operator.py`, `scipy/*`                                               | `__all__` is the gate. `numpy/__init__.py` delegates to `_core`, which quaxifies and caches onto `_core` — never onto the `quaxed.numpy` package module. In `_core`, `_DIRECT_TRANSFER` names (dtypes, constants) pass through unquaxified and `_FILTER_SPEC` sets a per-argument `filter_spec` for `kaiser`/`swapaxes`. |
| `plum.dispatch`                         | `numpy/_creation_functions.py`                                                                              | For functions whose leading args are shapes/dtypes, not arrays. These are the downstream extension points.                                                                                                                                                                                                               |
| Hand-written reimplementation           | `numpy/_higher_order.py` (`vectorize`), `_jax.py` (`grad`, `hessian`, `jacfwd`, `jacrev`, `value_and_grad`) | `vectorize` passes the real object through instead of an array-looking proxy. The `_jax.py` functions add `filter_spec`, forwarded to `quaxify`.                                                                                                                                                                         |
| `quax.register` patch                   | `lax/_patch.py`                                                                                             | Currently only `lax.scan_p`.                                                                                                                                                                                                                                                                                             |

`_setup.py::JAX_VERSION` is the version gate; its one use adds
`quaxed.lax.zeros_like` for JAX >= 0.7.0.

## Missing-name behavior differs by module — preserve it

| Module         | Name not in `__all__`                                                                                                                           |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| `quaxed.numpy` | If the name exists in `jax.numpy`: `UserWarning`, then returns **bare `jax.numpy`**, uncached (warns every access). Otherwise `AttributeError`. |
| `quaxed.lax`   | `AttributeError`, always                                                                                                                        |

This asymmetry is deliberate and load-bearing for downstream debugging. Do not
"harmonise" the two without a decision to do so.

## Adding a function

1. Add the name to the right `__all__` (`numpy/_core.py`, `lax/__init__.py`,
   `operator.py`, `scipy/special.py`). For most functions that is the whole
   change — `__getattr__` does the rest.
2. If its leading arguments are shapes/dtypes rather than arrays, it belongs in
   `numpy/_creation_functions.py` as a `plum.dispatch` function instead.
3. Regenerate the numpy stub if you touched `numpy`:
   `nox -s lint -- --remake-stubs`.
4. Add a doctest — see Testing.

## Stubs

Only `numpy/__init__.pyi` is generated (hatch custom build hook, path
`src/quaxed/_tools/make_numpy_stub.py`); it is **gitignored** and absent from a
fresh clone until the package is built. `lax/__init__.pyi`, `lax/linalg.pyi`,
`operator.pyi`, and `scipy/special.pyi` are hand-written and tracked — edit them
directly. `_version.pyi` pairs with the VCS-generated `_version.py`.

## Testing

- `tests/myarray.py` is the canonical `quax.ArrayValue` fixture, with
  `quax.register` rules for a long list of `lax` primitives. Use it rather than
  inventing a new type; add primitive rules there when a test needs one.
- **Docstrings are tests.** The root `conftest.py` installs
  [sybil](https://sybil.readthedocs.io) collecting `*.py` and `*.rst` via
  `DocTestParser`, `PythonCodeBlockParser`, and `SkipParser`. Any `>>>` example
  in `src/` runs in CI. Doctests are skipped on Windows (numpy 2.0 scalar repr).
- `tests/static/numpy.pyi` is type-checked by `nox -s mypy_test` — it is how
  stub correctness is asserted.
- **`skills/quaxed/SKILL.md`'s example blocks are tests too**, but not via sybil
  (that only collects `.py`/`.rst`). `tests/test_skill_examples.py` execs every
  ` ```python ` block in the skill, run via `nox -s skill_examples` — a
  separate, unlocked session, since it needs `unxt`, which can't join `uv.lock`
  (it depends on `quaxed`, and uv can't resolve the project against a
  differently-sourced package sharing its own name). A block meant to illustrate
  a failure or non-runnable pseudocode is fenced ` ```py ` instead, so the test
  skips it.
- Test layout mirrors the source: `tests/unit/test_numpy`, `test_lax`,
  `test_scipy`, `test_operator`.

## Pitfalls

- **`__all__` is the entire public API.** Adding a function without adding the
  name does nothing; removing a name silently switches `quaxed.numpy` users onto
  bare JAX via the fallback.
- **Top-level `quaxed.__getattr__` has no `__all__` guard** — it quaxifies any
  `jax` attribute, so `quaxed.jit` and `quaxed.vmap` resolve to
  `quaxify(jax.jit)`/`quaxify(jax.vmap)`. Those are accidents, not API; the
  intentional surface is the ten names in `quaxed.__all__` — four submodules
  plus six JAX helpers. A TODO in `__init__.py` notes it also cannot distinguish
  functions from modules.
- **The numpy stub is gitignored** — a fresh clone has no `numpy/__init__.pyi`,
  so type checkers behave differently before and after a build.
- **Don't `quaxify(jnp.vectorize)`** — `numpy/_higher_order.py::vectorize` is a
  deliberate reimplementation; the docstring explains why.
- **Commits use gitmoji** (`.czrc`, cz-conventional-gitmoji). Match the existing
  log style.

## Dependencies

See `pyproject.toml`. Core: `jax`, `quax`, `equinox`, `plum-dispatch`,
`jaxtyping`, `optype`. Minimum supported versions follow
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

## Further Reading

- [.github/skills/code-review/SKILL.md](.github/skills/code-review/SKILL.md) —
  what to look for when reviewing a quaxed change (also picked up by GitHub
  Copilot code review)
- [README.md](README.md) — install, quick start, contributor basics
- [docs/](docs/) — published at <https://galacticdynamics.github.io/quaxed/>
- [skills/quaxed/SKILL.md](skills/quaxed/SKILL.md) — using and extending quaxed
- the
  [`quax` agent skill](https://github.com/nstarman/quax/blob/main/skills/quax/SKILL.md)
  — the dispatch foundation (upstream, not in this repo)
