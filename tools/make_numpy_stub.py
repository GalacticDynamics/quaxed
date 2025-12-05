"""Custom hatch build hook to generate type stubs at build time."""

from __future__ import annotations

import re
import textwrap
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

# ---------------------------------------------------------------------------
# Stub generation logic (mirrors tools/update_numpy_stub.py)
# ---------------------------------------------------------------------------

# RE_UNARY matches unary ufunc stub signatures in the upstream jax.numpy .pyi.
# It targets lines of the form:
#
#     def func(x: ArrayLike, /, <tail>) -> Array: ...
#
# Captured groups:
#   - name: the function name (e.g. "sin", "sqrt")
#   - tail: everything after the positional-only slash up to the closing ')'
#
# The pattern is multiline and anchored at the start of a line to avoid
# accidental matches in other contexts.
RE_UNARY = re.compile(
    r"^def (?P<name>\w+)\(x: ArrayLike, /(?P<tail>[^)]*)\) -> Array: \.\.\.$",
    re.MULTILINE,
)

# RE_BINARY matches binary ufunc stub signatures in the upstream jax.numpy .pyi.
# It targets lines of the form:
#
#     def func(x: ArrayLike, y: ArrayLike, /, <tail>) -> Array: ...
#
# Captured groups:
#   - name: the function name (e.g. "add", "maximum")
#   - tail: everything after the positional-only slash up to the closing ')'
#
# Like RE_UNARY, this is anchored to full lines in MULTILINE mode.
RE_BINARY = re.compile(
    (
        r"^def (?P<name>\w+)\(x: ArrayLike, y: ArrayLike, /"
        r"(?P<tail>[^)]*)\) -> Array: \.\.\.$"
    ),
    re.MULTILINE,
)

BINARY_UFUNC = textwrap.dedent(
    """
    class BinaryUfunc(Protocol):
      @property
      def nin(self) -> int: ...
      @property
      def nout(self) -> int: ...
      @property
      def nargs(self) -> int: ...
      @property
      def identity(self) -> builtins.bool | int | float: ...
      @overload
      def __call__(self, x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
      @overload
      def __call__(self, x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
      @overload
      def __call__(self, x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
      @overload
      def __call__(self, x: ArrayLike, y: ArrayLike, /) -> Array: ...
      def reduce(
          self,
          a: ArrayLike,
          /,
          *,
          axis: int | None = 0,
          dtype: DTypeLike | None = None,
          out: None = None,
          keepdims: builtins.bool = False,
          initial: ArrayLike | None = None,
          where: ArrayLike | None = None,
      ) -> Array: ...
      def accumulate(
          self,
          a: ArrayLike,
          /,
          *,
          axis: int = 0,
          dtype: DTypeLike | None = None,
          out: None = None,
      ) -> Array: ...
      def at(
          self,
          a: ArrayLike,
          indices: Any,
          b: ArrayLike | None = None,
          /,
          *,
          inplace: builtins.bool = True,
      ) -> Array: ...
      def reduceat(
          self,
          a: ArrayLike,
          indices: Any,
          *,
          axis: int = 0,
          dtype: DTypeLike | None = None,
          out: None = None,
      ) -> Array: ...
      def outer(
          self,
          a: ArrayLike,
          b: ArrayLike,
          /,
      ) -> Array: ...
    """
)


def _add_typealias_imports(text: str, /) -> str:
    """Rewrite upstream stub imports to introduce quax-aware type aliases.

    This function performs a series of textual substitutions on the upstream
    `jax.numpy` stub:

    * Extends the imported symbols from ``typing`` to include ``TypeAlias``.
    * Injects an import of ``quax`` alongside the existing ``os`` import.
    * Renames ``ArrayLike`` to an internal alias ``_ArrayLike`` in the public
      imports and then introduces a new ``ArrayLike`` type alias that widens
      the upstream definition to also accept ``quax.ArrayValue``.
    * Extends the generic type variables with ``_ArrayValueT``, which is
      bounded by ``quax.ArrayValue`` and is used to express value-preserving
      overloads.

    The transformation assumes the upstream stub follows the structure
    produced by JAX's stub generation scripts (in particular, that the
    import blocks and ``_T`` definition appear exactly once).

    Args:
        text: The complete contents of the upstream ``jax.numpy`` stub file.

    Returns
    -------
        The rewritten stub text with additional imports and the widened
        ``ArrayLike`` type alias.

    """
    text = text.replace(
        "Protocol, TypeVar, Union, overload",
        "Protocol, TypeAlias, TypeVar, Union, overload",
        1,
    )
    text = text.replace("import os\n", "import os\nimport quax\n", 1)
    text = text.replace(
        "Array, ArrayLike, DType",
        "Array,\n    ArrayLike as _ArrayLike,\n    DType",
        1,
    )
    alias = "ArrayLike: TypeAlias = _ArrayLike | quax.ArrayValue\n\n"
    text = text.replace("import numpy as _np\n\n", f"import numpy as _np\n\n{alias}", 1)
    return text.replace(
        "_T = TypeVar('_T')",
        (
            "_T = TypeVar('_T')\n"
            "_ArrayValueT = TypeVar('_ArrayValueT', bound=quax.ArrayValue)"
        ),
        1,
    )


def _replace_binary_ufunc(text: str, /) -> str:
    """Replace the upstream BinaryUfunc protocol with a quax-aware version.

    The upstream ``BinaryUfunc`` protocol is matched as a contiguous block
    using a regular expression and then replaced with a locally defined
    version that:

    * Provides explicit attributes (``nin``, ``nout``, ``nargs``, ``identity``).
    * Defines overloads for calls involving ``_ArrayValueT`` and ``ArrayLike``,
      preserving the input ``_ArrayValueT`` type when possible.
    * Adds richer method signatures for ``reduce``, ``accumulate``, ``at``,
      ``reduceat``, and ``outer``.

    The replacement is applied at most once, and only if a matching block
    corresponding to the upstream ``BinaryUfunc`` definition is found.

    Args:
        text: The stub file contents in which the ``BinaryUfunc`` protocol
            should be replaced.

    Returns
    -------
        The stub text with the original ``BinaryUfunc`` protocol replaced by
        the custom ``BINARY_UFUNC`` definition.

    """
    pattern = re.compile(
        (
            r"class BinaryUfunc\(Protocol\):\n(?:  .+\n)+?  def outer"
            r"\(self, a: ArrayLike, b: ArrayLike, /\) -> Array: ...\n"
        ),
    )
    return pattern.sub(BINARY_UFUNC, text, count=1)


def _rewrite_unary(text: str, /) -> str:
    """Rewrite unary ufunc stubs to preserve ArrayValue where possible.

    This function searches for unary function definitions of the form::

        def name(x: ArrayLike, /..., ...) -> Array: ...

    using ``RE_UNARY`` and replaces each match with two overloads:

    * One overload in which ``x`` is ``_ArrayValueT``, returning
      ``_ArrayValueT`` (value-preserving behavior for quax arrays).
    * One overload in which ``x`` is general ``ArrayLike``, returning
      ``Array``.

    The original trailing parameters (captured as ``tail``) are preserved
    verbatim, including keyword-only markers and default values.

    Args:
        text: The stub text in which unary function signatures should be
            rewritten.

    Returns
    -------
        The stub text with matching unary ufunc definitions replaced by
        overload-based signatures.

    """

    def repl(match: re.Match[str], /) -> str:
        tail = match.group("tail")
        name = match.group("name")
        return (
            f"@overload\n"
            f"def {name}(x: _ArrayValueT, /{tail}) -> _ArrayValueT: ...\n"
            f"@overload\n"
            f"def {name}(x: ArrayLike, /{tail}) -> Array: ..."
        )

    return RE_UNARY.sub(repl, text)


def _rewrite_binary(text: str, /) -> str:
    """Rewrite binary ufunc stubs to preserve ArrayValue where possible.

    This function searches for binary function definitions of the form::

        def name(x: ArrayLike, y: ArrayLike, /..., ...) -> Array: ...

    using ``RE_BINARY`` and replaces each match with four overloads:

    * ``(x: _ArrayValueT, y: ArrayLike) -> _ArrayValueT``
    * ``(x: ArrayLike, y: _ArrayValueT) -> _ArrayValueT``
    * ``(x: _ArrayValueT, y: _ArrayValueT) -> _ArrayValueT``
    * ``(x: ArrayLike, y: ArrayLike) -> Array``

    This models binary ufuncs whose result stays in the ``ArrayValue`` type
    whenever at least one operand is an ``_ArrayValueT`` instance, while
    retaining the upstream ``Array`` return type for purely ``ArrayLike``
    inputs.

    The original trailing parameters (captured as ``tail``) are preserved
    verbatim.

    Args:
        text: The stub text in which binary function signatures should be
            rewritten.

    Returns
    -------
        The stub text with matching binary ufunc definitions replaced by
        overload-based signatures.

    """

    def repl(match: re.Match[str], /) -> str:
        tail = match.group("tail")
        name = match.group("name")
        return "\n".join(
            (
                "@overload",
                (
                    f"def {name}(x: _ArrayValueT, y: ArrayLike, /{tail}) "
                    "-> _ArrayValueT: ..."
                ),
                "@overload",
                (
                    f"def {name}(x: ArrayLike, y: _ArrayValueT, /{tail}) "
                    "-> _ArrayValueT: ..."
                ),
                "@overload",
                (
                    f"def {name}(x: _ArrayValueT, y: _ArrayValueT, /{tail}) "
                    "-> _ArrayValueT: ..."
                ),
                "@overload",
                f"def {name}(x: ArrayLike, y: ArrayLike, /{tail}) -> Array: ...",
            )
        )

    return RE_BINARY.sub(repl, text)


def generate_numpy_stub(output: Path, /) -> None:
    """Generate the ``quaxed.numpy`` stub from the upstream ``jax.numpy`` stub.

    This function locates the installed ``jax.numpy`` stub (``.pyi``),
    transforms it to be ``quax.ArrayValue``-aware, and writes the resulting
    stub to the requested output path. The transformation consists of:

    * Adding additional type imports and a widened ``ArrayLike`` type alias
      that accepts both the upstream ``ArrayLike`` and ``quax.ArrayValue``.
    * Replacing the upstream ``BinaryUfunc`` protocol with a richer protocol
      that preserves ``_ArrayValueT`` in its overloads.
    * Rewriting unary and binary ufunc signatures to introduce overloads that
      propagate ``_ArrayValueT`` where possible.

    The output directory is created if it does not already exist.

    Args:
        output: The target path for the generated ``__init__.pyi`` file under
            ``quaxed/numpy``.

    """
    upstream = Path(jnp.__file__).with_suffix(".pyi")
    text = upstream.read_text()
    text = _add_typealias_imports(text)
    text = _replace_binary_ufunc(text)
    text = _rewrite_unary(text)
    text = _rewrite_binary(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


# ---------------------------------------------------------------------------
# Hatch build hook
# ---------------------------------------------------------------------------


class NumPyStubBuildHook(BuildHookInterface):
    """Hatch build hook that generates the ``quaxed.numpy`` stub before build.

    This hook integrates the stub generation logic into Hatch's build process.
    During initialization it:

    * Resolves the project root directory.
    * Computes the path for ``src/quaxed/numpy/__init__.pyi``.
    * Invokes :func:`generate_numpy_stub` to regenerate the stub from the
      installed ``jax.numpy`` stub.
    * Registers the generated stub in ``build_data['force_include']`` so that
      the file is always included in the built distribution, even if it would
      otherwise be excluded by default file discovery.

    The hook is registered under the plugin name
    ``"quaxed-numpy-stub-hook"`` and can be enabled via Hatch's pyproject
    configuration.

    """

    PLUGIN_NAME = "quaxed-numpy-stub-hook"

    def initialize(
        self,
        _: str,
        build_data: dict[str, Any],
    ) -> None:
        """Generate the numpy stub before building."""
        root = Path(self.root)
        stub_path = root / "src" / "quaxed" / "numpy" / "__init__.pyi"

        # Generate the stub
        self.app.display_info(f"Generating numpy stub: {stub_path}")
        generate_numpy_stub(stub_path)

        # Ensure it's included in the build artifacts
        build_data.setdefault("force_include", {})[str(stub_path)] = (
            "quaxed/numpy/__init__.pyi"
        )
