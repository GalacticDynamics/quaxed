# /// script
# dependencies = [
#   "jax>=0.5.3",
#   "hatchling",
# ]
# ///
"""Custom hatch build hook to generate type stubs at build time."""

import logging
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

import jax.numpy as jnp
from hatchling.builders.hooks.plugin.interface import BuildHookInterface

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stub generation logic
# ---------------------------------------------------------------------------


def _add_type_ignore(line: str, ignore_code: str, /) -> str:
    """Add a type: ignore comment to a line, or append to existing ignore.

    If the line already has a type: ignore comment, this will append the new
    ignore code to the existing list (comma-separated). If there's no existing
    type: ignore comment, it will add one with the given code.

    Args:
        line: The line to add the ignore comment to.
        ignore_code: The ignore code to add (e.g., "type-arg", "valid-type").

    Returns
    -------
        The line with the type: ignore comment added or updated.

    Examples
    --------
        >>> _add_type_ignore("def foo(): ...", "type-arg")
        "def foo(): ...  # type: ignore[type-arg]"
        >>> _add_type_ignore("def foo(): ...  # type: ignore[type-arg]", "valid-type")
        "def foo(): ...  # type: ignore[type-arg, valid-type]"

    """
    # Check if line already has a type: ignore comment
    if "# type: ignore[" in line:
        # Find the closing bracket and insert the new code before it
        # Handle both "# type: ignore[code]" and "# type: ignore[code1, code2]"
        return line.replace("]", f", {ignore_code}]", 1)
    if "# type: ignore" in line:
        # Has type: ignore but no brackets, replace with bracketed version
        return line.replace("# type: ignore", f"# type: ignore[{ignore_code}]", 1)
    # No type: ignore comment, add one
    return f"{line}  # type: ignore[{ignore_code}]"


# RE_SINGLE_ARRAYLIKE_PARAM matches function signatures where the first parameter
# has type ArrayLike, regardless of parameter name or whether it uses positional-only
# syntax. This pattern explicitly excludes functions with multiple ArrayLike parameters.
#
# Matched patterns:
#     def func(a: ArrayLike, <rest>) -> Array: ...              (no /, with rest)
#     def func(x: ArrayLike, /, <rest>) -> Array: ...            (with /, with rest)
#     def func(m: ArrayLike) -> Array: ...                       (no / no rest)
#     def func(x: ArrayLike, /) -> Array: ...                    (with / no rest)
#     def func(\n    a: ArrayLike,\n    <rest>\n) -> Array: ...  (multiline)
#
# Explicitly NOT matched (handled by RE_BINARY):
#     def func(a: ArrayLike, b: ArrayLike, ...) -> Array: ...
#     def func(x: ArrayLike, /, y: ArrayLike, ...) -> Array: ...
#
# Captured groups:
#   - name: the function name
#   - param: the parameter name (a, x, m, z, etc.)
#   - tail: everything after ArrayLike up to the closing ')' (may be empty)
#
# The negative lookahead (?![^)]*?\w+:\s*[^,)]*?ArrayLike) checks only within the current
# function (up to the closing paren) to ensure there's no second parameter with ArrayLike
# in its type annotation. The pattern \w+:\s*[^,)]*?ArrayLike matches parameter annotations
# like "y: ArrayLike" or "b: SomeType | ArrayLike".
#
# The tail pattern uses (?:(?!->).)* to match any character except those in a "->" sequence,
# preventing the pattern from matching across function boundaries where it might find a
# different function's "-> Array" return type.
RE_SINGLE_ARRAYLIKE_PARAM = re.compile(
    r"^def (?P<name>\w+)\(\s*(?P<param>\w+): ArrayLike(?![^)]*?\w+:\s*[^,)]*?ArrayLike)(?P<tail>(?:(?!->).)*?)\) -> Array: \.\.\.$",
    re.MULTILINE | re.DOTALL,
)

# RE_BINARY matches binary ufunc stub signatures in the upstream jax.numpy .pyi.
# It targets functions where the first parameter is exactly ArrayLike, the second
# parameter has ArrayLike in its type (possibly with unions), and NO other parameter
# contains ArrayLike in its type annotation.
#
# Matched patterns:
#     def func(x: ArrayLike, y: ArrayLike, /) -> Array: ...
#     def func(x: ArrayLike, y: ArrayLike, /, out: None = ...) -> Array: ...
#     def func(a: ArrayLike, b: ArrayLike, axis: int = ...) -> Array: ...
#     def func(a: ArrayLike, fill_value: ArrayLike | None, ...) -> Array: ...
#
# Explicitly NOT matched:
#     def func(x: ArrayLike, y: ArrayLike, /, where: ArrayLike | None = ...) -> Array: ...
#     def func(a: ArrayLike | DuckTypedArray, fill_value: ArrayLike, ...) -> Array: ...
#
# Captured groups:
#   - name: the function name (e.g. "add", "maximum")
#   - param1: first parameter name
#   - param2: second parameter name
#   - param2type: second parameter's full type up to comma/paren (may include unions)
#   - tail: everything after the second parameter's type up to the closing ')'
#
# The negative lookahead ensures no other parameter has ArrayLike in its type.
RE_BINARY = re.compile(
    r"^def (?P<name>\w+)\(\s*(?P<param1>\w+): ArrayLike,\s*(?P<param2>\w+): (?P<param2type>[^,)]*?ArrayLike[^,)]*)(?![^)]*?\w+:\s*[^,)]*?ArrayLike)(?P<tail>(?:(?!->).)*?)\) -> Array: \.\.\.$",
    re.MULTILINE | re.DOTALL,
)


# RE_BINARY_WITH_UNION_FIRST matches binary function signatures where the first parameter
# has a union type containing ArrayLike, and the second parameter has ArrayLike in its type.
# This handles cases like full_like(a: ArrayLike | DuckTypedArray, fill_value: ArrayLike,
# ...).
#
# Matched patterns:
#     def full_like(a: ArrayLike | DuckTypedArray, fill_value: ArrayLike, ...) -> Array:
#     ...
#
# Explicitly NOT matched:
#     Functions with first param exactly ArrayLike (handled by RE_BINARY)
#     Functions with more than two ArrayLike params
RE_BINARY_WITH_UNION_FIRST = re.compile(
    r"^def (?P<name>\w+)\(\s*(?P<param1>\w+): (?P<param1type>(?:[^,]|,(?!\s*\w+:))*?ArrayLike(?:[^,]|,(?!\s*\w+:))*?\|(?:[^,]|,(?!\s*\w+:))*?),\s*"
    r"(?P<param2>\w+): (?P<param2type>[^,)]*?ArrayLike[^,)]*)"
    r"(?![^)]*?\w+:\s*[^,)]*?ArrayLike)(?P<tail>(?:(?!->).)*?)\) -> Array: \.\.\.$",
    re.MULTILINE | re.DOTALL,
)

# RE_MULTI_PARAM_FIRST_ARRAYLIKE matches function signatures where the first parameter
# is named and has type ArrayLike, followed by other parameters, AND there is at least
# one more parameter (NOT the second param) that has ArrayLike in its type (even in a Union).
# This is for reduction and aggregation functions like sum, mean, max, etc. that should
# preserve ArrayValue type when the input is ArrayValue.
#
# Matched patterns:
#     def func(a: ArrayLike, axis: int, initial: ArrayLike | None, ...) -> Array: ...
#     def func(x: ArrayLike, keepdims: bool, where: ArrayLike | None, ...) -> Array: ...\n#
# Explicitly NOT matched:
#     Functions already handled by RE_SINGLE_ARRAYLIKE_PARAM (single param, no other ArrayLike)
#     Functions already handled by RE_BINARY (first two params ArrayLike, no other ArrayLike)
#     Functions with first param ArrayLike but no other ArrayLike params (e.g., swapaxes)
#     Binary functions like add(x: ArrayLike, y: ArrayLike, ...) -> handled by RE_BINARY
#     Functions where second param contains ArrayLike (e.g., bincount(x: ArrayLike, weights: ArrayLike | None))
#
# This pattern requires at least one comma after the first ArrayLike parameter,
# the second parameter must NOT contain ArrayLike in its type annotation,
# AND there must be at least one more occurrence of ArrayLike after the second param.
RE_MULTI_PARAM_FIRST_ARRAYLIKE = re.compile(
    r"^def (?P<name>\w+)\(\s*(?P<param>\w+): ArrayLike,\s*(?P<param2>\w+):\s*(?P<param2type>(?:(?!ArrayLike)[^,)])+),(?P<tail>(?:(?!->).)*?ArrayLike(?:(?!->).)*?)\) -> Array: \.\.\.$",
    re.MULTILINE | re.DOTALL,
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
      def __call__(self, x: _ArrayValueT, y: _ArrayValueT, /) -> _ArrayValueT: ...
      @overload
      def __call__(self, x: _ArrayValueT, y: ArrayLike, /) -> _ArrayValueT: ...
      @overload
      def __call__(self, x: ArrayLike, y: _ArrayValueT, /) -> _ArrayValueT: ...
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


def _filter_arrayvalue_unions(type_with_arrayvalue: str, /) -> str:
    """Filter a type annotation to keep only _ArrayValueT-containing union members.

    Given a type annotation where ArrayLike has been replaced with _ArrayValueT,
    this function filters union members to keep only those containing _ArrayValueT.

    For example:
        "_ArrayValueT | DuckTypedArray | Sequence[_ArrayValueT]"
        becomes "_ArrayValueT | Sequence[_ArrayValueT]"

        "Array | _np.ndarray | Sequence[_ArrayValueT]"
        becomes "Sequence[_ArrayValueT]"

        "_ArrayValueT | None | int" becomes "_ArrayValueT"

        "_ArrayValueT" (no union) stays "_ArrayValueT"

    Args:
        type_with_arrayvalue: Type annotation with _ArrayValueT (and possibly unions).

    Returns
    -------
        The filtered type annotation with only _ArrayValueT-containing union members.

    """
    type_str = type_with_arrayvalue.strip()

    # Check if there's a union (contains |)
    if " | " not in type_str:
        return type_with_arrayvalue

    # Split by | and filter to keep only types containing _ArrayValueT
    union_types = [t.strip() for t in type_str.split("|")]
    filtered_types = [t for t in union_types if "_ArrayValueT" in t]

    if not filtered_types:
        # No _ArrayValueT types found - return original
        return type_with_arrayvalue

    # Reconstruct union with only _ArrayValueT-containing types
    return " | ".join(filtered_types)


def _add_typealias_imports(text: str, /) -> str:
    """Add quax imports and type variables for ArrayValue overloads.

    This function performs textual substitutions on the upstream `jax.numpy` stub:

    * Injects an import of ``quax`` alongside the existing ``os`` import.
    * Extends the generic type variables with ``_ArrayValueT``, which is
      bounded by ``quax.ArrayValue`` and is used to express value-preserving
      overloads in function signatures.

    Critically, this does NOT widen the ``ArrayLike`` type alias. The original
    JAX ``ArrayLike`` remains unchanged. The overloads with ``_ArrayValueT``
    provide type preservation for ``ArrayValue`` subclasses, while the base
    ``ArrayLike`` overload remains for standard JAX array-like types.

    The transformation assumes the upstream stub follows the structure
    produced by JAX's stub generation scripts (in particular, that the
    import blocks and ``_T`` definition appear exactly once).

    Args:
        text: The complete contents of the upstream ``jax.numpy`` stub file.

    Returns
    -------
        The rewritten stub text with quax imports and the ``_ArrayValueT``
        type variable, but with ``ArrayLike`` unchanged.

    """
    text = text.replace("import os\n", "import os\nimport quax\n", 1)
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


def _rewrite_single_arraylike_param(text: str, /) -> str:
    """Add overloads for functions with a single ArrayLike parameter.

    This function searches for function definitions where the first parameter
    has type ArrayLike, regardless of the parameter name or whether it uses
    positional-only syntax (/).

    Examples of matched patterns:
        def sum(a: ArrayLike, ...) -> Array: ...
        def flip(m: ArrayLike, /) -> Array: ...
        def angle(z: ArrayLike, /, deg: bool = ...) -> Array: ...
        def fix(x: ArrayLike, out: None = ...) -> Array: ...
        def i0(x: ArrayLike) -> Array: ...
        def fliplr(m: ArrayLike) -> Array: ...

    For each match, it adds an overload that takes _ArrayValueT and returns
    _ArrayValueT, preserving the type for ArrayValue subclasses.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with overloads added for single ArrayLike parameter functions.

    """

    def repl(match: re.Match[str], /) -> str:
        name = match.group("name")
        param = match.group("param")
        tail = match.group("tail")  # Everything after ArrayLike

        # Replace all ArrayLike occurrences in tail with _ArrayValueT
        tail_with_arrayvalue = tail.replace("ArrayLike", "_ArrayValueT")

        # Check if the parameter has a union type (e.g., " | None | int, ...")
        # tail starts right after "ArrayLike" in the parameter's type
        param_union_match = re.match(r"^\s*\|(.+?)(?=\s*[,)])", tail_with_arrayvalue)

        if param_union_match:
            # Parameter has a union - filter to keep only _ArrayValueT-containing types
            union_with_arrayvalue = "_ArrayValueT | " + param_union_match.group(1)
            filtered = _filter_arrayvalue_unions(union_with_arrayvalue)
            # Replace the union in tail with the filtered version
            tail_filtered = re.sub(
                r"^\s*\|.+?(?=\s*[,)])",
                ""
                if filtered == "_ArrayValueT"
                else " | " + filtered.split(" | ", 1)[1],
                tail_with_arrayvalue,
                count=1,
            )
        else:
            # No union - use tail as-is
            tail_filtered = tail_with_arrayvalue

        return "\n".join(
            (
                "@overload",
                f"def {name}({param}: _ArrayValueT{tail_filtered}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param}: ArrayLike{tail}) -> Array: ...",
            )
        )

    return RE_SINGLE_ARRAYLIKE_PARAM.sub(repl, text)


def _rewrite_multi_param_first_arraylike(text: str, /) -> str:
    """Add overload for multi-parameter functions with first param as ArrayLike.

    This function searches for function definitions where the first parameter
    has type ArrayLike and is followed by other parameters (reduction functions,
    aggregation functions, etc.). It adds an overload that preserves _ArrayValueT
    when the first argument is an _ArrayValueT.

    Examples of matched patterns:
        def sum(a: ArrayLike, axis: ..., ...) -> Array: ...
        def mean(a: ArrayLike, dtype: ..., ...) -> Array: ...

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with overloads added.

    """

    def repl(match: re.Match[str], /) -> str:
        name = match.group("name")
        param = match.group("param")
        param2 = match.group("param2")
        param2type = match.group("param2type")
        tail = match.group("tail")
        return "\n".join(
            (
                "@overload",
                f"def {name}({param}: _ArrayValueT, {param2}: {param2type},{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param}: ArrayLike, {param2}: {param2type},{tail}) -> Array: ...",
            )
        )

    return RE_MULTI_PARAM_FIRST_ARRAYLIKE.sub(repl, text)


def _rewrite_binary(text: str, /) -> str:
    """Rewrite binary function stubs to preserve ArrayValue where possible.

    This function searches for binary function definitions where the first
    two parameters have type ArrayLike and no other parameter contains ArrayLike::

        def name(x: ArrayLike, y: ArrayLike, ...) -> Array: ...

    using ``RE_BINARY`` and replaces each match with four overloads:

    * ``(x: ArrayLike, y: ArrayLike) -> Array`` (base case)
    * ``(x: _ArrayValueT, y: ArrayLike) -> _ArrayValueT``
    * ``(x: ArrayLike, y: _ArrayValueT) -> _ArrayValueT``
    * ``(x: _ArrayValueT, y: _ArrayValueT) -> _ArrayValueT``

    This models binary funcs whose result stays in the ``ArrayValue`` type
    whenever at least one operand is an ``_ArrayValueT`` instance, while
    retaining the upstream ``Array`` return type for purely ``ArrayLike``
    inputs. The base case must come first.

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
        name = match.group("name")
        param1 = match.group("param1")
        param2 = match.group("param2")
        param2type = match.group("param2type")  # Full type of param2, may have unions
        tail = match.group("tail")

        # Replace ArrayLike with _ArrayValueT and filter unions for param2
        param2type_with_arrayvalue = param2type.replace("ArrayLike", "_ArrayValueT")
        param2type_filtered = _filter_arrayvalue_unions(param2type_with_arrayvalue)

        return "\n".join(
            (
                "@overload",
                f"def {name}({param1}: _ArrayValueT, {param2}: {param2type_filtered}"
                f"{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: _ArrayValueT, {param2}: {param2type}{tail}) "
                "-> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: ArrayLike, {param2}: {param2type_filtered}"
                f"{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: ArrayLike, {param2}: {param2type}{tail}) -> Array: ...",
            )
        )

    return RE_BINARY.sub(repl, text)


def _rewrite_binary_with_union_first(text: str, /) -> str:
    """Rewrite binary functions where first param has union with ArrayLike.

    This handles functions like ``full_like(a: ArrayLike | DuckTypedArray,
    fill_value: ArrayLike, ...) -> Array``. For each such function, we generate
    four overload signatures that preserve or widen the ``_ArrayValueT`` type.

    Args:
        text: The stub text in which binary function signatures should be
            rewritten.

    Returns
    -------
        The stub text with matching binary function definitions replaced by
        overload-based signatures.

    """

    def repl(match: re.Match[str], /) -> str:
        name = match.group("name")
        param1 = match.group("param1")
        param1type = match.group("param1type")
        param2 = match.group("param2")
        param2type = match.group("param2type")
        tail = match.group("tail")

        # Replace ArrayLike with _ArrayValueT and filter unions for both params
        param1type_with_arrayvalue = param1type.replace("ArrayLike", "_ArrayValueT")
        param1type_filtered = _filter_arrayvalue_unions(param1type_with_arrayvalue)

        param2type_with_arrayvalue = param2type.replace("ArrayLike", "_ArrayValueT")
        param2type_filtered = _filter_arrayvalue_unions(param2type_with_arrayvalue)

        return "\n".join(
            (
                "@overload",
                f"def {name}({param1}: {param1type_filtered}, {param2}: "
                f"{param2type_filtered}{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: {param1type_filtered}, {param2}: "
                f"{param2type}{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: {param1type}, {param2}: "
                f"{param2type_filtered}{tail}) -> _ArrayValueT: ...",
                "@overload",
                f"def {name}({param1}: {param1type}, {param2}: {param2type}{tail}) "
                "-> Array: ...",
            )
        )

    return RE_BINARY_WITH_UNION_FIRST.sub(repl, text)


def _add_type_ignores(text: str, /) -> str:
    """Add type: ignore comments for upstream numpy stub issues.

    This function adds ``# type: ignore[type-arg]`` comments to lines containing
    generic types without type parameters (Callable, ndarray, DType). These are
    issues from the upstream numpy stubs that we don't introduce, so we suppress
    them to focus on actual issues in our transformations.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with type ignore comments added.

    """
    # Match function/method signature lines with unparameterized generics
    # Pattern matches lines that:
    # 1. Start with 'def ' OR are indented continuation lines with parameters
    # 2. Contain unparameterized Callable, ndarray, or DType (not DTypeLike)
    # 3. Are part of a signature (contain -> or are parameter lines with commas)
    function_sig_with_unparameterized = re.compile(
        r"""
        (?:^\s*def\s+\w+\s*\(.*          # Function definition line
        |^\s+\w+:\s+.*                    # Indented parameter line
        )
        .*\b(?:Callable|ndarray|DType\b(?!Like))  # Unparameterized generic (not DTypeLike)
        (?!\[)                             # Not followed by [
        """,
        re.VERBOSE,
    )

    lines = text.split("\n")
    result = []

    for line in lines:
        # Add ignore to lines with unparameterized generics
        if function_sig_with_unparameterized.search(
            line
        ) and not line.strip().startswith("#"):
            result.append(_add_type_ignore(line, "type-arg"))
        else:
            result.append(line)

    return "\n".join(result)


def _add_varargs_annotations(text: str, /) -> str:
    """Add type annotations to varargs and kwargs that lack them.

    This function adds ``: Any`` annotations to any ``*param`` (varargs)
    and ``**param`` (kwargs) parameters that don't already have type
    annotations. This includes common patterns like ``*args``, ``**kwargs``,
    ``*operands``, etc.

    This only matches parameters in function signatures, not type unpacking
    expressions like ``tuple[*Ts]``.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with varargs annotations added.

    """
    # Match *vararg without annotation when preceded by ( or , (parameter context)
    # This avoids matching type unpacking like tuple[*Ts] since [ doesn't match [,(]
    text = re.sub(
        r"([,(]\s*)\*(\w+)(?!:)(?=\s*[,)])",
        r"\1*\2: Any",
        text,
    )

    # Match **kwarg without annotation when preceded by ( or , (parameter context)
    text = re.sub(
        r"([,(]\s*)\*\*(\w+)(?!:)(?=\s*[,)])",
        r"\1**\2: Any",
        text,
    )

    return text


def _add_positional_param_annotations(text: str, /) -> str:
    """Add type annotations to positional parameters that lack them.

    This function adds ``: Any`` annotations to any positional parameters
    (without * or **) that don't already have type annotations. Only applies
    to parameters within function definitions.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with positional parameter annotations added.

    """

    # Match function definitions and process parameters within them
    # We need to only process the parameter list, not the return type
    def process_function(match: re.Match[str]) -> str:
        before = match.group(1)  # Everything before the parameters
        params = match.group(2)  # The parameter list
        after = match.group(3)  # Everything after the parameters (return type, etc.)

        # Match regular parameters without annotation within the parameter list
        # Pattern: preceded by comma or open paren (with optional whitespace),
        # not preceded by *, identifier, NOT followed by colon (with optional
        # whitespace), followed by comma/paren/equals/slash
        params = re.sub(
            r"((?:^|[,\(])\s*)(?!\*+)(\w+)(?!\s*:)(?=\s*[,)/=])",
            r"\1\2: Any",
            params,
        )
        return before + params + after

    # Match function definitions: def name(params) -> returntype: ...
    # Group 1: def name(
    # Group 2: params (matches across newlines with DOTALL)
    # Group 3: ) -> returntype: ... (does NOT match across function boundaries)
    # The key is that group 3 must match ) followed by anything up to : ... on same logical line
    text = re.sub(
        r"^(def \w+\()(.*?)(\) -> .*?: \.\.\.)",
        process_function,
        text,
        flags=re.MULTILINE | re.DOTALL,
    )

    return text


def _add_specific_import_ignores(text: str, /) -> str:
    """Add type: ignore comments for specific non-exported attributes.

    This function adds ``# type: ignore[attr-defined]`` comments to import
    statements for NamedSharding and PartitionSpec from jax._src.sharding_impls,
    which don't explicitly export these attributes.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with import ignore comments added.

    """
    # Add ignore specifically for NamedSharding and PartitionSpec from jax._src.sharding_impls
    text = re.sub(
        r"^(from jax\._src\.sharding_impls import .*(?:NamedSharding|PartitionSpec).*)$",
        r"\1  # type: ignore[attr-defined]",
        text,
        flags=re.MULTILINE,
    )

    return text


def _add_device_type_ignores(text: str, /) -> str:
    """Add type: ignore comments for _Device usage in type annotations.

    This function adds ``# type: ignore[valid-type]`` comments to lines that
    use _Device in type annotations, as _Device is a variable and not a valid type.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with _Device type ignore comments added.

    """
    lines = text.split("\n")
    result = []

    for line in lines:
        # Add ignore to lines with _Device in type annotations (parameters or return types)
        if (
            "_Device" in line
            and not line.strip().startswith("#")
            and (":" in line or "->" in line)  # Type annotation
        ):
            result.append(_add_type_ignore(line, "valid-type"))
        else:
            result.append(line)

    return "\n".join(result)


def _add_missing_return_types(text: str, /) -> str:
    """Add '-> Any' return type to functions missing return type annotations.

    This function finds function definitions that don't have a return type
    annotation and adds '-> Any' to them. This suppresses mypy errors for
    functions in the upstream stub that are missing return types.

    Args:
        text: The stub text to process.

    Returns
    -------
        The stub text with return types added to functions missing them.

    """
    # Match function definitions without return type annotations
    # Pattern: def name(...): ... but NOT def name(...) -> something: ...
    # We need to be careful to match the closing paren and colon without a ->
    pattern = re.compile(r"^(def \w+\([^)]*\)):\s*\.\.\.$", re.MULTILINE)

    return pattern.sub(r"\1 -> Any: ...", text)


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
    text = _rewrite_single_arraylike_param(text)
    text = _rewrite_multi_param_first_arraylike(text)
    text = _rewrite_binary(text)
    text = _rewrite_binary_with_union_first(text)
    text = _add_varargs_annotations(text)
    text = _add_positional_param_annotations(text)
    text = _add_specific_import_ignores(text)
    text = _add_device_type_ignores(text)
    text = _add_missing_return_types(text)
    text = _add_type_ignores(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> int:
    """CLI entry point to generate the numpy stub file.

    Returns
    -------
        Exit code (0 for success).

    """
    # Determine the project root
    # __file__ is in src/quaxed/_tools/make_numpy_stub.py
    # so we need to go up 3 levels to get to project root
    root = Path(__file__).parent.parent.parent.parent
    stub_path = root / "src" / "quaxed" / "numpy" / "__init__.pyi"

    logger.info("Generating numpy stub: %s", stub_path)
    generate_numpy_stub(stub_path)
    logger.info("Successfully generated: %s", stub_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover


# ---------------------------------------------------------------------------
# Hatch build hook
# ---------------------------------------------------------------------------


class NumPyStubBuildHook(BuildHookInterface[Any]):
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
