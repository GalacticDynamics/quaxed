"""Regenerate the `quaxed.numpy` type stub file with ArrayValue-aware overloads."""

import argparse
import re
import textwrap
from pathlib import Path

import jax.numpy as jnp

RE_UNARY = re.compile(
    (r"^def (?P<name>\w+)\(x: ArrayLike, /(?P<tail>[^)]*)\) -> Array: \.\.\.$"),
    re.MULTILINE,
)

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


def _add_typealias_imports(text: str) -> str:
    text = text.replace(
        "Protocol, TypeVar, Union, overload",
        "Protocol, TypeAlias, TypeVar, Union, overload",
        1,
    )

    # Add quax import
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


def _replace_binary_ufunc(text: str) -> str:
    pattern = re.compile(
        (
            r"class BinaryUfunc\(Protocol\):\n(?:  .+\n)+?  def outer"
            r"\(self, a: ArrayLike, b: ArrayLike, /\) -> Array: ...\n"
        ),
    )
    return pattern.sub(BINARY_UFUNC, text, count=1)


def _rewrite_unary(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        tail = match.group("tail")
        name = match.group("name")
        return (
            f"@overload\n"
            f"def {name}(x: _ArrayValueT, /{tail}) -> _ArrayValueT: ...\n"
            f"@overload\n"
            f"def {name}(x: ArrayLike, /{tail}) -> Array: ..."
        )

    return RE_UNARY.sub(repl, text)


def _rewrite_binary(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
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


def update_stub(output: Path) -> None:
    upstream = Path(jnp.__file__).with_suffix(".pyi")
    text = upstream.read_text()
    text = _add_typealias_imports(text)
    text = _replace_binary_ufunc(text)
    text = _rewrite_unary(text)
    text = _rewrite_binary(text)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate quaxed numpy stub with ArrayValue overloads",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/quaxed/numpy/__init__.pyi"),
        help="Path to write the generated stub",
    )
    args = parser.parse_args()
    update_stub(args.output)


if __name__ == "__main__":
    main()
