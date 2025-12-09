"""Nox setup."""

import shutil
from pathlib import Path

import nox
from nox_uv import session

nox.needs_version = ">=2024.3.2"
nox.options.default_venv_backend = "uv"

DIR = Path(__file__).parent.resolve()


# =============================================================================
# Helpers


def _process_stub_flag(s: nox.Session, /) -> list[str]:
    """Process --remake-stubs flag and regenerate stubs if requested.

    Args:
        s: The nox session.

    Returns
    -------
        Filtered posargs with --remake-stubs flag removed.

    """
    if "--remake-stubs" in s.posargs:
        s.run("quaxed-make-stubs")
        return [arg for arg in s.posargs if arg != "--remake-stubs"]
    return list(s.posargs)


# =============================================================================
# Linting


@session(uv_groups=["lint", "build"], reuse_venv=True)
def lint(s: nox.Session, /) -> None:
    """Run the linter.

    Pass --remake-stubs to regenerate type stubs before type checking.
    """
    precommit(s)  # reuse pre-commit session
    pylint(s)  # reuse pylint session
    mypy_lint(s)  # reuse mypy session
    pyright_lint(s)  # reuse pyright session


@session(uv_groups=["lint"], reuse_venv=True)
def precommit(s: nox.Session, /) -> None:
    """Run pre-commit."""
    s.run("pre-commit", "run", "--all-files", *s.posargs)


@session(uv_groups=["lint"], reuse_venv=True)
def pylint(s: nox.Session, /) -> None:
    """Run PyLint."""
    s.run("pylint", "quaxed", *s.posargs)


@session(uv_groups=["lint", "build"], reuse_venv=True)
def mypy_lint(s: nox.Session, /) -> None:
    """Run MyPy.

    Pass --remake-stubs to regenerate type stubs before type checking.
    """
    posargs = _process_stub_flag(s)
    s.run("mypy", "src/quaxed", *posargs)


@session(uv_groups=["lint", "build"], reuse_venv=True)
def pyright_lint(s: nox.Session, /) -> None:
    """Run Pyright.

    Pass --remake-stubs to regenerate type stubs before type checking.
    """
    posargs = _process_stub_flag(s)
    s.run("pyright", *posargs)


# =============================================================================
# Testing


@session(uv_groups=["test", "build"], reuse_venv=True)
def test(s: nox.Session, /) -> None:
    """Run the tests with all optional dependencies.

    Pass --remake-stubs to regenerate type stubs before type checking.
    """
    pytest(s)  # reuse pytest session
    mypy_test(s)  # reuse mypy test session


@session(uv_groups=["test"], reuse_venv=True)
def pytest(s: nox.Session, /) -> None:
    """Run the unit and regular tests."""
    s.run("pytest", *s.posargs)


@session(uv_groups=["test", "build"], reuse_venv=True)
def mypy_test(s: nox.Session, /) -> None:
    """Run MyPy as a test.

    Pass --remake-stubs to regenerate type stubs before type checking.
    """
    posargs = _process_stub_flag(s)
    # Need to run it on src/quaxed as well to pull in the installed package
    s.run("mypy", "src/quaxed", "tests/static", *posargs)


# =============================================================================
# Documentation


@session(uv_groups=["docs"], reuse_venv=True)
def docs(s: nox.Session, /) -> None:
    """Build the docs. Pass "--serve" to serve. Pass "-b linkcheck" to check links."""
    s.run("mkdocs", "build", *s.posargs)


# =============================================================================
# Build


@session(uv_groups=["build"])
def build(s: nox.Session, /) -> None:
    """Build an SDist and wheel."""
    build_path = DIR.joinpath("build")
    if build_path.exists():
        shutil.rmtree(build_path)

    s.run("python", "-m", "build")
