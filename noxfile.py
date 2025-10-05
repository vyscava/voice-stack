import nox


PYTHON_VERSIONS = ["3.10", "3.11"]  # expand if you like


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run unit tests with pytest (and coverage if desired)."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-q", "--disable-warnings", "--maxfail=1")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    """Static type checking via MyPy."""
    session.install("-e", ".[dev]")
    session.run("mypy")


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    """Lint with Ruff (no fixes)."""
    session.install("-e", ".[dev]")
    session.run("ruff", "check", ".")


@nox.session(python=PYTHON_VERSIONS)
def fmt(session: nox.Session) -> None:
    """Format with Black and fix import ordering with Ruff."""
    session.install("-e", ".[dev]")
    session.run("black", ".")
    session.run("ruff", "check", ".", "--select", "I", "--fix")


@nox.session(python=PYTHON_VERSIONS)
def ci(session: nox.Session) -> None:
    """
    Convenience ‘all-in-one’ session for CI:
    format check (non-mutating), lint, typecheck, tests.
    """
    session.install("-e", ".[dev]")
    # Check formatting without modifying files
    session.run("black", "--check", ".")
    session.run("ruff", "check", ".")
    session.run("mypy")
    session.run("pytest", "-q", "--disable-warnings", "--maxfail=1")
