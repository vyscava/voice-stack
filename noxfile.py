import nox

PYTHON_VERSIONS = ["3.10", "3.11"]  # expand if you like
STYLE_TARGETS = ("src", "tests", "scripts", "noxfile.py")

nox.options.reuse_existing_virtualenvs = True
nox.options.error_on_missing_interpreters = False
nox.options.sessions = ("lint", "tests")


def install_with_dev(session: nox.Session) -> None:
    """Install the project with dev dependencies enabled."""
    session.install("-e", ".[dev]")


@nox.session(python=PYTHON_VERSIONS)
def tests(session: nox.Session) -> None:
    """Run unit tests with pytest (and coverage if desired)."""
    install_with_dev(session)
    session.run("pytest", "-q", "--disable-warnings", "--maxfail=1")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    """Static type checking via MyPy."""
    install_with_dev(session)
    session.run("mypy", "src")


@nox.session(name="lint", python=False)
def lint(session: nox.Session) -> None:
    """Format check via Black and lint with Ruff (no fixes)."""
    session.run(
        "hatch",
        "run",
        "black",
        "--check",
        *STYLE_TARGETS,
        external=True,
    )
    session.run(
        "hatch",
        "run",
        "ruff",
        "check",
        *STYLE_TARGETS,
        external=True,
    )


@nox.session(name="fmt", python=False)
def fmt(session: nox.Session) -> None:
    """Format with Black and fix import ordering with Ruff."""
    session.run("hatch", "run", "black", *STYLE_TARGETS, external=True)
    session.run(
        "hatch",
        "run",
        "ruff",
        "check",
        "--select",
        "I",
        "--fix",
        *STYLE_TARGETS,
        external=True,
    )


@nox.session(python=PYTHON_VERSIONS)
def ci(session: nox.Session) -> None:
    """
    Convenience ‘all-in-one’ session for CI:
    format check (non-mutating), lint, typecheck, tests.
    """
    install_with_dev(session)
    session.run("black", "--check", *STYLE_TARGETS)
    session.run("ruff", "check", *STYLE_TARGETS)
    session.run("mypy", "src")
    session.run("pytest", "-q", "--disable-warnings", "--maxfail=1")
