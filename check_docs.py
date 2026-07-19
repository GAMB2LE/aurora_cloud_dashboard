#!/usr/bin/env python3
"""Build the local MkDocs site in an isolated docs-check environment."""

from __future__ import annotations

import subprocess
import sys
import venv
from shutil import rmtree
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DOCS_ENV = ROOT / ".venv-docs"


def _python() -> Path:
    if sys.platform == "win32":
        return DOCS_ENV / "Scripts" / "python.exe"
    return DOCS_ENV / "bin" / "python"


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _create_docs_environment() -> None:
    venv.EnvBuilder(with_pip=True).create(DOCS_ENV)


def _docs_environment_has_pip() -> bool:
    """Return whether the disposable environment can repair its packages."""
    python = _python()
    if not python.exists():
        return False
    return subprocess.run(
        [str(python), "-m", "pip", "--version"],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def main() -> None:
    if not _docs_environment_has_pip():
        # This environment is created only for this check and contains no user
        # data. Rebuilding it makes an interrupted or incompatible pip repairable.
        if DOCS_ENV.exists():
            rmtree(DOCS_ENV)
        _create_docs_environment()
    py = _python()
    _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
    _run([str(py), "-m", "pip", "install", "-r", "requirements-docs.txt"])
    _run([str(py), "-m", "mkdocs", "build", "--strict"])


if __name__ == "__main__":
    main()
