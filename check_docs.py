#!/usr/bin/env python3
"""Build the local MkDocs site in an isolated docs-check environment."""

from __future__ import annotations

import subprocess
import sys
import venv
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


def _runs(command: list[str]) -> bool:
    print("+", " ".join(command))
    return subprocess.run(command, cwd=ROOT).returncode == 0


def main() -> None:
    if not DOCS_ENV.exists():
        venv.EnvBuilder(with_pip=True).create(DOCS_ENV)
    py = _python()
    if not _runs([str(py), "-m", "mkdocs", "--version"]):
        _run([str(py), "-m", "pip", "install", "--upgrade", "pip"])
        _run([str(py), "-m", "pip", "install", "-r", "requirements-docs.txt"])
    _run([str(py), "-m", "mkdocs", "build", "--strict"])


if __name__ == "__main__":
    main()
