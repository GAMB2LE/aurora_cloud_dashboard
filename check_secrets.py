#!/usr/bin/env python3
"""Fail CI when a likely credential is introduced into a tracked text file."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PRIVATE_KEY = re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----")
ASSIGNMENT = re.compile(
    r"(?ix)\b(?:api[_-]?key|secret|password|token|private[_-]?key)\b\s*[:=]\s*"
    r"[\"'](?!example|secret|token|redacted|changeme|your[_-]?|\$|\{|<)[a-z0-9_./+=-]{16,}[\"']"
)
TEXT_SUFFIXES = {".cfg", ".conf", ".env", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yml", ".yaml"}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    return [ROOT / value.decode("utf-8") for value in output.split(b"\0") if value]


def main() -> int:
    findings: list[str] = []
    for path in tracked_files():
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for expression in (PRIVATE_KEY, ASSIGNMENT):
            match = expression.search(content)
            if match:
                findings.append(f"{path.relative_to(ROOT)}: likely credential")
                break
    if findings:
        print("Potential credentials found:\n" + "\n".join(findings), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
