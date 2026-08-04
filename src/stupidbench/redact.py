"""Taking back what a cell was given, before anyone else can read it.

This repository is public, and so is every artifact and cache a run leaves
behind. A cell is handed real credentials, and an agent is free to write
anything anywhere in its workspace — including, if it ever prints its own
environment, the token it was given. So nothing leaves a runner until the files
have been read through: what a CLI stores credentials in is deleted outright,
and every occurrence of a live secret, or of anything shaped like one, is
replaced.
"""

import os
import re
import shutil
from collections.abc import Iterable
from pathlib import Path

from stupidbench.cell import TOKEN_VARIABLES

#: Files and directories a CLI keeps credentials in. They are never wanted in
#: an artifact, whatever they happen to hold at the time.
CREDENTIAL_NAMES = (
    ".claude.json",
    ".credentials.json",
    "auth.json",
    "credentials",
    "oauth",
)

#: Shapes of credential that are worth removing even when they are not one of
#: ours: a token a CLI minted for itself is no better to publish.
PATTERNS = (
    rb"sk-ant-[A-Za-z0-9_-]{16,}",
    rb"sk-proj-[A-Za-z0-9_-]{16,}",
    rb"sk-[A-Za-z0-9]{32,}",
    rb"gh[pousr]_[A-Za-z0-9]{36,}",
)

PLACEHOLDER = b"[REDACTED]"


def secrets() -> list[str]:
    """The live credentials this process was given, longest first.

    Longest first so that a token which contains another is replaced whole.
    """
    found = {
        value
        for name in TOKEN_VARIABLES.values()
        if (value := os.environ.get(name, "").strip())
    }
    return sorted(found, key=len, reverse=True)


def redact(root: Path, values: Iterable[str]) -> tuple[int, int]:
    """Cleans everything under ``root``.

    Returns how many paths were deleted and how many files were rewritten.
    """
    patterns = [re.escape(value.encode()) for value in values if value]
    expression = re.compile(b"|".join(patterns + list(PATTERNS)))
    deleted = 0
    for path in sorted(root.rglob("*"), reverse=True):
        if path.name in CREDENTIAL_NAMES:
            shutil.rmtree(path, ignore_errors=True) if path.is_dir() else path.unlink()
            deleted += 1
    rewritten = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        if _rewrite(path, expression):
            rewritten += 1
    return deleted, rewritten


def _rewrite(path: Path, expression: re.Pattern[bytes]) -> bool:
    """Replaces every match in ``path``, a line at a time.

    A session log runs to gigabytes, so it is never held whole; a secret holds
    no newline, so no match spans the lines it is read in.
    """
    with path.open("rb") as file:
        if not any(expression.search(line) for line in file):
            return False
    temporary = path.with_name(f"{path.name}.redacted")
    with path.open("rb") as source, temporary.open("wb") as target:
        for line in source:
            target.write(expression.sub(PLACEHOLDER, line))
    temporary.replace(path)
    return True
