"""Generate an instruction stream and submit it to the isolated evaluator."""

from __future__ import annotations

import json
import subprocess
import sys
import urllib.request
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
WORK_DIR = TESTS_DIR.parent
_BASELINE = 147734
_EVALUATOR_URL = "http://evaluator/score"
_REQUEST_TIMEOUT_SECONDS = 130


def main() -> int:
    instructions_path = WORK_DIR / "instrs.json"
    instructions_path.unlink(missing_ok=True)
    subprocess.run([sys.executable, str(TESTS_DIR / "generate.py")], check=False)
    try:
        instructions = instructions_path.read_bytes()
    except OSError:
        instructions = b"null"

    request = urllib.request.Request(
        _EVALUATOR_URL,
        data=instructions,
        method="POST",
    )
    request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=_REQUEST_TIMEOUT_SECONDS) as response:
        payload = json.loads(response.read())
    if not isinstance(payload, dict):
        raise ValueError("evaluator response must be an object")
    ok = payload.get("ok")
    score = payload.get("score")
    if not isinstance(ok, bool) or type(score) is not int or score <= 0:
        raise ValueError("evaluator returned an invalid score result")

    print("CYCLES: ", score)
    if ok:
        print("Speedup over baseline: ", _BASELINE / score)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
