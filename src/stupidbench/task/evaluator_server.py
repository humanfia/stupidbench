"""HTTP boundary for evaluating untrusted instruction streams."""

from __future__ import annotations

import datetime
import json
import os
import resource
import secrets
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_WORK_DIR = Path(__file__).resolve().parent
_SCORES_PATH = _WORK_DIR / "scores.jsonl"
_TEMP_DIR = Path(tempfile.gettempdir())
_MAX_BODY_BYTES = 8 * 1024 * 1024
_BODY_READ_TIMEOUT_SECONDS = 5
_SCORER_MEMORY_BYTES = 384 * 1024 * 1024
_SCORE_TIMEOUT_SECONDS = 125
_PENALTY = 147734 * 2
_SCORER_SEED = secrets.token_hex(32)


@dataclass(frozen=True)
class _Response:
    ok: bool
    score: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class _ScoreRecord:
    datetime: str
    score: int


def _history() -> list[int]:
    try:
        lines = _SCORES_PATH.read_text(encoding="utf-8").splitlines()
    except OSError:
        lines = []
    return [json.loads(line)["score"] for line in lines if line]


class _Handler(BaseHTTPRequestHandler):
    def _send(self, status: HTTPStatus, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _respond(self, status: HTTPStatus, response: _Response) -> None:
        if response.score is not None:
            _SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
            record = _ScoreRecord(
                datetime=datetime.datetime.now(datetime.UTC).isoformat(),
                score=response.score,
            )
            with _SCORES_PATH.open("a", encoding="utf-8") as file:
                file.write(json.dumps(asdict(record)) + "\n")
        self._send(status, asdict(response))

    def do_GET(self) -> None:
        if self.path != "/scores":
            self._send(
                HTTPStatus.NOT_FOUND,
                asdict(_Response(ok=False, error="not found")),
            )
            return
        self._send(HTTPStatus.OK, _history())

    def do_POST(self) -> None:
        if self.path != "/score":
            self._respond(
                HTTPStatus.NOT_FOUND,
                _Response(ok=False, error="not found"),
            )
            return

        length_headers = self.headers.get_all("Content-Length")
        if length_headers is None:
            self._respond(
                HTTPStatus.LENGTH_REQUIRED,
                _Response(ok=False, score=_PENALTY, error="content length required"),
            )
            return
        if len(length_headers) != 1:
            self._respond(
                HTTPStatus.BAD_REQUEST,
                _Response(ok=False, score=_PENALTY, error="duplicate content length"),
            )
            return
        try:
            content_length = int(length_headers[0])
        except ValueError:
            content_length = -1
        if content_length < 0:
            self._respond(
                HTTPStatus.BAD_REQUEST,
                _Response(ok=False, score=_PENALTY, error="invalid content length"),
            )
            return
        if content_length > _MAX_BODY_BYTES:
            self._respond(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                _Response(ok=False, score=_PENALTY, error="request body too large"),
            )
            return

        self.connection.settimeout(_BODY_READ_TIMEOUT_SECONDS)
        try:
            body = self.rfile.read(content_length)
        except TimeoutError:
            self._respond(
                HTTPStatus.REQUEST_TIMEOUT,
                _Response(ok=False, score=_PENALTY, error="request body timed out"),
            )
            return
        if len(body) != content_length:
            self._respond(
                HTTPStatus.BAD_REQUEST,
                _Response(ok=False, score=_PENALTY, error="incomplete request body"),
            )
            return

        with tempfile.NamedTemporaryFile(
            prefix="aopt-submission-",
            suffix=".json",
            delete=False,
            dir=_TEMP_DIR,
        ) as submission:
            submission.write(body)
            submission_path = Path(submission.name)
        try:
            try:
                completed = subprocess.run(
                    [sys.executable, str(_WORK_DIR / "score.py"), str(submission_path)],
                    cwd=_WORK_DIR,
                    capture_output=True,
                    text=True,
                    timeout=_SCORE_TIMEOUT_SECONDS,
                    check=False,
                    preexec_fn=lambda: resource.setrlimit(
                        resource.RLIMIT_AS,
                        (_SCORER_MEMORY_BYTES, _SCORER_MEMORY_BYTES),
                    ),
                    env=os.environ | {"_STUPIDBENCH_AOPT_SEED": _SCORER_SEED},
                )
            except subprocess.TimeoutExpired:
                self._respond(
                    HTTPStatus.OK,
                    _Response(ok=False, score=_PENALTY, error="scorer timed out"),
                )
                return
        finally:
            submission_path.unlink(missing_ok=True)

        try:
            raw_result = json.loads(completed.stdout)
            if not isinstance(raw_result, dict):
                raise TypeError
            ok = raw_result.get("ok")
            score = raw_result.get("score")
            valid_status = (ok is True and completed.returncode == 0) or (
                ok is False and score == _PENALTY and completed.returncode == 1
            )
            if (
                not isinstance(ok, bool)
                or type(score) is not int
                or score <= 0
                or not valid_status
            ):
                raise ValueError
        except (json.JSONDecodeError, TypeError, ValueError):
            self._respond(
                HTTPStatus.OK,
                _Response(ok=False, score=_PENALTY, error="invalid scorer result"),
            )
            return
        assert isinstance(ok, bool) and type(score) is int
        self._respond(HTTPStatus.OK, _Response(ok=ok, score=score))


def main() -> None:
    HTTPServer(("0.0.0.0", 80), _Handler).serve_forever()


if __name__ == "__main__":
    main()
