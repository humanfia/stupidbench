"""What a cell is: where it lives, what it is running, and how much it has run.

A cell is one flow on one seed. It outlives the job that runs it — a hosted
runner is capped at six hours and a cell is budgeted at twenty-four — so every
question about it is answered from the files it leaves behind rather than from
anything a process holds.
"""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path
from typing import Literal

#: The agent CLIs a flow can run on, and where each keeps its state under the
#: agent's work dir. Redaction and usage accounting both read these back.
Tool = Literal["claude", "codex", "kimi"]

STATE_DIRS: dict[Tool, str] = {
    "claude": ".stupidbench/claude",
    "codex": ".stupidbench/codex",
    "kimi": ".stupidbench/kimi",
}

#: The environment variable each CLI reads its state directory from.
HOME_VARIABLES: dict[Tool, str] = {
    "claude": "CLAUDE_CONFIG_DIR",
    "codex": "CODEX_HOME",
    "kimi": "KIMI_CODE_HOME",
}

#: The token each CLI reads straight from its environment. A runner never logs
#: in, so this is the only credential a cell is ever given.
TOKEN_VARIABLES: dict[Tool, str] = {
    "claude": "CLAUDE_CODE_OAUTH_TOKEN",
    "codex": "CODEX_ACCESS_TOKEN",
    "kimi": "KIMI_MODEL_API_KEY",
}


@dataclass(frozen=True)
class Endpoint:
    """Where a model is served, when that is not the CLI's own provider.

    A CLI reaches the provider it was written for by itself. Pointed at another,
    it needs the host to let out, the address to send to, and a key of its own —
    and the key it would otherwise hold is no good there, so a cell that talks
    here is never given one.
    """

    host: str
    base_url: str
    token_variable: str


#: DeepSeek serves its models behind an Anthropic-shaped API, which is what lets
#: the claude CLI run one of them.
DEEPSEEK = Endpoint(
    "api.deepseek.com",
    "https://api.deepseek.com/anthropic",
    "DEEPSEEK_API_KEY",
)

#: Every variable a credential can arrive in: one per CLI, and one more for each
#: endpoint that is not the CLI's own provider. Redaction takes all of them back
#: out of a cell, whichever one it was given.
CREDENTIAL_VARIABLES = (*TOKEN_VARIABLES.values(), DEEPSEEK.token_variable)


@dataclass(frozen=True)
class Flow:
    """One column of the matrix: a CLI, a model, and how hard it is asked."""

    tool: Tool
    model: str
    effort: str
    #: Where the model is served, when the CLI's own provider is not where it
    #: is. Nothing else about the flow changes: the same CLI, the same loop.
    endpoint: Endpoint | None = None
    #: What the CLI is to take the model's context window to be, where it has no
    #: way of knowing. Claude Code assumes two hundred thousand tokens for a
    #: model it does not recognise, and compacts the session against that rather
    #: than against what the model really holds.
    context_window: int | None = None


FLOWS: dict[str, Flow] = {
    "gpt56sol_max": Flow("codex", "gpt-5.6-sol", "max"),
    "gpt56terra_max": Flow("codex", "gpt-5.6-terra", "max"),
    "gpt56luna_max": Flow("codex", "gpt-5.6-luna", "max"),
    "opus5_max": Flow("claude", "claude-opus-5", "max"),
    "dsv4flash_max": Flow("claude", "deepseek-v4-flash", "max", DEEPSEEK, 1_048_576),
    "k3_max": Flow("kimi", "kimi-code/k3", "max"),
}

SEEDS = (0, 1, 2)

#: Lower is better, and a cell starts where the unoptimized kernel scores.
INIT_SCORE = 147734

#: Agent time a cell is given, over as many segments as it takes.
BUDGET_SECONDS = 24 * 60 * 60

#: Agent time a segment may take without the evaluator recording anything
#: before that segment counts as a failure. A flow that works scores inside an
#: hour — the slowest yet took less than one, and spent it thinking — and a real
#: segment is four and a half, so this catches an agent that cannot work at all
#: without catching one that is only slow.
SCORELESS_SECONDS = 2 * 60 * 60

EventType = Literal["start", "tick", "stop"]


@dataclass(frozen=True)
class Event:
    datetime: str
    type: EventType


@dataclass(frozen=True)
class Cell:
    """One flow on one seed, addressed by where its files are."""

    flow: str
    seed: int
    cell_dir: Path

    @classmethod
    def all(cls, cells_dir: Path) -> list["Cell"]:
        return [
            cls(flow, seed, cells_dir / flow / str(seed))
            for flow in FLOWS
            for seed in SEEDS
        ]

    @property
    def agent_dir(self) -> Path:
        return self.cell_dir / "agent"

    @property
    def evaluator_dir(self) -> Path:
        return self.cell_dir / "evaluator"

    @property
    def scores_path(self) -> Path:
        return self.cell_dir / "scores.jsonl"

    @property
    def events_path(self) -> Path:
        return self.cell_dir / "events.jsonl"

    @property
    def state_dir(self) -> Path:
        """Where the CLI this cell runs keeps its sessions."""
        return self.agent_dir / STATE_DIRS[FLOWS[self.flow].tool]

    def record(self, type: EventType) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps({"datetime": _now(), "type": type}, sort_keys=True) + "\n"
            )

    def events(self) -> list[Event]:
        return [Event(**row) for row in read_jsonl(self.events_path)]

    def scores(self) -> list[tuple[float, int]]:
        """The (timestamp, score) pairs the evaluator recorded, in order."""
        scores = [
            (timestamp(row["datetime"]), int(row["score"]))
            for row in read_jsonl(self.scores_path)
        ]
        return sorted(scores)

    @property
    def elapsed(self) -> float:
        """Seconds of agent time spent, which is what the budget is kept in."""
        return elapsed_at(self.events(), float("inf"))

    @property
    def state(self) -> Literal["pending", "running", "done"]:
        events = self.events()
        if not events:
            return "pending"
        spent = elapsed_at(events, float("inf"))
        return "done" if spent >= BUDGET_SECONDS else "running"


def elapsed_at(events: list[Event], moment: float) -> float:
    """Agent time spent up to ``moment``.

    Only the gaps between an event and the one before it count, and only when
    the later one is not a start: a segment that ends and one that begins hours
    later are two segments, and the wait between them — a job queueing, an image
    pulling — is not time the agent had.
    """
    spent = 0.0
    for previous, current in pairwise(events):
        if current.type == "start":
            continue
        end = min(timestamp(current.datetime), moment)
        spent += max(0.0, end - timestamp(previous.datetime))
    return spent


def read_jsonl(path: Path) -> Iterator[dict]:
    """Yields each line parsed as JSON, skipping what does not parse."""
    if not path.is_file():
        return
    with path.open(encoding="utf-8") as file:
        for line in file:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def timestamp(value: str) -> float:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.timestamp()


def _now() -> str:
    return datetime.now(UTC).isoformat()
