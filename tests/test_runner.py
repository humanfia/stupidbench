from pathlib import Path

import pytest

from stupidbench import runner
from stupidbench.cell import FLOWS, Cell


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_prepare_stages_a_runnable_cell(tmp_path: Path, flow: str) -> None:
    cell = Cell(flow, 0, tmp_path / flow / "0")

    runner.prepare(cell)

    for relative in (
        "perf_takehome.py",
        "problem.py",
        "tests/generate.py",
        "tests/submission_tests.py",
        ".stupidbench/task.md",
        ".stupidbench/run.sh",
    ):
        assert (cell.agent_dir / relative).is_file()
    for name in ("evaluator_server.py", "frozen_problem.py", "score.py"):
        assert (cell.evaluator_dir / name).is_file()
    assert cell.scores_path.is_file()
    assert cell.state == "pending"

    script = (cell.agent_dir / ".stupidbench/run.sh").read_text()
    assert f'export MODEL="{FLOWS[flow].model}"' in script
    assert 'export EFFORT="max"' in script
    # The loop is what makes it a bench: nothing ends it but the clock.
    assert "while true; do" in script
    assert "|| true" in script


def test_prepare_leaves_a_staged_cell_alone(tmp_path: Path) -> None:
    cell = Cell("opus5_max", 1, tmp_path / "cell")
    runner.prepare(cell)
    kernel = cell.agent_dir / "perf_takehome.py"
    kernel.write_text("# the agent's own work\n", encoding="utf-8")

    runner.prepare(cell)

    assert kernel.read_text() == "# the agent's own work\n"


def test_prepare_rewrites_the_run_script_of_a_cell_already_in_flight(
    tmp_path: Path,
) -> None:
    # A cell is carried for twenty-four hours over as many segments as that
    # takes, so a fix to what it runs has to reach the cell already running the
    # old one rather than only the next cell staged.
    cell = Cell("opus5_max", 1, tmp_path / "cell")
    runner.prepare(cell)
    cell.record("start")
    script = cell.agent_dir / ".stupidbench/run.sh"
    script.write_text("#!/bin/bash\n# what the segment before ran\n", encoding="utf-8")

    runner.prepare(cell)

    assert "while true; do" in script.read_text()


def test_prepare_starts_claude_without_the_settings_it_cached(tmp_path: Path) -> None:
    # The copy claude keeps of what its provider hands down turns off the mode
    # the cell runs in, and the segment that restored one had every command the
    # agent tried come back needing an approval nobody was there to give.
    cell = Cell("opus5_max", 0, tmp_path / "claude")

    runner.prepare(cell)

    script = (cell.agent_dir / ".stupidbench/run.sh").read_text()
    assert 'rm -f "$CLAUDE_CONFIG_DIR/remote-settings.json"' in script
    assert script.index("remote-settings.json") < script.index("claude --print")


@pytest.mark.parametrize("flow", sorted(FLOWS))
def test_a_turn_meets_the_task_with_nothing_but_what_it_wrote(
    tmp_path: Path, flow: str
) -> None:
    # The task is sent again every turn, and every turn is a session of its own:
    # nothing resumes what the turn before it was thinking, so a cell carries
    # what it wrote down and nothing else.
    cell = Cell(flow, 0, tmp_path / flow)

    runner.prepare(cell)

    script = (cell.agent_dir / ".stupidbench/run.sh").read_text()
    assert "--continue" not in script
    assert "resume" not in script


def test_prepare_denies_the_web_tools_a_cli_reaches_through_its_provider(
    tmp_path: Path,
) -> None:
    claude = Cell("opus5_max", 0, tmp_path / "claude")
    codex = Cell("gpt56sol_max", 0, tmp_path / "codex")
    runner.prepare(claude)
    runner.prepare(codex)

    assert "WebSearch" in (claude.state_dir / "settings.json").read_text()
    assert "WebFetch" in (claude.state_dir / "settings.json").read_text()
    config = (codex.state_dir / "config.toml").read_text()
    assert 'web_search = "disabled"' in config
    assert "web_search = false" in config


def test_agent_environment_gives_a_cell_only_its_own_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "claude-token")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "codex-token")
    monkeypatch.setenv("KIMI_MODEL_API_KEY", "kimi-token")

    environment = runner._agent_environment("codex", "gpt-5.6-sol", "max")

    assert environment["CODEX_ACCESS_TOKEN"] == "codex-token"
    assert "CLAUDE_CODE_OAUTH_TOKEN" not in environment
    assert "KIMI_MODEL_API_KEY" not in environment
    assert environment["CODEX_HOME"] == "/workspace/.stupidbench/codex"
    assert environment["HTTPS_PROXY"] == "http://proxy:8888"
    assert "evaluator" in environment["NO_PROXY"]


def test_agent_environment_configures_kimi_without_a_login(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KIMI_MODEL_API_KEY", "kimi-token")

    environment = runner._agent_environment("kimi", "kimi-code/k3", "max")

    # Naming the model in the environment is what makes it the default one; the
    # managed provider it would otherwise use needs an interactive login.
    assert environment["KIMI_MODEL_NAME"] == "k3"
    assert environment["KIMI_MODEL_API_KEY"] == "kimi-token"
    assert environment["KIMI_MODEL_THINKING_EFFORT"] == "max"
    # The default capabilities leave tool use out, and an agent needs it.
    assert "tool_use" in environment["KIMI_MODEL_CAPABILITIES"]


def test_agent_environment_omits_a_token_the_runner_was_not_given(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)

    assert "CLAUDE_CODE_OAUTH_TOKEN" not in runner._agent_environment(
        "claude", "claude-opus-5", "max"
    )


@pytest.mark.parametrize(
    "config", (runner._proxy_config("10.0.0.2"), runner._resolver_config())
)
def test_proxy_config_allows_the_providers_and_nothing_else(config: str) -> None:
    for host in runner.ALLOWED_HOSTS:
        assert host in config
    assert "deny *" in config
    # The allow rule is the only one, and it is HTTPS to 443 alone.
    allow = [line for line in config.splitlines() if line.startswith("allow")]
    assert len(allow) == 1
    assert allow[0].endswith("443 HTTPS")
    assert config.index(allow[0]) < config.index("deny *")


def test_proxy_config_denies_a_host_without_ever_looking_it_up() -> None:
    # 3proxy resolves a target before it applies its rules, so a cell that only
    # refused the connection would already have sent the name it was given to a
    # nameserver: the answer is denied, the question is the exfiltration.
    config = runner._proxy_config("10.0.0.2")

    assert "fakeresolve" in config
    assert "nserver" not in config
    # What the ACL did let through is handed on as a name, to the one half of
    # the proxy that resolves and that the agent's network cannot reach.
    assert f"parent 1000 http 10.0.0.2 {runner.RESOLVER_PORT}" in config
    assert "nserver 127.0.0.11" in runner._resolver_config()


def test_run_refuses_a_cell_that_was_never_staged(tmp_path: Path) -> None:
    # Docker would create the missing mount point as root and leave the cell
    # unusable, so this must fail before any container is made.
    cell = Cell("opus5_max", 0, tmp_path / "never-staged")

    with pytest.raises(RuntimeError, match="has not been staged"):
        runner.run(cell, 60)


def test_a_segment_that_never_started_records_no_stop(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # An unpaired stop would charge the cell for every hour since the segment
    # before it, and enough of those would retire the cell with budget unspent.
    cell = Cell("opus5_max", 0, tmp_path / "cell")
    runner.prepare(cell)
    cell.record("start")
    cell.record("stop")

    class _Failing:
        def __getattr__(self, _name: str) -> object:
            raise RuntimeError("no docker here")

    monkeypatch.setattr(runner.docker, "from_env", lambda: _Failing())
    with pytest.raises(RuntimeError):
        runner.run(cell, 60)

    assert [event.type for event in cell.events()] == ["start", "stop"]
