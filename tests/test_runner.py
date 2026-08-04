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


def test_proxy_config_allows_the_providers_and_nothing_else() -> None:
    config = runner._proxy_config()

    for host in runner.ALLOWED_HOSTS:
        assert host in config
    assert "deny *" in config
    # The allow rule is the only one, and it is HTTPS to 443 alone.
    allow = [line for line in config.splitlines() if line.startswith("allow")]
    assert len(allow) == 1
    assert allow[0].endswith("443 HTTPS")
    assert config.index(allow[0]) < config.index("deny *")
