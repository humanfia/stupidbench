from pathlib import Path

import pytest

from stupidbench.redact import PLACEHOLDER, redact, secrets


def test_redact_removes_credential_files_and_every_live_token(tmp_path: Path) -> None:
    state = tmp_path / "cell/agent/.stupidbench/claude"
    state.mkdir(parents=True)
    (state / ".claude.json").write_text("secret-token", encoding="utf-8")
    (state / ".credentials.json").write_text("secret-token", encoding="utf-8")
    (tmp_path / "cell/agent/.stupidbench/kimi/oauth").mkdir(parents=True)
    (tmp_path / "cell/agent/.stupidbench/kimi/oauth/kimi").write_text("x")
    sessions = state / "projects"
    sessions.mkdir()
    # An agent that prints its own environment writes the token into its own log.
    (sessions / "session.jsonl").write_text(
        '{"text": "CODEX_ACCESS_TOKEN=secret-token"}\n{"text": "harmless"}\n',
        encoding="utf-8",
    )
    (tmp_path / "cell/agent/perf_takehome.py").write_text("# work\n", encoding="utf-8")

    deleted, rewritten = redact(tmp_path, ["secret-token"])

    assert deleted == 3
    assert rewritten == 1
    assert not (state / ".claude.json").exists()
    assert not (tmp_path / "cell/agent/.stupidbench/kimi/oauth").exists()
    text = (sessions / "session.jsonl").read_text()
    assert "secret-token" not in text
    assert PLACEHOLDER.decode() in text
    assert "harmless" in text
    assert (tmp_path / "cell/agent/perf_takehome.py").read_text() == "# work\n"


def test_redact_removes_the_copies_a_config_is_rewritten_through(
    tmp_path: Path,
) -> None:
    # The live file goes by name, and the dated copies beside it hold everything
    # it held under a name that match walks past — so the token in a copy would
    # have been published while the file it came from was deleted.
    state = tmp_path / "cell/agent/.stupidbench/claude"
    (state / "backups").mkdir(parents=True)
    (state / ".claude.json").write_text("secret-token", encoding="utf-8")
    for copy in (
        "backups/.claude.json.backup.1785899797426",
        "backups/.claude.json.corrupted.1785899797427",
    ):
        (state / copy).write_text("secret-token", encoding="utf-8")

    deleted, rewritten = redact(tmp_path, ["secret-token"])

    # All three went, so nothing was left holding a token to be rewritten.
    assert (deleted, rewritten) == (3, 0)
    assert not any(path.is_file() for path in state.rglob("*"))


def test_redact_removes_anything_shaped_like_a_key(tmp_path: Path) -> None:
    log = tmp_path / "agent.log"
    log.write_text(
        "key=sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAA rest\n"
        "gh=ghp_AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA\n",
        encoding="utf-8",
    )

    redact(tmp_path, [])

    text = log.read_text()
    assert "sk-ant" not in text
    assert "ghp_" not in text
    assert "rest" in text


def test_redact_is_idempotent_and_leaves_clean_trees_alone(tmp_path: Path) -> None:
    clean = tmp_path / "clean.jsonl"
    clean.write_text('{"score": 1234}\n', encoding="utf-8")

    assert redact(tmp_path, ["secret-token"]) == (0, 0)

    dirty = tmp_path / "dirty.jsonl"
    dirty.write_text("secret-token\n", encoding="utf-8")
    assert redact(tmp_path, ["secret-token"]) == (0, 1)
    assert redact(tmp_path, ["secret-token"]) == (0, 0)
    assert clean.read_text() == '{"score": 1234}\n'


def test_redact_handles_a_file_too_large_to_hold(tmp_path: Path) -> None:
    # A session log runs to gigabytes; the token sits on one line deep inside.
    big = tmp_path / "big.jsonl"
    with big.open("w", encoding="utf-8") as file:
        for index in range(20_000):
            file.write(f'{{"row": {index}, "text": "padding padding padding"}}\n')
        file.write('{"text": "secret-token"}\n')

    assert redact(tmp_path, ["secret-token"]) == (0, 1)
    assert "secret-token" not in big.read_text()
    assert '"row": 19999' in big.read_text()


def test_secrets_are_read_from_the_environment_longest_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "short")
    monkeypatch.setenv("CODEX_ACCESS_TOKEN", "a-much-longer-token")
    monkeypatch.setenv("KIMI_MODEL_API_KEY", "   ")

    assert secrets() == ["a-much-longer-token", "short"]


def test_redact_removes_a_repository_a_token_could_be_committed_into(
    tmp_path: Path,
) -> None:
    # A token deflated inside a git object matches no search of the text, so
    # the container goes rather than being searched.
    objects = tmp_path / "cell/agent/.git/objects/ab"
    objects.mkdir(parents=True)
    (objects / "cdef").write_bytes(b"\x78\x9c\x00compressed-secret-token")
    (tmp_path / "cell/agent/work.py").write_text("# work\n", encoding="utf-8")

    deleted, _ = redact(tmp_path, ["secret-token"])

    assert deleted == 1
    assert not (tmp_path / "cell/agent/.git").exists()
    assert (tmp_path / "cell/agent/work.py").is_file()


def test_redact_unlinks_a_symlinked_credential_store_without_following_it(
    tmp_path: Path,
) -> None:
    target = tmp_path / "elsewhere"
    target.mkdir()
    (target / "token").write_text("secret-token", encoding="utf-8")
    state = tmp_path / "cell/agent/.stupidbench/claude"
    state.mkdir(parents=True)
    (state / "credentials").symlink_to(target)

    deleted, _ = redact(tmp_path / "cell", ["secret-token"])

    # The link is gone and counted; what it pointed at was never walked into.
    assert deleted == 1
    assert not (state / "credentials").is_symlink()
    assert (target / "token").read_text() == "secret-token"
