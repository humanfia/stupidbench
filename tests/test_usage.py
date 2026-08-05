import json
from pathlib import Path

from stupidbench.cell import Cell
from stupidbench.usage import Tier, cumulative, usages

PRICES = {
    "gpt-5.6-sol": [
        Tier(400_000, {"prompt": 2.0, "completion": 20.0, "input_cache_read": 1.0}),
        Tier(0, {"prompt": 1.0, "completion": 10.0, "input_cache_read": 0.5}),
    ],
    "claude-opus-5": [
        Tier(
            0,
            {
                "prompt": 3.0,
                "completion": 30.0,
                "input_cache_read": 0.3,
                "input_cache_write": 3.75,
            },
        )
    ],
    # What OpenRouter calls the model Kimi Code names `kimi-code/k3`.
    "kimi-k3": [Tier(0, {"prompt": 3.0, "completion": 15.0, "input_cache_read": 0.3})],
}


def _codex_rows(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_codex_usage_is_priced_per_tier_and_ordered(tmp_path: Path) -> None:
    cell = Cell("gpt56sol_max_ralph", 0, tmp_path / "cell")
    sessions = cell.state_dir / "sessions" / "2026" / "08"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text(
        _codex_rows(
            [
                {"type": "turn_context", "payload": {"model": "gpt-5.6-sol"}},
                {
                    "timestamp": "2026-08-04T00:00:10+00:00",
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "last_token_usage": {
                                "input_tokens": 100,
                                "cached_input_tokens": 40,
                                "output_tokens": 7,
                            }
                        },
                    },
                },
                {
                    "timestamp": "2026-08-04T00:00:20+00:00",
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {
                            "last_token_usage": {
                                "input_tokens": 500_000,
                                "output_tokens": 3,
                            }
                        },
                    },
                },
                # Neither of these is an accounted exchange.
                {"timestamp": "2026-08-04T00:00:30+00:00", "type": "response_item"},
                {
                    "timestamp": "2026-08-04T00:00:40+00:00",
                    "type": "event_msg",
                    "payload": {"type": "token_count", "info": None},
                },
            ]
        ),
        encoding="utf-8",
    )

    spent = usages(cell, PRICES)

    assert [usage.output_tokens for usage in spent] == [7, 3]
    # 60 uncached at 1.0, 40 cached at 0.5, 7 out at 10.0.
    assert spent[0].cost == 60 * 1.0 + 40 * 0.5 + 7 * 10.0
    # Past 400k the long-context tier applies.
    assert spent[1].cost == 500_000 * 2.0 + 3 * 20.0


def test_claude_usage_counts_each_message_once(tmp_path: Path) -> None:
    cell = Cell("opus5_max_ralph", 0, tmp_path / "cell")
    projects = cell.state_dir / "projects" / "-workspace"
    projects.mkdir(parents=True)
    message = {
        "timestamp": "2026-08-04T00:00:10+00:00",
        "message": {
            "id": "msg_1",
            "model": "claude-opus-5-20260101",
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 100,
                "cache_creation_input_tokens": 20,
                "output_tokens": 5,
            },
        },
    }
    (projects / "session.jsonl").write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                message,
                message,  # the same message, seen again in a resumed session
                {
                    "timestamp": "2026-08-04T00:00:20+00:00",
                    "message": {
                        "id": "msg_2",
                        "model": "<synthetic>",
                        "usage": {"output_tokens": 9},
                    },
                },
            )
        ),
        encoding="utf-8",
    )

    spent = usages(cell, PRICES)

    assert len(spent) == 1
    assert spent[0].cost == 10 * 3.0 + 100 * 0.3 + 20 * 3.75 + 5 * 30.0


def test_kimi_usage_is_read_off_the_wire_and_priced(tmp_path: Path) -> None:
    # Kimi Code writes no usage into its session; the count is in the wire log,
    # in milliseconds, against a model name the environment stood in for.
    cell = Cell("k3_max_ralph", 0, tmp_path / "cell")
    wire = cell.state_dir / "sessions/wd_workspace_ab/session_cd/agents/main"
    wire.mkdir(parents=True)
    (wire / "wire.jsonl").write_text(
        _codex_rows(
            [
                {
                    "type": "usage.record",
                    "model": "__kimi_env_model__",
                    "usage": {
                        "inputOther": 2208,
                        "output": 145,
                        "inputCacheRead": 18688,
                        "inputCacheCreation": 0,
                    },
                    "usageScope": "turn",
                    "time": 1785892010229,
                },
                # A wider scope is the same tokens summed again.
                {
                    "type": "usage.record",
                    "model": "__kimi_env_model__",
                    "usage": {"inputOther": 99, "output": 99, "inputCacheRead": 0},
                    "usageScope": "session",
                    "time": 1785892010230,
                },
                {"type": "llm.request", "time": 1785892010231},
            ]
        ),
        encoding="utf-8",
    )

    spent = usages(cell, PRICES)

    assert len(spent) == 1
    assert spent[0].output_tokens == 145
    assert spent[0].timestamp == 1785892010.229
    assert spent[0].cost == 2208 * 3.0 + 18688 * 0.3 + 145 * 15.0


def test_kimi_counts_tokens_even_where_it_cannot_price_them(tmp_path: Path) -> None:
    cell = Cell("k3_max_ralph", 1, tmp_path / "cell")
    wire = cell.state_dir / "sessions/wd/session_x/agents/main"
    wire.mkdir(parents=True)
    (wire / "wire.jsonl").write_text(
        _codex_rows(
            [
                {
                    "type": "usage.record",
                    "usage": {"inputOther": 10, "output": 7},
                    "usageScope": "turn",
                    "time": 1785892010229,
                }
            ]
        ),
        encoding="utf-8",
    )

    spent = usages(cell, {})

    assert (spent[0].output_tokens, spent[0].cost) == (7, 0.0)


def test_an_unpriced_model_still_counts_its_tokens(tmp_path: Path) -> None:
    cell = Cell("gpt56sol_max_ralph", 0, tmp_path / "cell")
    sessions = cell.state_dir / "sessions"
    sessions.mkdir(parents=True)
    (sessions / "rollout.jsonl").write_text(
        _codex_rows(
            [
                {"type": "turn_context", "payload": {"model": "gpt-5.7-unreleased"}},
                {
                    "timestamp": "2026-08-04T00:00:10+00:00",
                    "type": "event_msg",
                    "payload": {
                        "type": "token_count",
                        "info": {"last_token_usage": {"output_tokens": 11}},
                    },
                },
            ]
        ),
        encoding="utf-8",
    )

    spent = usages(cell, PRICES)

    assert spent[0].output_tokens == 11
    assert spent[0].cost == 0.0


def test_cumulative_sums_up_to_a_moment() -> None:
    from stupidbench.usage import Usage

    spent = [Usage(10.0, 1, 1.0), Usage(20.0, 2, 2.0), Usage(30.0, 4, 4.0)]

    assert cumulative(spent, 5.0) == (0, 0.0)
    assert cumulative(spent, 20.0) == (3, 3.0)
    assert cumulative(spent, float("inf")) == (7, 7.0)
