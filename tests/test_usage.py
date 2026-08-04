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
}


def _codex_rows(rows: list[dict]) -> str:
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_codex_usage_is_priced_per_tier_and_ordered(tmp_path: Path) -> None:
    cell = Cell("gpt56sol_max", 0, tmp_path / "cell")
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
    cell = Cell("opus5_max", 0, tmp_path / "cell")
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


def test_kimi_reports_no_usage(tmp_path: Path) -> None:
    cell = Cell("k3_max", 0, tmp_path / "cell")
    (cell.state_dir / "sessions").mkdir(parents=True)

    assert usages(cell, PRICES) == []


def test_an_unpriced_model_still_counts_its_tokens(tmp_path: Path) -> None:
    cell = Cell("gpt56sol_max", 0, tmp_path / "cell")
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
