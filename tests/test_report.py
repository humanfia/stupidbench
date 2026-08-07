import json
from math import log10
from pathlib import Path

import pandas as pd

from stupidbench.cell import INIT_SCORE, Cell
from stupidbench.report import (
    HEIGHT,
    History,
    Point,
    frame,
    markdown,
    read_history,
    report,
    window,
)
from stupidbench.usage import Tier

PRICES = {"gpt-5.6-sol": [Tier(0, {"prompt": 1.0, "completion": 10.0})]}


def _cell(tmp_path: Path, flow: str, seed: int, scores: list[tuple[str, int]]) -> Cell:
    cell = Cell(flow, seed, tmp_path / flow / str(seed))
    cell.cell_dir.mkdir(parents=True)
    cell.events_path.write_text(
        "".join(
            json.dumps({"datetime": moment, "type": kind}) + "\n"
            for moment, kind in (
                ("2026-08-04T00:00:00+00:00", "start"),
                ("2026-08-04T02:00:00+00:00", "stop"),
            )
        ),
        encoding="utf-8",
    )
    cell.scores_path.write_text(
        "".join(
            json.dumps({"datetime": moment, "score": score}) + "\n"
            for moment, score in scores
        ),
        encoding="utf-8",
    )
    return cell


def test_history_keeps_only_what_beat_everything_before_it(tmp_path: Path) -> None:
    cell = _cell(
        tmp_path,
        "gpt56sol_max",
        0,
        [
            (
                "2026-08-04T00:30:00+00:00",
                INIT_SCORE * 2,
            ),  # a penalty, never an improvement
            ("2026-08-04T01:00:00+00:00", 100_000),
            ("2026-08-04T01:15:00+00:00", 120_000),  # worse than the best so far
            ("2026-08-04T01:30:00+00:00", 100_000),  # equal, so not an improvement
            ("2026-08-04T02:00:00+00:00", 90_000),
        ],
    )

    history = read_history(cell, PRICES)

    assert [point.score for point in history.points] == [INIT_SCORE, 100_000, 90_000]
    assert [round(point.hours, 2) for point in history.points] == [0.0, 1.0, 2.0]
    assert history.end.score == 90_000
    assert history.end.hours == 2.0
    assert history.state == "running"


def test_frame_puts_every_seed_on_one_grid_and_stops_at_its_own_end() -> None:
    long_seed = _history("f", 0, [(INIT_SCORE, 0.0), (900.0, 1.0), (800.0, 9.0)], 9.0)
    short_seed = _history("f", 1, [(INIT_SCORE, 0.0), (700.0, 4.0)], 5.0)

    curves = frame([long_seed, short_seed])
    hours = curves[curves["metric"] == "hours"]

    grid = sorted(set(hours["x"]))
    assert grid == [0.0, 1.0, 4.0, 5.0, 9.0]
    # The long seed holds its value between improvements...
    long_scores = hours[hours["seed"] == 0].set_index("x")["score"].to_dict()
    assert long_scores == {
        0.0: INIT_SCORE,
        1.0: 900.0,
        4.0: 900.0,
        5.0: 900.0,
        9.0: 800.0,
    }
    # ...and the short one is not extrapolated past where it stopped.
    short_scores = hours[hours["seed"] == 1].set_index("x")["score"].to_dict()
    assert short_scores == {0.0: INIT_SCORE, 1.0: INIT_SCORE, 4.0: 700.0, 5.0: 700.0}
    # Which is what lets a mean over seeds mean anything at all.
    assert hours.groupby("x")["score"].mean()[5.0] == (900.0 + 700.0) / 2


def test_report_draws_the_curves_and_writes_the_summary(tmp_path: Path) -> None:
    cells = tmp_path / "cells"
    _cell(cells, "gpt56sol_max", 0, [("2026-08-04T01:00:00+00:00", 100_000)])
    _cell(cells, "gpt56sol_max", 1, [("2026-08-04T01:00:00+00:00", 110_000)])
    out = tmp_path / "report"

    text = report(cells, out, PRICES)

    assert (out / "curves.png").stat().st_size > 0
    assert (out / "curves.csv").is_file()
    assert (out / "report.md").read_text() == text
    # A cell whose CLI kept no session says nothing about what it ran at.
    assert "| `gpt56sol_max` | — | 2 | 100,000 | 105,000 |" in text
    assert "curves.png" in text


def test_window_gives_the_late_hours_most_of_the_height() -> None:
    # The first minutes are worth two decades and the day after them a few
    # hundred cycles. Drawn to fit everything, the day is a line on the floor.
    rows = [
        {"flow": "f", "seed": 0, "metric": "hours", "x": x, "score": score}
        for x, score in ((0.1, INIT_SCORE), (0.5, 2_000), (6.0, 1_400), (24.0, 1_000))
    ]

    bottom, top = window(pd.DataFrame(rows))

    height = log10(top / bottom)
    assert log10(1_400 / 1_000) / height >= HEIGHT
    # What fell before the first quarter of the time runs off the top rather
    # than setting the scale for everything after it.
    assert top < INIT_SCORE
    assert bottom < 1_000


def test_window_leaves_the_axis_alone_when_nothing_improved() -> None:
    rows = [{"flow": "f", "seed": 0, "metric": "hours", "x": 1.0, "score": 900.0}]

    assert window(pd.DataFrame(rows)) is None


def test_report_says_so_when_nothing_has_run(tmp_path: Path) -> None:
    text = report(tmp_path / "cells", tmp_path / "report", PRICES)

    assert "No cell has run yet" in text
    assert (tmp_path / "report/report.md").is_file()


def test_markdown_reports_progress_against_the_budget() -> None:
    history = _history("f", 0, [(INIT_SCORE, 0.0), (900.0, 12.0)], 12.0)

    text = markdown([history], "curves.png")

    assert "| `f` | 0 | max | running (50%) | 900 |" in text


def test_markdown_reports_what_a_response_ran_at_and_what_it_wrote() -> None:
    # Twelve thousand output tokens over four responses, the largest of them
    # nine thousand and two of every three spent thinking, and two budgets
    # answered at where one was asked for.
    history = _history(
        "f",
        0,
        [(INIT_SCORE, 0.0), (900.0, 12.0)],
        12.0,
        juice=("high", "max"),
        responses=4,
        max_output=9_000,
        reasoning_tokens=8_000,
    )

    text = markdown([history], "curves.png")

    assert "| high/max (2,000) | running (50%) | 900 | 12.00 | 12,000 | 3,000 |" in text
    assert "| `f` | high/max (2,000) | 1 | 900 | 900 | 12.00 | 12,000 | 3,000 |" in text


def _history(
    flow: str,
    seed: int,
    points: list[tuple[float, float]],
    end: float,
    juice: tuple[str, ...] = ("max",),
    responses: int = 4,
    max_output: int = 9_000,
    reasoning_tokens: int = 0,
) -> History:
    return History(
        flow=flow,
        seed=seed,
        state="running",
        points=tuple(
            Point(score, hours, int(hours * 1000), hours) for score, hours in points
        ),
        end=Point(points[-1][0], end, int(end * 1000), end),
        juice=juice,
        responses=responses,
        max_output=max_output,
        reasoning_tokens=reasoning_tokens,
    )
