import json
from pathlib import Path

from stupidbench.cell import BUDGET_SECONDS, Cell, Event, elapsed_at, timestamp


def write_events(cell: Cell, events: list[tuple[str, str]]) -> None:
    cell.events_path.parent.mkdir(parents=True, exist_ok=True)
    cell.events_path.write_text(
        "".join(
            json.dumps({"datetime": moment, "type": kind}) + "\n"
            for moment, kind in events
        ),
        encoding="utf-8",
    )


def test_elapsed_counts_only_the_time_a_segment_was_running() -> None:
    events = [
        Event("2026-08-04T00:00:00+00:00", "start"),
        Event("2026-08-04T00:10:00+00:00", "tick"),
        Event("2026-08-04T00:20:00+00:00", "stop"),
        # Six hours later, the next segment picks the cell up again.
        Event("2026-08-04T06:20:00+00:00", "start"),
        Event("2026-08-04T06:35:00+00:00", "stop"),
    ]

    assert elapsed_at(events, float("inf")) == 35 * 60
    # The wait between segments is not time the agent had.
    assert elapsed_at(events, timestamp("2026-08-04T06:20:00+00:00")) == 20 * 60


def test_elapsed_at_stops_at_the_moment_asked_about() -> None:
    events = [
        Event("2026-08-04T00:00:00+00:00", "start"),
        Event("2026-08-04T00:30:00+00:00", "stop"),
    ]
    assert elapsed_at(events, timestamp("2026-08-04T00:10:00+00:00")) == 600
    assert elapsed_at(events, timestamp("2026-08-03T00:00:00+00:00")) == 0


def test_state_follows_the_budget(tmp_path: Path) -> None:
    cell = Cell("opus5_max", 0, tmp_path / "cell")
    assert cell.state == "pending"

    write_events(
        cell,
        [("2026-08-04T00:00:00+00:00", "start"), ("2026-08-04T01:00:00+00:00", "stop")],
    )
    assert cell.state == "running"
    assert cell.elapsed == 3600

    write_events(
        cell,
        [("2026-08-04T00:00:00+00:00", "start"), ("2026-08-05T00:00:00+00:00", "stop")],
    )
    assert cell.elapsed == BUDGET_SECONDS
    assert cell.state == "done"


def test_records_events_and_reads_scores_in_order(tmp_path: Path) -> None:
    cell = Cell("k3_max", 2, tmp_path / "cell")
    cell.record("start")
    cell.record("stop")
    assert [event.type for event in cell.events()] == ["start", "stop"]

    cell.scores_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"datetime": "2026-08-04T00:00:20+00:00", "score": 900},
                {"datetime": "2026-08-04T00:00:10+00:00", "score": 1000},
                "not json",
            )
            if isinstance(row, dict)
        )
        + "not json\n",
        encoding="utf-8",
    )
    assert [score for _, score in cell.scores()] == [1000, 900]


def test_all_cells_cover_the_matrix(tmp_path: Path) -> None:
    cells = Cell.all(tmp_path)
    assert len(cells) == 15
    assert cells[0].cell_dir == tmp_path / cells[0].flow / str(cells[0].seed)
