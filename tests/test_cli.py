from pathlib import Path

import pytest

from stupidbench import cli, runner
from stupidbench.cell import Cell


def test_prepare_then_report_over_a_cell_that_never_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(cli, "pricing", dict)
    cells = tmp_path / "cells"

    assert (
        cli.main(["--cells", str(cells), "prepare", "--flow", "k3_max", "--seed", "0"])
        == 0
    )
    assert (
        cli.main(["--cells", str(cells), "report", "--out", str(tmp_path / "report")])
        == 0
    )

    assert "No cell has run yet" in capsys.readouterr().out


def test_report_fails_the_run_when_a_cell_is_short(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # The report is the last thing a run does and the only thing that says what
    # became of it. Every segment before it is free to fail, because the leg
    # after it takes the cell on; a result that is not there is what no leg can
    # recover, and it is the whole of what a red run should mean.
    monkeypatch.setattr(cli, "pricing", dict)
    cells = tmp_path / "cells"
    cell = Cell("opus5_max", 0, cells / "opus5_max" / "0")
    cell.cell_dir.mkdir(parents=True)
    cell.events_path.write_text(
        '{"datetime": "2026-08-04T00:00:00+00:00", "type": "start"}\n'
        '{"datetime": "2026-08-04T02:00:00+00:00", "type": "stop"}\n',
        encoding="utf-8",
    )
    cell.scores_path.write_text(
        '{"datetime": "2026-08-04T01:00:00+00:00", "score": 1405}\n', encoding="utf-8"
    )

    status = cli.main(
        ["--cells", str(cells), "report", "--out", str(tmp_path / "report")]
    )

    assert status == 1
    assert "2.0h of the 24h" in capsys.readouterr().err


def test_run_fails_when_the_cell_never_started(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # A cell the host cannot admit is waited on rather than refused, so a
    # segment that ran nothing must not be read as one that found nothing.
    monkeypatch.setattr(runner, "run", lambda cell, seconds: "ran")

    status = cli.main(
        [
            "--cells",
            str(tmp_path / "cells"),
            "run",
            "--flow",
            "opus5_max",
            "--seed",
            "0",
            "--seconds",
            "60",
        ]
    )

    assert status == 1
    assert "never started" in capsys.readouterr().err


def test_run_reports_what_the_segment_did(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    cells = tmp_path / "cells"
    cell = Cell("opus5_max", 0, cells / "opus5_max" / "0")

    def fake_run(cell_argument: Cell, seconds: int) -> str:
        cell_argument.record("start")
        cell_argument.events_path.write_text(
            '{"datetime": "2026-08-04T00:00:00+00:00", "type": "start"}\n'
            '{"datetime": "2026-08-04T01:00:00+00:00", "type": "stop"}\n',
            encoding="utf-8",
        )
        return "ran"

    monkeypatch.setattr(runner, "run", fake_run)
    runner.prepare(cell)

    status = cli.main(
        [
            "--cells",
            str(cells),
            "run",
            "--flow",
            "opus5_max",
            "--seed",
            "0",
            "--seconds",
            "60",
        ]
    )

    # An hour is what the slowest flow yet took to reach its first score, so a
    # segment that short with none of them is not yet evidence of anything.
    assert status == 0
    assert "1.00h of 24h spent" in capsys.readouterr().out


def test_run_fails_when_a_long_segment_recorded_no_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    # An agent that cannot run a single command spends the budget as quietly as
    # one that can: the run this comes from was green in all twelve jobs, one of
    # which had done nothing at all.
    cells = tmp_path / "cells"
    cell = Cell("opus5_max", 0, cells / "opus5_max" / "0")

    def fake_run(cell_argument: Cell, seconds: int) -> str:
        cell_argument.events_path.write_text(
            '{"datetime": "2026-08-04T00:00:00+00:00", "type": "start"}\n'
            '{"datetime": "2026-08-04T04:30:00+00:00", "type": "stop"}\n',
            encoding="utf-8",
        )
        return "ran"

    monkeypatch.setattr(runner, "run", fake_run)
    runner.prepare(cell)

    status = cli.main(
        [
            "--cells",
            str(cells),
            "run",
            "--flow",
            "opus5_max",
            "--seed",
            "0",
            "--seconds",
            "16200",
        ]
    )

    assert status == 1
    assert "recorded nothing" in capsys.readouterr().err


def test_run_passes_a_long_segment_that_scored(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    cells = tmp_path / "cells"
    cell = Cell("opus5_max", 0, cells / "opus5_max" / "0")

    def fake_run(cell_argument: Cell, seconds: int) -> str:
        cell_argument.events_path.write_text(
            '{"datetime": "2026-08-04T00:00:00+00:00", "type": "start"}\n'
            '{"datetime": "2026-08-04T04:30:00+00:00", "type": "stop"}\n',
            encoding="utf-8",
        )
        cell_argument.scores_path.write_text(
            '{"datetime": "2026-08-04T02:00:00+00:00", "score": 1405}\n',
            encoding="utf-8",
        )
        return "ran"

    monkeypatch.setattr(runner, "run", fake_run)
    runner.prepare(cell)

    assert (
        cli.main(
            [
                "--cells",
                str(cells),
                "run",
                "--flow",
                "opus5_max",
                "--seed",
                "0",
                "--seconds",
                "16200",
            ]
        )
        == 0
    )
