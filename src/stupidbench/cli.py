"""The four things a run does, one command each."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from stupidbench import redact as redaction
from stupidbench import report as reporting
from stupidbench import runner
from stupidbench.cell import BUDGET_SECONDS, FLOWS, SEEDS, Cell
from stupidbench.usage import pricing


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="stupidbench")
    parser.add_argument(
        "--cells", type=Path, default=Path("cells"), help="where the cells live"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (
        ("prepare", "stage one cell"),
        ("run", "carry one cell further"),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--flow", required=True, choices=sorted(FLOWS))
        command.add_argument("--seed", required=True, type=int, choices=SEEDS)
        if name == "run":
            command.add_argument(
                "--seconds",
                required=True,
                type=int,
                help="agent time this segment may give the cell",
            )
    commands.add_parser("redact", help="take every credential out of the cells")
    report = commands.add_parser("report", help="draw the curves and write the report")
    report.add_argument("--out", type=Path, default=Path("report"))
    args = parser.parse_args(argv)

    if args.command == "redact":
        deleted, rewritten = redaction.redact(args.cells, redaction.secrets())
        print(f"deleted {deleted} credential paths, rewrote {rewritten} files")
        return 0

    if args.command == "report":
        print(reporting.report(args.cells, args.out, pricing()))
        return 0

    cell = Cell(args.flow, args.seed, args.cells / args.flow / str(args.seed))
    if args.command == "prepare":
        runner.prepare(cell)
        print(f"{cell.flow}/{cell.seed} is {cell.state}")
        return 0

    outcome = runner.run(cell, args.seconds)
    spent = cell.elapsed
    print(
        f"{cell.flow}/{cell.seed} {outcome}: "
        f"{spent / 3600:.2f}h of {BUDGET_SECONDS / 3600:.0f}h spent, "
        f"{len(cell.scores())} scores, now {cell.state}"
    )
    if spent <= 0:
        # A cell that cannot be admitted is waited on rather than refused, and
        # a segment that never ran must not read as one that ran and found
        # nothing.
        print("the cell never started", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
