"""What the week came to: curves against spend, and a table to read them by.

A cell is only credited with a score that beat everything before it, and each
improvement is placed at what had been spent when it landed — dollars, output
tokens, and agent time. Seeds of one flow are averaged, which they can only be
once they are read off a shared grid: the union of every seed's own points, each
seed's curve held flat between its improvements and cut off where its own
coverage ends rather than extrapolated.
"""

from bisect import bisect_right
from dataclasses import dataclass
from datetime import UTC, datetime
from math import log10
from pathlib import Path

import matplotlib
import pandas as pd
import seaborn as sns

from stupidbench.cell import BUDGET_SECONDS, INIT_SCORE, Cell, elapsed_at
from stupidbench.usage import Pricing, cumulative, usages

# A report is drawn where no display is, and the backend must be chosen before
# pyplot is imported.
matplotlib.use("Agg")

from matplotlib import pyplot as plt

#: Every axis a score is plotted against, and how it is labelled.
METRICS = (
    ("cost", "cost (USD)"),
    ("output_tokens", "output tokens"),
    ("hours", "agent hours"),
)

#: The part of a cell's time the vertical axis is set by, and how much of the
#: height that part is given. A cell falls two decades in its first minutes and
#: spends the rest of its day on the last few hundred cycles; the day is what
#: the bench is about, so it gets three quarters of the picture and the fall
#: runs off the top.
LATE = 0.25
HEIGHT = 0.75


@dataclass(frozen=True)
class Point:
    score: float
    cost: float
    output_tokens: int
    hours: float


@dataclass(frozen=True)
class History:
    """One cell's improvements, and how far its curves are allowed to run."""

    flow: str
    seed: int
    state: str
    points: tuple[Point, ...]
    end: Point
    #: What the cell was answered at, and how much of an answer it got: the
    #: reasoning budgets its CLI recorded, how many responses it was sent, the
    #: largest one of them, and how many of their tokens went on thinking where
    #: the CLI counts those apart. A flow that asks for `max` and is quietly
    #: answered at less says so here and nowhere else.
    juice: tuple[str, ...]
    responses: int
    max_output: int
    reasoning_tokens: int


def read_history(cell: Cell, prices: Pricing) -> History:
    events = cell.events()
    spent = usages(cell, prices)
    best = float(INIT_SCORE)
    points = [Point(best, 0.0, 0, 0.0)]
    for moment, score in cell.scores():
        if score >= best:
            continue
        best = float(score)
        tokens, cost = cumulative(spent, moment)
        points.append(Point(best, cost, tokens, elapsed_at(events, moment) / 3600))
    tokens, cost = cumulative(spent, float("inf"))
    return History(
        flow=cell.flow,
        seed=cell.seed,
        state=cell.state,
        points=tuple(points),
        end=Point(best, cost, tokens, elapsed_at(events, float("inf")) / 3600),
        juice=tuple(sorted({usage.effort for usage in spent if usage.effort})),
        responses=len(spent),
        max_output=max((usage.output_tokens for usage in spent), default=0),
        reasoning_tokens=sum(usage.reasoning_tokens for usage in spent),
    )


def frame(histories: list[History]) -> pd.DataFrame:
    """Every seed's curve on its flow's shared grid, ready to be averaged."""
    rows: list[dict[str, object]] = []
    for metric, _ in METRICS:
        for flow in sorted({history.flow for history in histories}):
            group = [
                history
                for history in histories
                if history.flow == flow
                # A cell that improved while reporting nothing on this axis has
                # no curve here, only a point at the origin that would drag the
                # mean of the seeds that do report.
                and not (getattr(history.end, metric) == 0 and len(history.points) > 1)
            ]
            if not group:
                continue
            grid = sorted(
                {
                    getattr(point, metric)
                    for history in group
                    for point in history.points
                }
                | {getattr(history.end, metric) for history in group}
            )
            for history in group:
                coordinates = [getattr(point, metric) for point in history.points]
                end = getattr(history.end, metric)
                for value in grid:
                    if value > end:
                        break
                    index = bisect_right(coordinates, value)
                    rows.append(
                        {
                            "flow": flow,
                            "seed": history.seed,
                            "metric": metric,
                            "x": value,
                            "score": history.points[max(0, index - 1)].score,
                        }
                    )
    return pd.DataFrame(rows, columns=["flow", "seed", "metric", "x", "score"])


def window(curves: pd.DataFrame) -> tuple[float, float] | None:
    """The vertical limits to draw in, or nothing when nothing needs one.

    Set by what was scored after the first quarter of the time — those points
    take three quarters of the height, standing just off the bottom of it — so
    that what the agents did with their day is not a flat line under the two
    decades their first minutes fell. Time is the same for every panel, so all
    three are drawn in one window and can be read against each other.
    """
    hours = curves[curves["metric"] == "hours"]
    if hours.empty:
        return None
    late = hours[hours["x"] >= LATE * hours["x"].max()]["score"]
    low, high = float(late.min()), float(late.max())
    if low <= 0 or high <= low:
        return None
    span = log10(high / low) / HEIGHT
    bottom = low / 10 ** (0.05 * span)
    return bottom, bottom * 10**span


def plot(curves: pd.DataFrame, path: Path) -> None:
    """Draws one panel per metric, each seed group averaged into one line."""
    limits = window(curves)
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, len(METRICS), figsize=(21, 6))
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        # The origin is a cell that has spent nothing, which no log axis can
        # place. What it stands for — the score every cell starts at — is in the
        # report beside these.
        panel = curves[(curves["metric"] == metric) & (curves["x"] > 0)]
        if panel.empty:
            axis.set_axis_off()
            continue
        sns.lineplot(
            data=panel,
            x="x",
            y="score",
            hue="flow",
            estimator="mean",
            errorbar=None,
            drawstyle="steps-post",
            ax=axis,
        )
        axis.set_xlabel(label)
        axis.set_ylabel("cycles (lower is better)")
        # The first minutes take a cell from a hundred and fifty thousand cycles
        # to about two thousand, the twenty-four hours after them are worth a
        # few hundred, and the flows that differ do so across three decades of
        # spend. On linear axes that is a cliff and then a flat line on the
        # floor: the whole bench drawn as nothing.
        axis.set_xscale("log")
        axis.set_yscale("log")
        if limits:
            axis.set_ylim(*limits)
        axis.legend(title="", fontsize="x-small")
    figure.suptitle("stupid bench — mean over seeds")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=120)
    plt.close(figure)


def markdown(histories: list[History], curves_name: str) -> str:
    """The report a run leaves in its job summary.

    A job summary resolves no relative path, so the curves are named rather
    than shown: they are in the artifact this run publishes beside it.
    """
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# stupid bench — {stamp}",
        "",
        f"Curves are in `{curves_name}`, in the **report** artifact of this run.",
        "",
        "## By flow",
        "",
        (
            "| flow | juice | seeds | best | mean best | cost (USD) | "
            "output tokens | mean out | max out | agent hours |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for flow in sorted({history.flow for history in histories}):
        group = [history for history in histories if history.flow == flow]
        best = [history.end.score for history in group]
        responses = sum(history.responses for history in group)
        tokens = sum(history.end.output_tokens for history in group)
        thinking = sum(history.reasoning_tokens for history in group)
        juice = "/".join(sorted({effort for h in group for effort in h.juice})) or "—"
        if thinking and responses:
            juice += f" ({thinking / responses:,.0f})"
        lines.append(
            f"| `{flow}` | {juice} | {len(group)} | {min(best):,.0f} | "
            f"{sum(best) / len(best):,.0f} | "
            f"{sum(h.end.cost for h in group):,.2f} | {tokens:,} | "
            f"{tokens / responses if responses else 0:,.0f} | "
            f"{max(h.max_output for h in group):,} | "
            f"{sum(h.end.hours for h in group):,.1f} |"
        )
    lines += [
        "",
        "## By cell",
        "",
        (
            "| flow | seed | juice | state | best | cost (USD) | "
            "output tokens | mean out | max out | agent hours |"
        ),
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for history in sorted(histories, key=lambda h: (h.flow, h.seed)):
        progress = history.end.hours / (BUDGET_SECONDS / 3600)
        mean_output = (
            history.end.output_tokens / history.responses if history.responses else 0
        )
        juice = "/".join(history.juice) or "—"
        if history.reasoning_tokens and history.responses:
            juice += f" ({history.reasoning_tokens / history.responses:,.0f})"
        lines.append(
            f"| `{history.flow}` | {history.seed} | {juice} | "
            f"{history.state} ({progress:.0%}) | {history.end.score:,.0f} | "
            f"{history.end.cost:,.2f} | {history.end.output_tokens:,} | "
            f"{mean_output:,.0f} | {history.max_output:,} | {history.end.hours:,.1f} |"
        )
    footnote = (
        f"Starting score is {INIT_SCORE:,}, and lower is better. Every flow meets "
        "the task with a new session every turn, so what it carries from one turn "
        "to the next is only what it wrote down. `juice` is the reasoning budget "
        "the CLI recorded its responses running at, and in brackets what one of "
        "them spent thinking, where the CLI counts thinking apart from the rest "
        "of what it wrote; `mean out` and `max out` are the output tokens of one "
        "response, thinking included."
    )
    lines += ["", footnote, ""]
    return "\n".join(lines)


def report(cells_dir: Path, out_dir: Path, prices: Pricing) -> str:
    """Writes the curves and the report, and returns the report."""
    histories = [
        read_history(cell, prices)
        for cell in Cell.all(cells_dir)
        if cell.state != "pending"
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    if not histories:
        text = "# stupid bench\n\nNo cell has run yet.\n"
        (out_dir / "report.md").write_text(text, encoding="utf-8")
        return text
    curves = frame(histories)
    plot(curves, out_dir / "curves.png")
    curves.to_csv(out_dir / "curves.csv", index=False)
    text = markdown(histories, "curves.png")
    (out_dir / "report.md").write_text(text, encoding="utf-8")
    return text
