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


def plot(curves: pd.DataFrame, path: Path) -> None:
    """Draws one panel per metric, each seed group averaged into one line."""
    sns.set_theme(style="whitegrid", context="talk")
    figure, axes = plt.subplots(1, len(METRICS), figsize=(21, 6))
    for axis, (metric, label) in zip(axes, METRICS, strict=True):
        panel = curves[curves["metric"] == metric]
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
        "| flow | seeds | best | mean best | cost (USD) | output tokens | agent hours |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for flow in sorted({history.flow for history in histories}):
        group = [history for history in histories if history.flow == flow]
        best = [history.end.score for history in group]
        lines.append(
            f"| `{flow}` | {len(group)} | {min(best):,.0f} | "
            f"{sum(best) / len(best):,.0f} | "
            f"{sum(h.end.cost for h in group):,.2f} | "
            f"{sum(h.end.output_tokens for h in group):,} | "
            f"{sum(h.end.hours for h in group):,.1f} |"
        )
    lines += [
        "",
        "## By cell",
        "",
        "| flow | seed | state | best | cost (USD) | output tokens | agent hours |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for history in sorted(histories, key=lambda h: (h.flow, h.seed)):
        progress = history.end.hours / (BUDGET_SECONDS / 3600)
        lines.append(
            f"| `{history.flow}` | {history.seed} | {history.state} ({progress:.0%}) | "
            f"{history.end.score:,.0f} | {history.end.cost:,.2f} | "
            f"{history.end.output_tokens:,} | {history.end.hours:,.1f} |"
        )
    footnote = (
        f"Starting score is {INIT_SCORE:,}, and lower is better. A `ralph` flow "
        "meets the task with a new session every turn; a `stateful` one resumes "
        "the session it left and is sent the task again."
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
