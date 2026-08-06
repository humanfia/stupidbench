"""What a cell spent: output tokens and dollars, read back from session logs.

Each CLI keeps its own record of what it sent and was sent back, and that
record is the only place the count exists — nothing here is metered live. The
prices are OpenRouter's, looked up per model, so a run is costed at what the
tokens it used were worth rather than at what a subscription was billed.
"""

import json
import re
import urllib.request
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

from stupidbench.cell import FLOWS, Cell, read_jsonl, timestamp

PRICING_URL = "https://openrouter.ai/api/v1/models"


@dataclass(frozen=True)
class Usage:
    """One accounted exchange."""

    timestamp: float
    output_tokens: int
    cost: float
    #: The reasoning budget the CLI recorded this exchange as having run at —
    #: the juice, where a provider calls it that — or empty where the CLI writes
    #: none down. A flow asks for `max` and is answered at whatever the provider
    #: gives it, and this is the only place what it was given is stated.
    effort: str = ""
    #: Of the output tokens, the ones spent thinking. Only Codex counts them
    #: apart; Claude and Kimi Code bill them as output and never say how many,
    #: so this is zero where the budget is all that is known.
    reasoning_tokens: int = 0


@dataclass(frozen=True)
class Tier:
    """Prices that apply once a prompt is at least this long."""

    min_prompt_tokens: int
    rates: dict[str, float]


Pricing = dict[str, list[Tier]]


def pricing() -> Pricing:
    """Every model's prices, longest-context tier first."""
    with urllib.request.urlopen(PRICING_URL) as response:
        models = json.load(response)["data"]
    prices: Pricing = {}
    for model in models:
        base = {
            key: float(value)
            for key, value in model["pricing"].items()
            if isinstance(value, str)
        }
        tiers = [Tier(0, base)] + [
            Tier(
                int(override["min_prompt_tokens"]),
                base
                | {
                    key: float(value)
                    for key, value in override.items()
                    if isinstance(value, str)
                },
            )
            for override in model["pricing"].get("overrides") or []
            if override.get("min_prompt_tokens") is not None
        ]
        prices[model["id"].split("/")[-1]] = sorted(
            tiers, key=lambda tier: tier.min_prompt_tokens, reverse=True
        )
    return prices


def usages(cell: Cell, prices: Pricing) -> list[Usage]:
    """Everything the cell's CLI recorded spending, in order."""
    flow = FLOWS[cell.flow]
    if flow.tool == "codex":
        found = _codex_usages(cell.state_dir / "sessions", prices)
    elif flow.tool == "claude":
        found = _claude_usages(cell.state_dir / "projects", prices)
    else:
        found = _kimi_usages(cell.state_dir / "sessions", prices, flow.model)
    return sorted(found, key=lambda usage: usage.timestamp)


def cumulative(usages: list[Usage], moment: float) -> tuple[int, float]:
    """The tokens and dollars spent up to ``moment``."""
    index = bisect_right([usage.timestamp for usage in usages], moment)
    return (
        sum(usage.output_tokens for usage in usages[:index]),
        sum(usage.cost for usage in usages[:index]),
    )


def _rates(prices: Pricing, model: str, prompt_tokens: int) -> dict[str, float]:
    """The rates for ``model`` at that prompt length, or nothing if unpriced."""
    for tier in prices.get(model, []):
        if prompt_tokens >= tier.min_prompt_tokens:
            return tier.rates
    return {}


def _codex_usages(sessions_dir: Path, prices: Pricing) -> list[Usage]:
    found: list[Usage] = []
    for path in sorted(sessions_dir.rglob("*.jsonl")):
        rows = list(read_jsonl(path))
        # A rollout can report usage before it says which model spent it, or at
        # what budget.
        opening = next(
            (row["payload"] for row in rows if row.get("type") == "turn_context"), {}
        )
        model = opening.get("model")
        effort = opening.get("effort") or ""
        for row in rows:
            if row.get("type") == "turn_context":
                model = row["payload"].get("model", model)
                effort = row["payload"].get("effort") or ""
            if row.get("type") != "event_msg" or not isinstance(model, str):
                continue
            payload = row["payload"]
            if payload.get("type") != "token_count" or not payload.get("info"):
                continue
            usage = payload["info"]["last_token_usage"]
            cached = usage.get("cached_input_tokens", 0)
            input_tokens = usage.get("input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            rates = _rates(prices, model, input_tokens)
            found.append(
                Usage(
                    timestamp(row["timestamp"]),
                    output_tokens,
                    (input_tokens - cached) * rates.get("prompt", 0.0)
                    + cached * rates.get("input_cache_read", 0.0)
                    + output_tokens * rates.get("completion", 0.0),
                    effort,
                    usage.get("reasoning_output_tokens", 0),
                )
            )
    return found


def _kimi_usages(sessions_dir: Path, prices: Pricing, model: str) -> list[Usage]:
    # Kimi Code keeps its count on the wire rather than in the session, and
    # records the model it ran as the placeholder the environment filled in, so
    # what the tokens are worth is priced from the flow instead. The two name
    # the same model differently: the CLI takes the family from its own
    # provider, OpenRouter carries it in the identifier.
    model = f"kimi-{model.rpartition('/')[2]}"
    found: list[Usage] = []
    for path in sorted(sessions_dir.rglob("wire.jsonl")):
        effort = ""
        for row in read_jsonl(path):
            # What the CLI was set to think at is a line of its own, and it
            # stands until another one says otherwise.
            if row.get("type") == "config.update":
                effort = row.get("thinkingEffort") or effort
            # A turn is what an exchange costs. Any wider scope the CLI records
            # is the same tokens summed again.
            if row.get("type") != "usage.record" or row.get("usageScope") != "turn":
                continue
            usage = row["usage"]
            input_tokens = usage.get("inputOther", 0)
            cache_read = usage.get("inputCacheRead", 0)
            cache_write = usage.get("inputCacheCreation", 0)
            output_tokens = usage.get("output", 0)
            rates = _rates(prices, model, input_tokens + cache_read + cache_write)
            found.append(
                Usage(
                    # Milliseconds, where every other clock in a cell is seconds.
                    row["time"] / 1000,
                    output_tokens,
                    input_tokens * rates.get("prompt", 0.0)
                    + cache_read * rates.get("input_cache_read", 0.0)
                    + cache_write * rates.get("input_cache_write", 0.0)
                    + output_tokens * rates.get("completion", 0.0),
                    effort,
                )
            )
    return found


def _claude_usages(projects_dir: Path, prices: Pricing) -> list[Usage]:
    found: list[Usage] = []
    seen: set[str] = set()
    for path in sorted(projects_dir.rglob("*.jsonl")):
        for row in read_jsonl(path):
            message = row.get("message") or {}
            usage = message.get("usage")
            if not usage or not message.get("id") or message["id"] in seen:
                continue
            seen.add(message["id"])
            model = message["model"]
            if not isinstance(model, str) or model == "<synthetic>":
                continue
            # Claude's own model names are not OpenRouter's price identifiers.
            model = re.sub(r"-\d{8}$", "", model)
            model = re.sub(r"-(\d+)-(\d+)$", r"-\1.\2", model)
            input_tokens = usage.get("input_tokens", 0)
            cache_read = usage.get("cache_read_input_tokens", 0)
            cache_write = usage.get("cache_creation_input_tokens", 0)
            output_tokens = usage.get("output_tokens", 0)
            rates = _rates(prices, model, input_tokens + cache_read + cache_write)
            found.append(
                Usage(
                    timestamp(row["timestamp"]),
                    output_tokens,
                    input_tokens * rates.get("prompt", 0.0)
                    + cache_read * rates.get("input_cache_read", 0.0)
                    + cache_write * rates.get("input_cache_write", 0.0)
                    + output_tokens * rates.get("completion", 0.0),
                    # The CLI writes it beside the message rather than in it.
                    row.get("effort") or "",
                )
            )
    return found
