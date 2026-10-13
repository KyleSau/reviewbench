"""Estimated cost from token usage.

Prices are USD per million tokens, copied from Anthropic's public pricing page
(https://platform.claude.com/docs/en/about-claude/pricing) on 2026-10-13. They change;
treat the cost column as an estimate and re-check the page before quoting numbers.
Unknown models return ``None`` rather than a guessed price.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_RETRIEVED = "2026-10-13"


@dataclass(frozen=True)
class ModelPrice:
    input: float
    output: float
    cache_write_5m: float
    cache_read: float


PRICES_PER_MTOK: dict[str, ModelPrice] = {
    "claude-opus-5-5": ModelPrice(input=4.00, output=20.00, cache_write_5m=5.00, cache_read=0.20),
    "claude-opus-5": ModelPrice(input=5.00, output=25.00, cache_write_5m=6.25, cache_read=0.50),
    "claude-opus-4-8": ModelPrice(input=5.00, output=25.00, cache_write_5m=6.25, cache_read=0.50),
    "claude-sonnet-5-5": ModelPrice(input=2.00, output=10.00, cache_write_5m=2.50, cache_read=0.10),
    # Haiku 5.5 rate for prompts up to 100K tokens; every reviewbench prompt is far below that.
    "claude-haiku-5-5": ModelPrice(input=0.10, output=0.50, cache_write_5m=0.125, cache_read=0.01),
}


def estimate_cost(
    model: str,
    *,
    input_tokens: int,
    output_tokens: int,
    cache_read_input_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
) -> float | None:
    price = PRICES_PER_MTOK.get(model)
    if price is None:
        return None
    total = (
        input_tokens * price.input
        + output_tokens * price.output
        + cache_read_input_tokens * price.cache_read
        + cache_creation_input_tokens * price.cache_write_5m
    )
    return total / 1_000_000
