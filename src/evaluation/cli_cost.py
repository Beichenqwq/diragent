"""DeepSeek price snapshot; estimate from provider usage, never CLI model prices."""

from __future__ import annotations

PRICE_SNAPSHOT = {
    "verified_on": "2026-10-05",
    "source": "https://api-docs.deepseek.com/quick_start/pricing/",
    "model": "deepseek-flash",
    "model_version": "DeepSeek-V4.1-Flash",
    "usd_per_million": {
        "peak": {"cache_hit_input": 0.006, "cache_miss_input": 0.3, "output": 1.2},
        "off_peak": {"cache_hit_input": 0.003, "cache_miss_input": 0.15, "output": 0.6},
    },
    "peak_rule": "Monday-Friday 01:00-04:00 and 06:00-10:00 UTC, excluding Chinese public holidays",
}


def estimate_cost(usage: dict) -> dict:
    inputs = usage.get("input_tokens", 0)
    cached = usage.get("cached_input_tokens", 0)
    outputs = usage.get("output_tokens", 0)
    if not 0 <= cached <= inputs or outputs < 0:
        raise ValueError("Invalid provider token counts")
    return {period: round(((inputs - cached) * price["cache_miss_input"] +
                            cached * price["cache_hit_input"] + outputs * price["output"]) / 1_000_000, 9)
            for period, price in PRICE_SNAPSHOT["usd_per_million"].items()}
