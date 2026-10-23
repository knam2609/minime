"""Compute-cost accounting.

Every run appends a row to ``experiments/costs.csv``.  The point is not the
dollar total -- most of this project runs on free tiers -- but the ratio the
report actually argues from: information gained per GPU-hour.  A row is
written even when the cost is zero, because the hours are the scarce resource.

The same GPU costs different amounts depending on where it is rented, so the
rate is keyed by *venue* rather than by device name.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

# Approximate USD/hour, September 2026.  Free tiers are 0.0 by definition;
# Colab Pro is a monthly subscription amortised over ~100 hours of use.
VENUE_HOURLY_USD = {
    "local": 0.0,
    "kaggle-free": 0.0,
    "colab-free": 0.0,
    "colab-pro": 0.14,
    "vast-4090": 0.35,
    "vast-3090": 0.22,
    "runpod-a10": 0.25,
    "runpod-a100": 1.19,
    "lambda-a10": 0.60,
}

COST_FIELDS = [
    "date", "experiment", "run_id", "venue", "device", "gpu_type",
    "runtime_hours", "est_cost_usd", "tokens_processed", "model_params",
    "tokens_per_second", "final_val_loss", "notes",
]


def hourly_rate(venue: str) -> float:
    if venue not in VENUE_HOURLY_USD:
        raise KeyError(
            f"unknown venue {venue!r}; known venues: {sorted(VENUE_HOURLY_USD)}. "
            "Pass --cost-per-hour to override."
        )
    return VENUE_HOURLY_USD[venue]


def estimate_cost(venue: str, seconds: float, rate_override: float | None = None) -> float:
    rate = rate_override if rate_override is not None else hourly_rate(venue)
    return round(rate * (seconds / 3600.0), 4)


def append_cost_row(
    path: str | Path,
    *,
    experiment: str,
    run_id: str,
    venue: str,
    device: str,
    gpu_type: str,
    runtime_seconds: float,
    tokens_processed: int,
    model_params: int,
    tokens_per_second: float,
    final_val_loss: float | None = None,
    rate_override: float | None = None,
    notes: str = "",
) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "experiment": experiment,
        "run_id": run_id,
        "venue": venue,
        "device": device,
        "gpu_type": gpu_type,
        "runtime_hours": round(runtime_seconds / 3600.0, 4),
        "est_cost_usd": estimate_cost(venue, runtime_seconds, rate_override),
        "tokens_processed": int(tokens_processed),
        "model_params": int(model_params),
        "tokens_per_second": round(tokens_per_second, 1),
        "final_val_loss": "" if final_val_loss is None else round(final_val_loss, 5),
        "notes": notes,
    }

    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COST_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_costs(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def total_spend(path: str | Path) -> tuple[float, float]:
    """Returns ``(total_usd, total_gpu_hours)`` across every recorded run."""
    rows = read_costs(path)
    usd = sum(float(r["est_cost_usd"] or 0) for r in rows)
    hours = sum(float(r["runtime_hours"] or 0) for r in rows)
    return round(usd, 2), round(hours, 2)
