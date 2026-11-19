"""Tier-logit trend forecaster for gta6-trophy-rarity-forecast-01.

Forecasts each GTA VI trophy's Exophase earn rate (%) at a case's target time.

Method (variant A):
  * For every trophy, fit an ordinary-least-squares line to its logit earn
    rate against log-time, x = ln(1 + hours since the 2026-11-19T00:00:00Z
    launch epoch), over all of the case's history reads.
  * Pool the fitted slopes by trophy tier (platinum / gold / silver /
    bronze) and shrink each tier's slope halfway toward the global mean
    slope, so rare tiers borrow strength from the whole board.
  * Extrapolate each trophy's line to the case's target on the log-time
    axis, anchored at the centroid of its own reads (mean logit minus
    pooled slope times mean x), so the level is the trophy's own.
  * Spread the five quantiles on the logit scale like a random walk:
    sigma = 0.5 * sqrt(horizon in days), z = [-1.645, -0.674, 0, 0.674,
    1.645], mapped back to percent and rounded to 4 decimals (half up).

Standard library only. Reads $MAKEFUN_INPUT_DIR/*.in (else /input),
writes $MAKEFUN_OUTPUT_DIR/<id>.out (else /output). Deterministic: no
randomness, no clock reads, cases processed in filename order.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z_VALUES = [-1.645, -0.674, 0.0, 0.674, 1.645]
WALK_SIGMA = 0.5
TIER_SHRINK = 0.5  # weight on the tier's own pooled slope vs the global slope
LAUNCH = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def _clamp_pct(rate):
    """Percent into the logit-safe range (Exophase prints 2 decimals)."""
    return min(99.99, max(0.01, rate))


def _logit(rate):
    p = _clamp_pct(rate) / 100.0
    return math.log(p / (1.0 - p))


def _inv_logit(lg):
    return 100.0 / (1.0 + math.exp(-lg))


def _round4(value):
    """Round to 4 decimals, half up; whole numbers come back as ints."""
    scaled = value * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    out = whole / 10000.0
    return int(out) if float(out).is_integer() else out


def _parse_time(text):
    """ISO-8601 '...Z' timestamp -> aware datetime (UTC)."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _log_hours(moment):
    """ln(1 + hours from the launch epoch to `moment`).

    Clamped so a read up to an hour before launch still maps cleanly
    (log1p is undefined at -1); the earliest real read is 10 minutes
    before launch.
    """
    hours = (moment - LAUNCH).total_seconds() / 3600.0
    return math.log1p(max(hours, -0.999))


def _ols_slope(xs, ys):
    """Least-squares slope of ys on xs; 0.0 when x never varies."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    sxx = sum((x - mean_x) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return sxy / sxx


def _trophy_series(case, index):
    """(x, y) log-time / logit-rate pairs for one trophy, skipping nulls."""
    xs, ys = [], []
    for read in case["history"]:
        rate = read["rates"][index]
        if rate is None:
            continue
        xs.append(_log_hours(_parse_time(read["at"])))
        ys.append(_logit(rate))
    return xs, ys


def forecast_case(case):
    trophies = case["trophies"]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    target_x = _log_hours(_parse_time(case["target"]))
    horizon_days = case["horizon_minutes"] / 1440.0
    spread = WALK_SIGMA * math.sqrt(max(horizon_days, 0.0))

    # Per-trophy own line: slope by least squares, level at its centroid.
    own = []  # (tier, slope, mean_x, mean_y) per trophy
    for i, trophy in enumerate(trophies):
        xs, ys = _trophy_series(case, i)
        if not ys:
            own.append((trophy["tier"], 0.0, target_x, _logit(0.0)))
            continue
        own.append((trophy["tier"], _ols_slope(xs, ys),
                    sum(xs) / len(xs), sum(ys) / len(ys)))

    # Pool slopes per tier, then shrink each tier halfway to the global mean.
    slopes = [entry[1] for entry in own]
    global_slope = sum(slopes) / len(slopes) if slopes else 0.0
    tier_mean = {}
    for tier in {entry[0] for entry in own}:
        members = [entry[1] for entry in own if entry[0] == tier]
        tier_mean[tier] = sum(members) / len(members)
    pooled = {tier: TIER_SHRINK * m + (1.0 - TIER_SHRINK) * global_slope
              for tier, m in tier_mean.items()}

    rows = []
    for i, trophy in enumerate(trophies):
        tier, _slope, mean_x, mean_y = own[i]
        # Forecast on the trophy's own line, re-anchored at the centroid of
        # its reads and extended by the pooled tier slope. Anchoring at the
        # centroid (rather than at the last read or the intercept) keeps the
        # level the trophy's own even when the pooled slope differs from
        # the slope its own reads imply.
        median_logit = mean_y + pooled[tier] * (target_x - mean_x)
        quantiles = [_round4(_clamp_pct(_inv_logit(median_logit + z * spread)))
                     for z in Z_VALUES]
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / (path.stem + ".out")).write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
