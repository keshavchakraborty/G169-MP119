"""Reproducible Stage 1 checks for the seven train.csv data issues."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
TRAIN = ROOT / "data" / "raw" / "train.csv"
LATER = ROOT / "data" / "raw" / "later.csv"
HOLDOUT = ROOT / "data" / "raw" / "holdout_features.csv"
NUMERIC = (
    "loom_age_yr", "thread_count", "power_cuts_min", "weaver_exp_yr",
    "humidity_pct", "shift_hours", "metres_per_shift",
)
TEXT = ("fabric", "shed")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def audit() -> dict[str, object]:
    rows = read_csv(TRAIN)
    holdout_columns = set(read_csv(HOLDOUT)[0])
    later_rows = read_csv(LATER)
    columns = set(rows[0])
    findings: dict[str, object] = {}

    # A post-event feature must be present in train, absent from holdout, and
    # near-identical to the target. Exclude the target itself from candidates.
    target = [float(row["metres_per_shift"]) for row in rows]
    leaked = []
    for column in sorted(columns - holdout_columns - {"metres_per_shift"}):
        try:
            values = [float(row[column]) for row in rows]
        except (TypeError, ValueError):
            continue
        if len(set(values)) > 1:
            mean_x = sum(values) / len(values)
            mean_y = sum(target) / len(target)
            covariance = sum((x - mean_x) * (y - mean_y) for x, y in zip(values, target))
            scale = (sum((x - mean_x) ** 2 for x in values) * sum((y - mean_y) ** 2 for y in target)) ** 0.5
            if scale and abs(covariance / scale) >= 0.995:
                leaked.append(column)
    findings["leaking_column"] = {"present": bool(leaked), "column": leaked[0] if leaked else None}

    # Compare per-shed ranges and means; investigate material orders-of-magnitude
    # shifts rather than treating normal differences as unit errors.
    unit_candidates = []
    for column in NUMERIC[:-1]:
        by_shed: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_shed[row["shed"]].append(float(row[column]))
        means = {group: sum(values) / len(values) for group, values in by_shed.items()}
        positive = [value for value in means.values() if value > 0]
        if positive and max(positive) / min(positive) >= 10:
            unit_candidates.append((column, max(means, key=means.get)))
    findings["unit_mixup"] = {
        "present": bool(unit_candidates),
        "column": unit_candidates[0][0] if unit_candidates else None,
        "group_value": unit_candidates[0][1] if unit_candidates else None,
        "factor": None,
    }

    sentinel = []
    for column in NUMERIC:
        counts = Counter(row[column] for row in rows)
        for raw_value, count in counts.items():
            try:
                value = float(raw_value)
            except ValueError:
                continue
            if count >= 3 and value in {-9999, -999, -99, -1, 999, 9999}:
                sentinel.append((column, raw_value, count))
    findings["sentinel_value"] = {
        "present": bool(sentinel),
        "column": sentinel[0][0] if sentinel else None,
        "value": float(sentinel[0][1]) if sentinel else None,
        "count": sentinel[0][2] if sentinel else None,
    }

    signatures = Counter(tuple(row[column] for column in rows[0]) for row in rows)
    findings["duplicate_rows"] = {
        "present": any(count > 1 for count in signatures.values()),
        "count": sum(count - 1 for count in signatures.values() if count > 1),
    }

    # Find long exact-value runs on consecutive calendar dates, requiring every
    # row in the span to hold the value. Same-day replicate rows are supported.
    dated = sorted(rows, key=lambda row: (row["date"], row["record_id"]))
    stuck = []
    for column in NUMERIC[:-1]:
        per_day: dict[date, list[str]] = defaultdict(list)
        for row in dated:
            per_day[date.fromisoformat(row["date"])].append(row[column])
        days = sorted(per_day)
        run_start = 0
        for index in range(1, len(days) + 1):
            continues = index < len(days) and days[index] == days[index - 1].fromordinal(days[index - 1].toordinal() + 1) and len(set(per_day[days[index]] + per_day[days[index - 1]])) == 1
            if continues:
                continue
            if index - run_start >= 7:
                first, last = days[run_start], days[index - 1]
                stuck.append((index - run_start, column, first.isoformat(), last.isoformat()))
            run_start = index
    stuck.sort(reverse=True)
    findings["stuck_sensor"] = {
        "present": bool(stuck),
        "column": stuck[0][1] if stuck else None,
        "start_date": stuck[0][2] if stuck else None,
        "end_date": stuck[0][3] if stuck else None,
    }

    # Find target values about ten times other plausible targets, with a clear
    # high-end gap. Compare against rounded tenths to allow target precision.
    ordinary = [float(row["metres_per_shift"]) for row in rows if float(row["metres_per_shift"]) < 100]
    typos = []
    for row in rows:
        value = float(row["metres_per_shift"])
        if value >= 100 and any(abs(value / 10 - candidate) <= 0.11 for candidate in ordinary):
            typos.append(row)
    findings["target_typos"] = {"present": bool(typos), "count": len(typos) if typos else None}

    variants = []
    for column in TEXT:
        observed = set(row[column] for row in rows)
        later_levels = set(row[column] for row in later_rows)
        normalized = {value.strip().casefold() for value in observed}
        if len(observed) > len(normalized) and len(later_levels) <= len(normalized):
            variants.append((column, len(normalized)))
    findings["spelling_variants"] = {
        "present": bool(variants),
        "column": variants[0][0] if variants else None,
        "true_levels": variants[0][1] if variants else None,
    }
    return findings


if __name__ == "__main__":
    import json

    print(json.dumps(audit(), indent=2))

