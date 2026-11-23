"""Repair the data issues found in the original MP119 training CSV."""

import pandas as pd


NUMERIC_FEATURES = (
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
)
CATEGORICAL_FEATURES = ("fabric", "shed")
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "metres_per_shift"

STUCK_SHIFT_START = "2026-02-15"
STUCK_SHIFT_END = "2026-03-02"
STUCK_SHIFT_VALUE = 6.3
# The Stage 1 audit found regular targets up to 71.5 m and extra-zero targets
# starting at 275 m. Its 100 m boundary separates the two in this file.
TARGET_TYPO_MIN = 100


def clean_training_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Return cleaned training records without changing the input frame.

    This repair applies to the original training period only. The model fits
    the imputation value for missing shift lengths on its training split.
    """
    columns = ["record_id", "date", *FEATURE_COLUMNS, TARGET]
    cleaned = frame.drop_duplicates().loc[:, columns].copy()

    dates = pd.to_datetime(cleaned["date"], format="%Y-%m-%d")
    shift_hours = pd.to_numeric(cleaned["shift_hours"]).astype(float)
    during_outage = dates.between(STUCK_SHIFT_START, STUCK_SHIFT_END)
    stuck = during_outage & shift_hours.eq(STUCK_SHIFT_VALUE)
    cleaned["shift_hours"] = shift_hours.mask(stuck)

    target = pd.to_numeric(cleaned[TARGET]).astype(float)
    cleaned[TARGET] = target.mask(target.ge(TARGET_TYPO_MIN), target / 10)
    return cleaned.reset_index(drop=True)
