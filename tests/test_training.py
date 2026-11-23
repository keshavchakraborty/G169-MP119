import math

import pandas as pd
import pytest

from src.mp119.cleaning import FEATURE_COLUMNS
from src.mp119.train import chronological_split, make_pipeline


def test_chronological_split_keeps_all_cutoff_day_rows_in_validation():
    rows = pd.DataFrame(
        {
            "record_id": ["last", "middle-1", "first", "middle-2", "middle-3"],
            "date": ["2026-01-03", "2026-01-02", "2026-01-01", "2026-01-02", "2026-01-02"],
        }
    )

    fitting, validation = chronological_split(rows, fraction=0.4)

    assert fitting["record_id"].tolist() == ["first"]
    assert validation["record_id"].tolist() == ["middle-1", "middle-2", "middle-3", "last"]
    assert fitting["date"].max() < validation["date"].min()


def test_pipeline_imputes_missing_shift_hours_with_unseen_fabric():
    fitting = pd.DataFrame(
        {
            "loom_age_yr": [5, 9, 13, 20],
            "thread_count": [100, 130, 160, 180],
            "power_cuts_min": [20, 40, 20, 30],
            "weaver_exp_yr": [10, 12, 14, 12],
            "humidity_pct": [50, 53, 51, 55],
            "shift_hours": [6.0, 8.0, 10.0, float("nan")],
            "fabric": ["cotton", "cotton", "silk", "silk"],
            "shed": ["Shed 1", "Shed 2", "Shed 1", "Shed 2"],
        },
        columns=FEATURE_COLUMNS,
    )
    pipeline = make_pipeline("ridge", {"alpha": 1.0}, seed=119)
    pipeline.fit(fitting, [28.0, 37.0, 49.0, 40.0])
    validation = pd.DataFrame([fitting.iloc[0], fitting.iloc[0]], columns=FEATURE_COLUMNS)
    validation["fabric"] = "linen"
    validation["shift_hours"] = [float("nan"), 8.0]

    missing_shift, explicit_median = pipeline.predict(validation)

    assert math.isfinite(missing_shift)
    assert missing_shift == pytest.approx(explicit_median)
