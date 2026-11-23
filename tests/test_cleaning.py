import pandas as pd
from pandas.testing import assert_frame_equal

from src.mp119.cleaning import FEATURE_COLUMNS, TARGET, clean_training_data


def loom_record(**changes):
    return {
        "record_id": "R1",
        "date": "2026-02-14",
        "loom_age_yr": 5,
        "thread_count": 120,
        "power_cuts_min": 20,
        "weaver_exp_yr": 10,
        "humidity_pct": 55,
        "shift_hours": 6.3,
        "fabric": "cotton",
        "shed": "Shed 2",
        "metres_per_shift": 45.8,
        **changes,
    }


def test_removes_exact_duplicates_without_removing_separate_records():
    record = loom_record()
    raw = pd.DataFrame([record, record.copy(), loom_record(record_id="R2")])

    cleaned = clean_training_data(raw)

    assert cleaned["record_id"].tolist() == ["R1", "R2"]


def test_masks_the_stuck_shift_value_only_in_the_audited_period():
    raw = pd.DataFrame(
        [
            loom_record(record_id="before", date="2026-02-14"),
            loom_record(record_id="first", date="2026-02-15"),
            loom_record(record_id="last", date="2026-03-02"),
            loom_record(record_id="after", date="2026-03-03"),
            loom_record(record_id="other", date="2026-02-20", shift_hours=8.4),
        ]
    )

    shifts = clean_training_data(raw).set_index("record_id")["shift_hours"]

    assert shifts.loc[["first", "last"]].isna().all()
    assert shifts.loc[["before", "after", "other"]].tolist() == [6.3, 6.3, 8.4]


def test_fixes_extra_zero_targets_without_mutating_the_input():
    raw = pd.DataFrame([loom_record(metres_per_shift=275), loom_record(record_id="R2")])
    original = raw.copy(deep=True)

    cleaned = clean_training_data(raw)

    assert cleaned[TARGET].tolist() == [27.5, 45.8]
    assert_frame_equal(raw, original)


def test_only_keeps_service_features_target_and_record_metadata():
    raw = pd.DataFrame([loom_record(post_shift_reading=45.8)])

    cleaned = clean_training_data(raw)

    assert cleaned.columns.tolist() == ["record_id", "date", *FEATURE_COLUMNS, TARGET]
