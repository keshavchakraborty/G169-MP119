"""Tests for Stage 4 drift monitoring, retraining, and holdout prediction."""

import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from src.mp119.monitoring import calculate_psi
from src.mp119.cleaning import NUMERIC_FEATURES


ROOT = Path(__file__).resolve().parents[1]
FINDINGS_PATH = ROOT / "reports" / "findings.yaml"
PREDICTIONS_PATH = ROOT / "predictions" / "holdout_predictions.csv"
DB_PATH = ROOT / "mlflow.db"


def test_psi_identical_distributions_near_zero() -> None:
    rng = np.random.default_rng(42)
    s = pd.Series(rng.normal(loc=10.0, scale=2.0, size=500))
    psi = calculate_psi(s, s)
    assert psi == 0.0 or psi < 0.01


def test_psi_detects_distribution_drift() -> None:
    rng = np.random.default_rng(42)
    s1 = pd.Series(rng.normal(loc=10.0, scale=2.0, size=500))
    s2 = pd.Series(rng.normal(loc=20.0, scale=2.0, size=500))
    psi = calculate_psi(s1, s2)
    # Significant shift should have PSI well above 0.25
    assert psi > 0.5


def test_holdout_predictions_format_and_completeness() -> None:
    assert PREDICTIONS_PATH.is_file(), "Holdout predictions file must exist"
    df = pd.read_csv(PREDICTIONS_PATH)
    assert list(df.columns) == ["record_id", "prediction"]
    assert len(df) == 250
    assert not df["prediction"].isna().any()
    assert df["record_id"].str.startswith("R").all()


def test_findings_yaml_monitoring_complete() -> None:
    assert FINDINGS_PATH.is_file(), "findings.yaml must exist"
    with FINDINGS_PATH.open(encoding="utf-8") as stream:
        findings = yaml.safe_load(stream)

    assert "monitoring" in findings
    monitoring = findings["monitoring"]

    # All numeric features must have a PSI score
    for feature in NUMERIC_FEATURES:
        assert feature in monitoring["psi"]
        assert monitoring["psi"][feature] is not None
        assert isinstance(monitoring["psi"][feature], (int, float))

    assert monitoring["most_drifted_feature"] == "power_cuts_min"
    assert monitoring["worst_served_group"] == "Shed 1"

    assert "Shed 1" in monitoring["mae_by_group"]
    assert "Shed 2" in monitoring["mae_by_group"]
    assert "Shed 3" in monitoring["mae_by_group"]

    assert monitoring["mae_v1_on_check"] is not None
    assert monitoring["mae_v2_on_check"] is not None
    assert monitoring["decision"] in ["do_not_promote", "promote"]


def test_mlflow_model_versions_and_champion() -> None:
    assert DB_PATH.is_file(), "mlflow.db must exist"
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    model_name = "mp119-power-loom-output"
    # Check registered model champion alias
    cur.execute(
        "SELECT alias, version FROM registered_model_aliases WHERE name = ?",
        (model_name,),
    )
    aliases = dict(cur.fetchall())
    assert "champion" in aliases

    # Must have 2 or more versions (v1 baseline and v2 retrained)
    cur.execute(
        "SELECT version, run_id FROM model_versions WHERE name = ?",
        (model_name,),
    )
    versions = cur.fetchall()
    assert len(versions) >= 2

    # Check that runs have data_sha256 logged in params
    for version, run_id in versions:
        cur.execute(
            "SELECT value FROM params WHERE run_uuid = ? AND key = 'data_sha256'",
            (run_id,),
        )
        row = cur.fetchone()
        assert row is not None, f"Missing data_sha256 for run {run_id}"

    conn.close()
