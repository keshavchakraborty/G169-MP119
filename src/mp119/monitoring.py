"""Stage 4 drift monitoring, model retraining, and holdout prediction."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error

from .cleaning import FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET, clean_training_data
from .train import RUN_TAGS, export_pipeline, make_pipeline


ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"
FINDINGS_PATH = ROOT / "reports" / "findings.yaml"
MODEL_PATH = ROOT / "models" / "model.joblib"
TRAIN_PATH = ROOT / "data" / "raw" / "train.csv"
LATER_PATH = ROOT / "data" / "raw" / "later.csv"
HOLDOUT_PATH = ROOT / "data" / "raw" / "holdout_features.csv"
PREDICTIONS_PATH = ROOT / "predictions" / "holdout_predictions.csv"


def calculate_psi(reference: pd.Series, actual: pd.Series, num_bins: int = 10, eps: float = 1e-4) -> float:
    """Compute Population Stability Index (PSI) between reference and actual distributions.

    Uses 10 quantile bins derived from the reference distribution, as in P13.
    """
    ref_clean = reference.dropna()
    act_clean = actual.dropna()

    percentiles = np.linspace(0, 100, num_bins + 1)
    bins = np.percentile(ref_clean, percentiles)
    bins[0] = -np.inf
    bins[-1] = np.inf
    bins = np.unique(bins)

    ref_counts = pd.cut(ref_clean, bins=bins).value_counts(sort=False)
    act_counts = pd.cut(act_clean, bins=bins).value_counts(sort=False)

    ref_pct = ref_counts / len(ref_clean)
    act_pct = act_counts / len(act_clean)

    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    act_pct = np.where(act_pct == 0, eps, act_pct)

    psi_value = float(np.sum((act_pct - ref_pct) * np.log(act_pct / ref_pct)))
    return round(psi_value, 4)


def compute_all_psi(train_clean: pd.DataFrame, later: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Compute PSI for all numeric features and return the most drifted feature."""
    psi_dict: dict[str, float] = {}
    for col in NUMERIC_FEATURES:
        psi_dict[col] = calculate_psi(train_clean[col], later[col])

    most_drifted = max(psi_dict, key=psi_dict.get)
    return psi_dict, most_drifted


def compute_mae_by_shed(model: Any, later: pd.DataFrame) -> tuple[dict[str, float], str]:
    """Compute v1 MAE on later.csv grouped by shed, and identify worst served group."""
    features = list(FEATURE_COLUMNS)
    mae_dict: dict[str, float] = {}
    for shed in sorted(later["shed"].unique()):
        subset = later[later["shed"] == shed]
        mae = float(mean_absolute_error(subset[TARGET], model.predict(subset[features])))
        mae_dict[shed] = round(mae, 4)

    worst_group = max(mae_dict, key=mae_dict.get)
    return mae_dict, worst_group


def run_stage_4(record_findings: bool = True) -> dict[str, Any]:
    """Execute monitoring, retraining v2, decision rule, and holdout predictions."""
    with PARAMS_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    tracking = config["tracking"]
    model_name = tracking["model"]
    seed = config["training"]["random_state"]
    candidate_settings = config["training"]["candidates"]

    # 1. Load data
    train_raw = TRAIN_PATH.read_bytes()
    data_sha256 = hashlib.sha256(train_raw).hexdigest()
    train_clean = clean_training_data(pd.read_csv(BytesIO(train_raw)))
    later = pd.read_csv(LATER_PATH)

    # 2. PSI calculation
    psi_results, most_drifted = compute_all_psi(train_clean, later)
    print(f"PSI Results: {psi_results}")
    print(f"Most drifted feature: {most_drifted}")

    # 3. Load v1 model and compute MAE per group on later.csv
    v1_model = joblib.load(MODEL_PATH)
    mae_by_group, worst_group = compute_mae_by_shed(v1_model, later)
    print(f"MAE by Shed: {mae_by_group}")
    print(f"Worst served group: {worst_group}")

    # 4. Retrain v2 on clean train + first 70% of sorted later.csv
    features = list(FEATURE_COLUMNS)
    later_sorted = later.sort_values("date", kind="stable").reset_index(drop=True)
    split_index = int(len(later_sorted) * 0.70)
    later_first_70 = later_sorted.iloc[:split_index]
    later_last_30 = later_sorted.iloc[split_index:]

    # Evaluate v1 on last 30% check set
    mae_v1_check = float(mean_absolute_error(later_last_30[TARGET], v1_model.predict(later_last_30[features])))

    # Combined dataset for v2
    v2_train_data = pd.concat([train_clean, later_first_70], ignore_index=True)

    # Train v2 using winning candidate architecture (ridge)
    v2_candidate_name = "ridge"
    v2_model = make_pipeline(v2_candidate_name, candidate_settings[v2_candidate_name], seed)
    v2_model.fit(v2_train_data[features], v2_train_data[TARGET])

    mae_v2_check = float(mean_absolute_error(later_last_30[TARGET], v2_model.predict(later_last_30[features])))

    mae_v1_rounded = round(mae_v1_check, 4)
    mae_v2_rounded = round(mae_v2_check, 4)
    print(f"v1 MAE on check set (last 30%): {mae_v1_rounded} m")
    print(f"v2 MAE on check set (last 30%): {mae_v2_rounded} m")

    # 5. Client promotion rule:
    # Promote v2 only if its MAE on that last 30% is at least 10% lower than v1's.
    rel_improvement = (mae_v1_check - mae_v2_check) / mae_v1_check
    promoted = rel_improvement >= 0.10
    decision = "promote" if promoted else "do_not_promote"
    print(f"Relative improvement: {rel_improvement * 100:.2f}%. Promotion decision: {decision}")

    # 6. MLflow logging and model registration
    mlflow.set_tracking_uri(tracking["uri"])
    mlflow.set_registry_uri(tracking["uri"])
    client = MlflowClient()

    experiment = client.get_experiment_by_name(tracking["experiment"])
    experiment_id = experiment.experiment_id if experiment else client.create_experiment(
        tracking["experiment"], artifact_location=tracking["artifact_location"]
    )

    signature = infer_signature(v2_train_data[features], v2_model.predict(v2_train_data[features]))

    with mlflow.start_run(experiment_id=experiment_id, run_name="v2-retrain", tags=RUN_TAGS) as run:
        mlflow.log_params({
            "data_sha256": data_sha256,
            "candidate": v2_candidate_name,
            "random_state": seed,
            "fit_rows": len(v2_train_data),
            "check_rows": len(later_last_30),
            "promotion_decision": decision,
            **candidate_settings[v2_candidate_name],
        })
        mlflow.log_metric("mae_v1_check", mae_v1_check)
        mlflow.log_metric("mae_v2_check", mae_v2_check)
        mlflow.log_metric("relative_improvement", rel_improvement)

        logged = mlflow.sklearn.log_model(
            v2_model, name="model", signature=signature, serialization_format="cloudpickle"
        )
        version_v2 = mlflow.register_model(logged.model_uri, model_name)
        print(f"Registered {model_name} v{version_v2.version} (run {run.info.run_id})")

    if promoted:
        client.set_registered_model_alias(model_name, "champion", version_v2.version)
        export_pipeline(v2_model, MODEL_PATH)
        champion_version = int(version_v2.version)
        print(f"Promoted v2! {model_name} v{champion_version} is now @champion.")
    else:
        # Keep v1 as champion
        client.set_registered_model_alias(model_name, "champion", "1")
        champion_version = 1
        print(f"Decision was do_not_promote. {model_name} v1 remains @champion.")

    # 7. Generate Holdout Predictions with @champion model
    champion_pipeline = joblib.load(MODEL_PATH)
    holdout_df = pd.read_csv(HOLDOUT_PATH)
    predictions = champion_pipeline.predict(holdout_df[features])

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    pred_df = pd.DataFrame({
        "record_id": holdout_df["record_id"],
        "prediction": predictions,
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Generated {len(pred_df)} holdout predictions at {PREDICTIONS_PATH}")

    # 8. Update findings.yaml
    if record_findings and FINDINGS_PATH.is_file():
        with FINDINGS_PATH.open(encoding="utf-8") as stream:
            findings = yaml.safe_load(stream)

        findings["monitoring"] = {
            "psi": psi_results,
            "most_drifted_feature": most_drifted,
            "mae_by_group": mae_by_group,
            "worst_served_group": worst_group,
            "mae_v1_on_check": mae_v1_rounded,
            "mae_v2_on_check": mae_v2_rounded,
            "decision": decision,
        }

        with FINDINGS_PATH.open("w", encoding="utf-8") as stream:
            yaml.dump(findings, stream, sort_keys=False)
        print(f"Updated {FINDINGS_PATH}")

    return {
        "psi": psi_results,
        "most_drifted": most_drifted,
        "mae_by_group": mae_by_group,
        "worst_served_group": worst_group,
        "mae_v1_on_check": mae_v1_rounded,
        "mae_v2_on_check": mae_v2_rounded,
        "decision": decision,
        "champion_version": champion_version,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    run_stage_4(record_findings=True)
