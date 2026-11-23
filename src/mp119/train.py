"""Compare v1 settings, then refit and register the selected pipeline."""

from __future__ import annotations

import argparse
import hashlib
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .cleaning import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES, TARGET
from .cleaning import clean_training_data


RUN_TAGS = {
    "mlflow.user": "G169",
    "mlflow.source.name": "src/mp119/train.py",
}


def chronological_split(frame: pd.DataFrame, fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Hold out the latest rows, including every row on the cutoff date."""
    if not 0 < fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")
    ordered = frame.sort_values("date", kind="stable")
    if ordered["date"].nunique() < 2:
        raise ValueError("Validation requires at least two distinct training dates.")
    split_index = min(len(ordered) - 1, int(len(ordered) * (1 - fraction)))
    cutoff = ordered["date"].iloc[split_index]
    fitting = ordered.loc[ordered["date"] < cutoff]
    validation = ordered.loc[ordered["date"] >= cutoff]
    if fitting.empty:
        raise ValueError("The validation cutoff takes every training date.")
    return fitting, validation


def make_pipeline(name: str, settings: dict, seed: int) -> Pipeline:
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer([
        ("numeric", numeric, list(NUMERIC_FEATURES)),
        ("categorical", categorical, list(CATEGORICAL_FEATURES)),
    ])
    if name == "ridge":
        model = Ridge(**settings)
    elif name.startswith("gradient_boosting_"):
        model = GradientBoostingRegressor(random_state=seed, **settings)
    else:
        raise ValueError(f"Unknown candidate: {name}")
    return Pipeline([("preprocess", preprocess), ("regression", model)])


def export_pipeline(pipeline: Pipeline, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp_directory:
        temp_path = Path(temp_directory) / path.name
        joblib.dump(pipeline, temp_path, compress=3)
        temp_path.replace(path)


def train(evaluate_only: bool = False) -> None:
    with Path("params.yaml").open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    options = config["training"]
    tracking = config["tracking"]
    model_name = tracking["model"]
    model_path = Path(config["paths"]["champion"])
    candidates = options["candidates"]
    if len(candidates) < 3:
        raise ValueError("Configure at least three candidate settings.")

    raw = Path(config["paths"]["train"]).read_bytes()
    data_sha256 = hashlib.sha256(raw).hexdigest()
    clean = clean_training_data(pd.read_csv(BytesIO(raw)))
    fitting, validation = chronological_split(clean, options["validation_fraction"])
    features = list(FEATURE_COLUMNS)
    seed = options["random_state"]

    mlflow.set_tracking_uri(tracking["uri"])
    mlflow.set_registry_uri(tracking["uri"])
    client = MlflowClient()
    versions = client.search_model_versions(f"name = '{model_name}'")
    if versions and not evaluate_only:
        if len(versions) != 1 or str(versions[0].version) != "1":
            raise ValueError("The registry contains later versions. Use --evaluate-only.")
        version = versions[0]
        original_run = client.get_run(version.run_id) if version.run_id else None
        if original_run is None or original_run.data.params.get("data_sha256") != data_sha256:
            raise ValueError("The existing v1 was not registered by this training data.")
        registered = client.get_registered_model(model_name)
        if not model_path.is_file() or str(registered.aliases.get("champion")) != "1":
            pipeline = mlflow.sklearn.load_model(f"models:/{model_name}/1")
            export_pipeline(pipeline, model_path)
            client.set_registered_model_alias(model_name, "champion", "1")
        print(f"{model_name} v1 @champion ({version.run_id})")
        return
    experiment = client.get_experiment_by_name(tracking["experiment"])
    if experiment is None:
        experiment_id = client.create_experiment(
            tracking["experiment"], artifact_location=tracking["artifact_location"]
        )
    else:
        experiment_id = experiment.experiment_id

    validation_start = pd.Timestamp(validation["date"].min()).date().isoformat()
    results = []
    for name, settings in candidates.items():
        with mlflow.start_run(experiment_id=experiment_id, run_name=name, tags=RUN_TAGS) as run:
            mlflow.log_params({
                "data_sha256": data_sha256,
                "candidate": name,
                "random_state": seed,
                "fit_rows": len(fitting),
                "validation_rows": len(validation),
                "validation_start": validation_start,
                **settings,
            })
            pipeline = make_pipeline(name, settings, seed)
            pipeline.fit(fitting[features], fitting[TARGET])
            mae = mean_absolute_error(validation[TARGET], pipeline.predict(validation[features]))
            mlflow.log_metric("validation_mae", mae)
            results.append((mae, name, run.info.run_id))
            print(f"{name}: validation MAE {mae:.4f} m ({run.info.run_id})")

    best_mae, best_name, selection_run_id = min(results)
    if evaluate_only:
        return

    winner = make_pipeline(best_name, candidates[best_name], seed)
    winner.fit(clean[features], clean[TARGET])
    signature = infer_signature(clean[features], winner.predict(clean[features]))
    with mlflow.start_run(experiment_id=experiment_id, run_name="v1-refit", tags=RUN_TAGS) as run:
        mlflow.log_params({
            "data_sha256": data_sha256,
            "candidate": best_name,
            "random_state": seed,
            "fit_rows": len(clean),
            "selection_run_id": selection_run_id,
            **candidates[best_name],
        })
        mlflow.log_metric("selection_mae", best_mae)
        logged = mlflow.sklearn.log_model(
            winner, name="model", signature=signature, serialization_format="cloudpickle"
        )
        version = mlflow.register_model(logged.model_uri, model_name)
        export_pipeline(winner, model_path)
        client.set_registered_model_alias(model_name, "champion", version.version)
        print(f"{model_name} v{version.version} @champion ({run.info.run_id})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate-only", action="store_true", help="Log candidate runs only.")
    train(evaluate_only=parser.parse_args().evaluate_only)
