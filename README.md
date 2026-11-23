# G169 — MP119 Power Loom Output

SCSE3040 MLOps mini project. Stages 1 and 2 cover the data audit and the v1 model.

- Group: **G169**
- Problem: **MP119 — Power Loom Output (Bhopal)**
- Members:
  - S24CSEU0164 — VanshChauhan
  - S24CSEU0063 — LAKSHYASINGHPRATAP
  - S24CSEU0008 — KeshavChakraborty

## Stage 1 data audit

Run `python -m src.mp119.data_audit` from the repository root. It reads the
original CSVs in `data/raw/`, prints the seven checks, and does not rewrite the
data. Results are recorded in `reports/findings.yaml`.

The raw `train.csv` and `later.csv` files are managed by DVC and are not meant
to be committed directly. `holdout_features.csv` is retained as supplied.

## Source data fingerprint

SHA-256 of the original `train.csv`: `e9b618dfcfac6c867aedfd051a2cade181ccb1b421a184d8479314f4b2143ddd`.

## Stage 2 training and tests

Use Python 3.12. Run these commands from the repository root. The tests use
small in-memory examples, so they also run on a fresh clone.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

Put the original `train.csv` from the group archive in `data/raw/` before
training. The pipeline reads it without rewriting it. The cleaner removes 39
exact duplicate copies, marks 94 stuck shift-duration readings as missing
after deduplication, and divides 10 extra-zero targets by ten. The model fits
median imputation on the fitting rows. Only the eight inference features enter
the model.

The comparison fits on 793 records through May 21, 2026 and validates on 207
records from May 22 onward. All settings use the same split.

| Setting | Validation MAE (m) |
| --- | ---: |
| Ridge | 2.1180 |
| Shallow gradient boosting | 2.6296 |
| Deeper gradient boosting | 2.3754 |

Ridge had the lowest validation MAE and was refit on all 1,000 cleaned rows for
v1. The comparison and refit log the original data SHA-256 to `mlflow.db`,
with the v1 run ID in `reports/findings.yaml`. The model is registered as
`mp119-power-loom-output@champion`. Its fitted pipeline is also exported to
`models/model.joblib` for the service.

```bash
python -m src.mp119.train --evaluate-only
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

`--evaluate-only` adds validation runs without creating a new model version.
For a fresh database, run `python -m src.mp119.train` to compare settings,
refit, and register v1.

