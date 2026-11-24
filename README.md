# G169 — MP119 Power Loom Output

SCSE3040 MLOps mini project covering data audit, model training, prediction service, containerization, Kubernetes manifests, CI/CD, drift monitoring, retraining, and holdout prediction.

- Group: **G169**
- Problem: **MP119 — Power Loom Output (Bhopal)**
- Members:
  - S24CSEU0164 — VanshChauhan
  - S24CSEU0063 — LAKSHYASINGHPRATAP
  - S24CSEU0008 — KeshavChakraborty

---

## Stage 1: Data Audit

Run `python -m src.mp119.data_audit` from the repository root. It reads the
original CSVs in `data/raw/`, prints the seven checks, and does not rewrite the
data. Results are recorded in `reports/findings.yaml`.

The raw `train.csv` and `later.csv` files are managed by DVC and are not meant
to be committed directly. `holdout_features.csv` is retained as supplied.

### Source Data Fingerprint
SHA-256 of the original `train.csv`: `e9b618dfcfac6c867aedfd051a2cade181ccb1b421a184d8479314f4b2143ddd`.

---

## Stage 2: Training and Model Tracking

Use Python 3.12. Run these commands from the repository root:

```bash
python -m pip install -r requirements.txt
python -m pytest -v
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

---

## Stage 3: Service, Containerization, CI, and Kubernetes

### 1. FastAPI Service (`app/main.py`)
- `GET /health` returns `{"status": "ok"}`.
- `POST /predict` accepts the 8 required features and returns the prediction in metres:
  - `loom_age_yr`: float
  - `thread_count`: float
  - `power_cuts_min`: float (Rule 1: must be `<= 250` via `Field(..., le=250)`, else HTTP 422)
  - `weaver_exp_yr`: float
  - `humidity_pct`: float
  - `shift_hours`: float (Rule 2: must be `>= 3` via `Field(..., ge=3)`, else HTTP 422)
  - `fabric`: str
  - `shed`: str
- **Structured JSON Logging**: Every prediction is logged as a structured JSON line to `sys.stdout` (via `JsonFormatter`), recording timestamp, level, inputs, and prediction without logging sensitive personal data.

To run the service locally:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 2. Docker Container (`Dockerfile`, `.dockerignore`)
Built with `python:3.12-slim`:
```bash
docker build -t mp119-api:1.0 .
docker run -p 8000:8000 mp119-api:1.0
```

### 3. Continuous Integration (`.github/workflows/ci.yml`)
Configured to trigger on every `push` and `pull_request`, installing pinned dependencies from `requirements.txt` and running the test suite with `pytest`. Tests do not rely on raw DVC files and run reproducibly in CI.

### 4. Kubernetes Manifests (`k8s/`)
- `k8s/deployment.yaml`: Deployment name `mp119-api`, 4 replicas, container image `mp119-api:1.0`, container port `8000`, readiness probe on `/health` (initialDelaySeconds: 5), and liveness probe on `/health` (initialDelaySeconds: 15).
- `k8s/service.yaml`: Service name `mp119-svc`, port `80` forwarding to targetPort `8000`, matching deployment label `app: mp119-api`.

---

## Stage 4: Monitoring, Retraining, and Holdout Predictions

Run `python -m src.mp119.monitoring` from the repository root.

### 1. Feature Drift (PSI)
Population Stability Index (PSI) computed across 10 quantile bins between cleaned `train.csv` and `later.csv`:

| Feature | PSI | Drift Status |
| --- | ---: | --- |
| `loom_age_yr` | 0.0274 | Stable |
| `thread_count` | 0.0226 | Stable |
| `power_cuts_min` | 3.7266 | **Severe Drift** |
| `weaver_exp_yr` | 0.0192 | Stable |
| `humidity_pct` | 0.0324 | Stable |
| `shift_hours` | 0.0339 | Stable |

- **Most drifted feature**: `power_cuts_min` (PSI: 3.7266)

### 2. Subgroup Performance
v1 model MAE evaluated on `later.csv` by shed:
- `Shed 1`: 7.9253 m
- `Shed 2`: 3.3482 m
- `Shed 3`: 3.4922 m
- **Worst served group**: `Shed 1` (MAE: 7.9253 m)

### 3. Retraining v2 and Promotion Decision
`later.csv` is sorted chronologically by date:
- First 70% (210 rows) combined with cleaned `train.csv` to retrain **v2**.
- Last 30% (90 rows) used as the check set.
- v1 MAE on check set: **4.7367 m**
- v2 MAE on check set: **4.3752 m**
- Relative improvement: `(4.7367 - 4.3752) / 4.7367 = 7.63%`
- **Promotion Rule**: Promote v2 only if MAE on check set improves by at least 10%.
- **Decision**: `do_not_promote` (7.63% < 10%). v1 remains `@champion`.
- Model version 2 registered in `mlflow.db`.

### 4. Holdout Predictions
Generated using `@champion` model on `data/raw/holdout_features.csv` (250 rows).
Output saved at `predictions/holdout_predictions.csv` with exactly columns `record_id,prediction`.
