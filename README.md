# G169 — MP119 Power Loom Output

Stage 1 repository for the SCSE3040 MLOps mini project.

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

