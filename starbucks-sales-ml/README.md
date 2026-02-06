# Starbucks Cafe Sales Prediction (Sample)

This is a small, self-contained ML sample that uses the Kaggle Coffee Shop Sales dataset and trains a few models, including LightGBM.

## Setup (uv)

```bash
cd /Users/fredericzhang/starbucks-sales-ml
uv venv
source .venv/bin/activate
uv pip install -e .
```

## Data

Download the Kaggle Coffee Shop Sales dataset CSV and place it here:

- `data/coffee_shop_sales_raw.csv`

The pipeline aggregates daily store sales and adds lag features.

## Train

```bash
uv run python -m starbucks_sales_ml.train
```

Include LightGBM in the baseline comparison:

```bash
uv run python -m starbucks_sales_ml.train --include-lightgbm
```

## Tune LightGBM

```bash
uv run python -m starbucks_sales_ml.train --tune-lightgbm
```

Outputs:
- `data/coffee_shop_sales_daily.csv`
- `models/best_model.joblib`
- `models/lightgbm_tuned.joblib` (when tuning)
- `reports/metrics.json`
- `reports/feature_importance.csv` (when LightGBM wins)
- `reports/feature_importance_tuned.csv` (when tuning)

## Lint and Type Check

```bash
uv run ruff check src scripts
uv run ty check src
```

## Notes

- The dataset is the public Coffee Shop Sales dataset from Kaggle.
- The time split preserves temporal ordering (train -> validation -> test).
- Use `--synthetic` to run the old synthetic Starbucks-like data.
