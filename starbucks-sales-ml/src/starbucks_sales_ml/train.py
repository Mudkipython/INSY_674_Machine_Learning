from __future__ import annotations

import argparse
import json

from starbucks_sales_ml.config import REPORTS_DIR
from pathlib import Path

from starbucks_sales_ml.data import ensure_dataset
from starbucks_sales_ml.modeling import train_and_evaluate, tune_lightgbm


def main() -> None:
    parser = argparse.ArgumentParser(description="Starbucks cafe sales prediction training")
    parser.add_argument("--force-data", action="store_true", help="Regenerate processed data")
    parser.add_argument(
        "--raw-data",
        type=Path,
        default=None,
        help="Path to Kaggle coffee shop sales CSV (raw)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic Starbucks-like data instead of Kaggle data",
    )
    parser.add_argument(
        "--tune-lightgbm",
        action="store_true",
        help="Run LightGBM hyperparameter search and save tuned model",
    )
    parser.add_argument(
        "--include-lightgbm",
        action="store_true",
        help="Include LightGBM in the baseline model comparison",
    )
    args = parser.parse_args()

    df = ensure_dataset(force=args.force_data, raw_path=args.raw_data, use_synthetic=args.synthetic)
    if args.tune_lightgbm:
        results = tune_lightgbm(df)
    else:
        results = train_and_evaluate(df, include_lightgbm=args.include_lightgbm)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORTS_DIR / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
