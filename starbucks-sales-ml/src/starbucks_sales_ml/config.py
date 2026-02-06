from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
RANDOM_SEED = 42

RAW_DATA_FILE = DATA_DIR / "coffee_shop_sales_raw.csv"
PROCESSED_DATA_FILE = DATA_DIR / "coffee_shop_sales_daily.csv"
