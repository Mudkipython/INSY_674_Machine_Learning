from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from starbucks_sales_ml.config import DATA_DIR, PROCESSED_DATA_FILE, RAW_DATA_FILE, RANDOM_SEED


@dataclass(frozen=True)
class SyntheticConfig:
    n_days: int = 365 * 3
    n_stores: int = 60
    start_date: date = date(2021, 1, 1)
    seed: int = RANDOM_SEED


def _seasonal_index(day_of_year: np.ndarray) -> np.ndarray:
    return 1.0 + 0.12 * np.sin(2 * np.pi * day_of_year / 365.25)


def generate_synthetic_sales_data(cfg: SyntheticConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    dates = np.array([cfg.start_date + timedelta(days=i) for i in range(cfg.n_days)])
    store_ids = np.arange(cfg.n_stores)

    store_size = rng.normal(2400, 450, size=cfg.n_stores).clip(900, 4500)
    seating = (store_size / 35 + rng.normal(0, 10, cfg.n_stores)).clip(10, 140)
    drive_thru = rng.integers(0, 2, size=cfg.n_stores)
    region = rng.choice(["NE", "SE", "MW", "SW", "W"], size=cfg.n_stores)
    urbanicity = rng.choice(["urban", "suburban", "rural"], size=cfg.n_stores, p=[0.5, 0.4, 0.1])

    records: list[dict[str, object]] = []

    for store_idx in store_ids:
        base = 9800 + store_size[store_idx] * 2.2 + seating[store_idx] * 12
        base *= 1.15 if drive_thru[store_idx] else 1.0
        region_mult = {"NE": 1.05, "SE": 0.98, "MW": 0.95, "SW": 1.0, "W": 1.1}[region[store_idx]]
        urban_mult = {"urban": 1.15, "suburban": 1.0, "rural": 0.85}[urbanicity[store_idx]]

        for dt in dates:
            day_of_year = dt.timetuple().tm_yday
            weekday = dt.weekday()
            weekend = 1 if weekday >= 5 else 0

            promo = rng.binomial(1, 0.18)
            holiday = rng.binomial(1, 0.04)
            local_event = rng.binomial(1, 0.07)

            temp_c = rng.normal(16, 10)
            precip_mm = rng.gamma(1.3, 2.0)
            foot_traffic = rng.normal(50, 10) + weekend * 8 + local_event * 12

            seasonal = _seasonal_index(np.array([day_of_year]))[0]
            weekday_mult = 1.18 if weekend else 1.0
            promo_mult = 1.12 if promo else 1.0
            holiday_mult = 1.2 if holiday else 1.0
            weather_penalty = 1.0 - 0.004 * max(0.0, precip_mm - 2)
            temp_boost = 1.0 + 0.0035 * max(0.0, temp_c - 12)

            noise = rng.normal(0, 450)
            sales = (
                base
                * region_mult
                * urban_mult
                * seasonal
                * weekday_mult
                * promo_mult
                * holiday_mult
                * weather_penalty
                * temp_boost
                + 55 * foot_traffic
                + noise
            )

            records.append(
                {
                    "date": dt,
                    "store_id": int(store_idx),
                    "region": region[store_idx],
                    "urbanicity": urbanicity[store_idx],
                    "store_size_sqft": float(store_size[store_idx]),
                    "seating_capacity": float(seating[store_idx]),
                    "drive_thru": int(drive_thru[store_idx]),
                    "weekday": int(weekday),
                    "is_weekend": int(weekend),
                    "month": int(dt.month),
                    "promo": int(promo),
                    "holiday": int(holiday),
                    "local_event": int(local_event),
                    "temp_c": float(temp_c),
                    "precip_mm": float(precip_mm),
                    "foot_traffic_index": float(foot_traffic),
                    "sales_usd": float(max(0.0, sales)),
                }
            )

    return pd.DataFrame.from_records(records)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def load_kaggle_coffee_sales(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Kaggle dataset at {path}")
    df = pd.read_csv(path)
    df = _standardize_columns(df)
    return df


def prepare_daily_store_sales(df_raw: pd.DataFrame) -> pd.DataFrame:
    required = {
        "transaction_date",
        "transaction_qty",
        "unit_price",
        "store_id",
        "store_location",
    }
    missing = required - set(df_raw.columns)
    if missing:
        raise ValueError(f"Missing columns in raw data: {sorted(missing)}")

    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["transaction_date"])
    df["sales_usd"] = df["transaction_qty"] * df["unit_price"]

    daily = (
        df.groupby(["date", "store_id", "store_location"], as_index=False)["sales_usd"]
        .sum()
        .sort_values(["store_id", "date"])
    )

    daily["weekday"] = daily["date"].dt.weekday
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["weekday"] >= 5).astype(int)

    daily["sales_lag_1"] = daily.groupby("store_id")["sales_usd"].shift(1)
    daily["sales_lag_7"] = daily.groupby("store_id")["sales_usd"].shift(7)
    daily["sales_lag_14"] = daily.groupby("store_id")["sales_usd"].shift(14)
    daily["rolling_7"] = daily.groupby("store_id")["sales_usd"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    daily["rolling_28"] = daily.groupby("store_id")["sales_usd"].transform(
        lambda s: s.shift(1).rolling(28).mean()
    )

    daily = daily.dropna().reset_index(drop=True)
    return daily


def ensure_dataset(
    *,
    force: bool = False,
    raw_path: Path | None = None,
    use_synthetic: bool = False,
) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    processed_path = PROCESSED_DATA_FILE

    if processed_path.exists() and not force:
        return pd.read_csv(processed_path, parse_dates=["date"])

    if use_synthetic:
        df = generate_synthetic_sales_data(SyntheticConfig())
        df.to_csv(processed_path, index=False)
        return df

    raw_path = raw_path or RAW_DATA_FILE
    raw_df = load_kaggle_coffee_sales(raw_path)
    daily = prepare_daily_store_sales(raw_df)
    daily.to_csv(processed_path, index=False)
    return daily
