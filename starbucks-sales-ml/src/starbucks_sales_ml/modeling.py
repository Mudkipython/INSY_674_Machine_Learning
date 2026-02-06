from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from starbucks_sales_ml.config import MODELS_DIR, REPORTS_DIR, RANDOM_SEED


@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.15
    val_size: float = 0.15
    random_seed: int = RANDOM_SEED


def _split_time(df: pd.DataFrame, cfg: SplitConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_sorted = df.sort_values("date").reset_index(drop=True)
    n = len(df_sorted)
    n_test = int(n * cfg.test_size)
    n_val = int(n * cfg.val_size)

    train = df_sorted.iloc[: n - n_val - n_test]
    val = df_sorted.iloc[n - n_val - n_test : n - n_test]
    test = df_sorted.iloc[n - n_test :]
    return train, val, test


def _build_preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", "passthrough", numeric),
        ]
    )


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(y_true, 1.0, None))) * 100)
    return {"rmse": float(rmse), "mae": float(mae), "mape": float(mape)}


def _feature_columns() -> tuple[list[str], list[str]]:
    categorical = ["store_location", "weekday", "month"]
    numeric = [
        "store_id",
        "is_weekend",
        "sales_lag_1",
        "sales_lag_7",
        "sales_lag_14",
        "rolling_7",
        "rolling_28",
    ]
    return categorical, numeric


def train_and_evaluate(df: pd.DataFrame, *, include_lightgbm: bool = False) -> dict[str, dict[str, float]]:
    categorical, numeric = _feature_columns()

    train_df, val_df, test_df = _split_time(df, SplitConfig())

    X_train = train_df[categorical + numeric]
    y_train = train_df["sales_usd"].to_numpy()
    X_val = val_df[categorical + numeric]
    y_val = val_df["sales_usd"].to_numpy()
    X_test = test_df[categorical + numeric]
    y_test = test_df["sales_usd"].to_numpy()

    preprocessor = _build_preprocessor(categorical, numeric)

    models: dict[str, Any] = {
        "ridge": Ridge(alpha=1.2, random_state=RANDOM_SEED),
        "random_forest": RandomForestRegressor(
            n_estimators=240, max_depth=14, random_state=RANDOM_SEED, n_jobs=-1
        ),
        "hist_gbdt": HistGradientBoostingRegressor(
            max_depth=8, learning_rate=0.08, max_iter=300, random_state=RANDOM_SEED
        ),
    }
    if include_lightgbm:
        models["lightgbm"] = LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=RANDOM_SEED,
        )

    results: dict[str, dict[str, float]] = {}
    best_model_name = ""
    best_rmse = float("inf")
    best_pipeline: Pipeline | None = None

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        pipeline = Pipeline([("prep", preprocessor), ("model", model)])
        pipeline.fit(X_train, y_train)
        val_pred = pipeline.predict(X_val)
        metrics = _metrics(y_val, val_pred)
        results[name] = metrics

        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model_name = name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("No model trained.")

    test_pred = best_pipeline.predict(X_test)
    results["best_model"] = {
        "name": best_model_name,
        **_metrics(y_test, test_pred),
    }

    dump(best_pipeline, MODELS_DIR / "best_model.joblib")

    if best_model_name == "lightgbm":
        feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
        importances = best_pipeline.named_steps["model"].feature_importances_
        fi = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi.sort_values("importance", ascending=False).to_csv(
            REPORTS_DIR / "feature_importance.csv", index=False
        )

    return results


def tune_lightgbm(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    categorical, numeric = _feature_columns()
    train_df, val_df, test_df = _split_time(df, SplitConfig())
    train_val = pd.concat([train_df, val_df], ignore_index=True)

    X_train_val = train_val[categorical + numeric]
    y_train_val = train_val["sales_usd"].to_numpy()
    X_test = test_df[categorical + numeric]
    y_test = test_df["sales_usd"].to_numpy()

    preprocessor = _build_preprocessor(categorical, numeric)
    base_model = LGBMRegressor(random_state=RANDOM_SEED, n_estimators=500)
    pipeline = Pipeline([("prep", preprocessor), ("model", base_model)])

    param_distributions = {
        "model__learning_rate": [0.03, 0.05, 0.08, 0.1],
        "model__num_leaves": [31, 48, 64, 96],
        "model__max_depth": [-1, 6, 8, 10],
        "model__subsample": [0.7, 0.85, 0.95],
        "model__colsample_bytree": [0.7, 0.85, 0.95],
        "model__min_child_samples": [10, 20, 30, 50],
    }

    cv = TimeSeriesSplit(n_splits=4)
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=18,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=RANDOM_SEED,
        n_jobs=1,
    )
    search.fit(X_train_val, y_train_val)

    best_pipeline = search.best_estimator_
    test_pred = best_pipeline.predict(X_test)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    dump(best_pipeline, MODELS_DIR / "lightgbm_tuned.joblib")

    results = {
        "lightgbm_tuned": {
            **_metrics(y_test, test_pred),
            "best_params": search.best_params_,
        }
    }

    feature_names = best_pipeline.named_steps["prep"].get_feature_names_out()
    importances = best_pipeline.named_steps["model"].feature_importances_
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi.sort_values("importance", ascending=False).to_csv(
        REPORTS_DIR / "feature_importance_tuned.csv", index=False
    )

    return results
