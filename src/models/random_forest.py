from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

try:
    from features import TARGET_COLUMN
except ModuleNotFoundError:
    from src.features import TARGET_COLUMN


def make_model(cfg: dict) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators     = cfg.get("n_estimators", 300),
        max_depth        = cfg.get("max_depth", None),
        min_samples_leaf = cfg.get("min_samples_leaf", 20),
        max_features     = cfg.get("max_features", "sqrt"),
        n_jobs           = -1,
        random_state     = cfg.get("random_state", 42),
    )


def fit(
    model: RandomForestRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    features: list,
    target: str = TARGET_COLUMN,
) -> RandomForestRegressor:
    model.fit(train_df[features], train_df[target])
    return model


def n_estimators_used(model: RandomForestRegressor) -> int:
    return model.n_estimators
