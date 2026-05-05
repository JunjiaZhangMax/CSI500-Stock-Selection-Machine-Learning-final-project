from __future__ import annotations

import pandas as pd
import xgboost as xgb

try:
    from features import TARGET_COLUMN
except ModuleNotFoundError:
    from src.features import TARGET_COLUMN


def make_model(cfg: dict) -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        n_estimators          = cfg.get("n_estimators", 400),
        max_depth             = cfg.get("max_depth", 5),
        learning_rate         = cfg.get("learning_rate", 0.05),
        subsample             = cfg.get("subsample", 0.8),
        colsample_bytree      = cfg.get("colsample_bytree", 0.8),
        min_child_weight      = cfg.get("min_child_weight", 10),
        reg_lambda            = cfg.get("reg_lambda", 1.0),
        tree_method           = "hist",
        n_jobs                = -1,
        early_stopping_rounds = cfg.get("early_stopping_rounds", 30),
    )


def fit(
    model: xgb.XGBRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    features: list,
    target: str = TARGET_COLUMN,
    sample_weight=None,
) -> xgb.XGBRegressor:
    kwargs = {}
    if val_df is not None and model.early_stopping_rounds is not None:
        kwargs["eval_set"] = [(val_df[features], val_df[target])]
    model.fit(train_df[features], train_df[target],
              sample_weight=sample_weight, verbose=False, **kwargs)
    return model


def n_estimators_used(model: xgb.XGBRegressor) -> int:
    try:
        return model.best_iteration + 1
    except AttributeError:
        return model.n_estimators
