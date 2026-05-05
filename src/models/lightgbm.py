from __future__ import annotations

import pandas as pd
import lightgbm as lgb

try:
    from features import TARGET_COLUMN
except ModuleNotFoundError:
    from src.features import TARGET_COLUMN


def make_model(cfg: dict) -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(
        n_estimators      = cfg.get("n_estimators", 500),
        max_depth         = cfg.get("max_depth", -1),
        num_leaves        = cfg.get("num_leaves", 31),
        learning_rate     = cfg.get("learning_rate", 0.05),
        subsample         = cfg.get("subsample", 0.8),
        colsample_bytree  = cfg.get("colsample_bytree", 0.8),
        min_child_samples = cfg.get("min_child_samples", 20),
        reg_lambda        = cfg.get("reg_lambda", 1.0),
        n_jobs            = -1,
        random_state      = cfg.get("random_state", 42),
        verbose           = -1,
    )


def fit(
    model: lgb.LGBMRegressor,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    features: list,
    target: str = TARGET_COLUMN,
    sample_weight=None,
) -> lgb.LGBMRegressor:
    if val_df is None:
        model.fit(train_df[features], train_df[target], sample_weight=sample_weight)
        return model
    callbacks = [lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)]
    model.fit(
        train_df[features], train_df[target],
        sample_weight=sample_weight,
        eval_set=[(val_df[features], val_df[target])],
        callbacks=callbacks,
    )
    return model


def n_estimators_used(model: lgb.LGBMRegressor) -> int:
    if hasattr(model, "best_iteration_") and model.best_iteration_ > 0:
        return model.best_iteration_
    return model.n_estimators
