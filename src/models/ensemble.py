from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata

try:
    from features import TARGET_COLUMN
except ModuleNotFoundError:
    from src.features import TARGET_COLUMN


class EnsembleModel:
    def __init__(self, models: list, weights: list[float] | None = None):
        self.models  = models
        self.weights = weights if weights is not None else [1 / len(models)] * len(models)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = np.array([m.predict(X) for m in self.models])
        ranked = np.array([rankdata(p) for p in preds])
        return (ranked * np.array(self.weights)[:, None]).sum(axis=0)


def make_model(cfg: dict) -> dict:
    # Returns cfg as a config sentinel; actual sub-models are built in fit().
    return cfg


def fit(
    cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None,
    features: list,
    target: str = TARGET_COLUMN,
    sample_weight=None,
) -> EnsembleModel:
    sub_model_names = cfg.get("sub_models", ["xgboost", "lightgbm"])
    fitted = []
    for name in sub_model_names:
        mod = importlib.import_module(f"models.{name}")
        m   = mod.fit(mod.make_model(cfg), train_df, val_df, features, target, sample_weight)
        fitted.append(m)
    return EnsembleModel(fitted)


def n_estimators_used(model: EnsembleModel) -> int:
    totals = []
    for m in model.models:
        if hasattr(m, "best_iteration") and m.best_iteration is not None:
            totals.append(m.best_iteration + 1)
        elif hasattr(m, "best_iteration_") and m.best_iteration_ > 0:
            totals.append(m.best_iteration_)
        elif hasattr(m, "n_estimators"):
            totals.append(m.n_estimators)
    return int(np.mean(totals)) if totals else 0
