from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    from features import FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN
except ModuleNotFoundError:
    from src.features import FEATURE_COLUMNS, FORWARD_HORIZON, TARGET_COLUMN

MIN_STOCKS  = 30
MAX_WEIGHT  = 0.10
DEFAULT_TOP_K = 30


def rank_ic(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray,
            top_pct: float | None = None) -> float:
    """Mean daily cross-sectional Spearman IC across all dates.

    If top_pct is given, restrict each day to the predicted top-pct fraction
    (e.g. top_pct=0.2 → top-20% by predicted score).
    """
    ics = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < 20:
            continue
        pred_d = y_pred[mask]
        true_d = y_true[mask]
        if top_pct is not None:
            cutoff   = np.quantile(pred_d, 1 - top_pct)
            top_mask = pred_d >= cutoff
            if top_mask.sum() < 10:
                continue
            pred_d = pred_d[top_mask]
            true_d = true_d[top_mask]
        rho, _ = spearmanr(true_d, pred_d)
        if not np.isnan(rho):
            ics.append(rho)
    return float(np.mean(ics)) if ics else float("nan")


def hit_rate(y_true: np.ndarray, y_pred: np.ndarray, dates: np.ndarray,
             top_k: int = DEFAULT_TOP_K, actual_k: int | None = None) -> float:
    """Mean daily fraction of predicted top-K that are in actual top-actual_k.

    actual_k defaults to top_k (strict). Set actual_k > top_k for a lenient
    version, e.g. actual_k=60 asks "how many of my 30 picks land in the true top-60?"
    """
    if actual_k is None:
        actual_k = top_k
    rates = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < actual_k * 2:
            continue
        pred_d = y_pred[mask]
        true_d = y_true[mask]
        pred_top = set(np.argsort(pred_d)[-top_k:])
        true_top = set(np.argsort(true_d)[-actual_k:])
        rates.append(len(pred_top & true_top) / top_k)
    return float(np.mean(rates)) if rates else float("nan")


def build_portfolio(scores: pd.Series, top_k: int = DEFAULT_TOP_K) -> pd.Series:
    """Top-K names weighted by rank, capped at MAX_WEIGHT, summing to 1.

    Rank-weights are used instead of raw score-weights so pathological score
    scales don't produce a single dominant name. Excess weight from capped
    names is redistributed iteratively to uncapped names.
    """
    if top_k < MIN_STOCKS:
        raise ValueError(f"top_k must be >= {MIN_STOCKS} (competition rule)")
    chosen = scores.sort_values(ascending=False).head(top_k).copy()

    ranks = np.arange(top_k, 0, -1, dtype=float)
    w = pd.Series(ranks / ranks.sum(), index=chosen.index)

    for _ in range(50):
        over = w > MAX_WEIGHT
        if not over.any():
            break
        excess = (w[over] - MAX_WEIGHT).sum()
        w[over] = MAX_WEIGHT
        free = ~over
        if not free.any():
            break
        w[free] += excess * w[free] / w[free].sum()

    assert abs(w.sum() - 1.0) < 1e-6, f"weights sum to {w.sum()}"
    assert (w <= MAX_WEIGHT + 1e-9).all(), "cap violated"
    assert (w > 0).sum() >= MIN_STOCKS, "too few names"
    return w


def rolling_backtest(
    panel: pd.DataFrame,
    model,
    start_date: str,
    end_date: str,
    top_k: int = DEFAULT_TOP_K,
    features: list | None = None,
    index_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """In-process rolling backtest stepping over non-overlapping FORWARD_HORIZON windows.

    Returns a DataFrame with columns:
        as_of, portfolio_return, bench_return, excess_return
    bench_return is CSI500 return over the same window (NaN if index_df not provided).
    excess_return = portfolio_return - bench_return.
    """
    if features is None:
        features = FEATURE_COLUMNS

    trading_dates = np.sort(panel["date"].unique())
    window_dates  = trading_dates[
        (trading_dates >= np.datetime64(pd.Timestamp(start_date))) &
        (trading_dates <= np.datetime64(pd.Timestamp(end_date)))
    ]

    idx_dates = np.sort(index_df["date"].unique()) if index_df is not None else None

    results = []
    for i in range(0, len(window_dates) - FORWARD_HORIZON + 1, FORWARD_HORIZON):
        as_of   = pd.Timestamp(window_dates[i])
        pred_df = panel[panel["date"] == as_of].dropna(subset=features).copy()
        if pred_df.empty:
            continue
        pred_df = pred_df.assign(score=model.predict(pred_df[features]))
        scores  = pred_df.set_index("stock_code")["score"]
        try:
            weights = build_portfolio(scores, top_k=top_k)
        except Exception:
            continue
        realized = pred_df.set_index("stock_code")[TARGET_COLUMN].dropna()
        common   = weights.index.intersection(realized.index)
        if len(common) < 10:
            continue
        port_ret = float((weights[common] * realized[common]).sum())

        bench_ret = float("nan")
        if idx_dates is not None:
            pos = int(np.searchsorted(idx_dates, np.datetime64(as_of)))
            exit_pos = pos + FORWARD_HORIZON
            if exit_pos < len(idx_dates):
                entry_row = index_df[index_df["date"] == pd.Timestamp(idx_dates[pos])]["close"].values
                exit_row  = index_df[index_df["date"] == pd.Timestamp(idx_dates[exit_pos])]["close"].values
                if len(entry_row) > 0 and len(exit_row) > 0 and entry_row[0] > 0:
                    bench_ret = float(exit_row[0] / entry_row[0] - 1)

        results.append({
            "as_of":            as_of,
            "portfolio_return": port_ret,
            "bench_return":     bench_ret,
            "excess_return":    port_ret - bench_ret if not np.isnan(bench_ret) else float("nan"),
        })

    cols = ["as_of", "portfolio_return", "bench_return", "excess_return"]
    return pd.DataFrame(results) if results else pd.DataFrame(columns=cols)
