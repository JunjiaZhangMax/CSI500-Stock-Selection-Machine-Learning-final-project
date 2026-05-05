"""
run_experiment.py — unified experiment entry point (Walk-Forward Validation)

Usage
-----
  python src/run_experiment.py
  python src/run_experiment.py --notes "depth=7 vs baseline"

Edit the CONFIG dict at the bottom, then run. Results are appended to
outputs/experiments.csv and the submission file is saved to
outputs/submissions/<exp_name>.csv.

Walk-Forward split layout (n_folds=4)
--------------------------------------
Fold 1: [─── train ───────────────────][emb][val]
Fold 2: [─── train ──────────────────────][emb][val]
Fold 3: [─── train ─────────────────────────][emb][val]
Fold 4: [─── train ────────────────────────────][emb][val]  <- final model
                                                  val windows are non-overlapping
The logged val_rank_ic is the mean across all folds.
"""
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from features import FEATURE_COLUMNS, INDUSTRY_FEATURE_COLUMNS, TARGET_COLUMN, TARGET_RANK_COLUMN, build_features, training_frame, prediction_frame
from portfolio import rank_ic, hit_rate, build_portfolio, rolling_backtest
from experiment_tracker import log_experiment

DATA_DIR   = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

FORWARD_HORIZON = 3
VAL_DAYS        = 10
EMBARGO_DAYS    = 3


# ── time-decay sample weights ────────────────────────────────────────────────

def time_decay_weights(
    train_df: pd.DataFrame,
    half_life: int,
    floor: float = 0.0,
) -> np.ndarray:
    """Exponential time-decay weights with a given half-life in trading days.

    Most recent date in train_df gets weight 1.0; a date half_life trading days
    earlier gets weight 0.5.  w(t) = 2^(-(T - t) / half_life).

    With floor > 0, weights are clipped to floor (so old data keeps a baseline
    influence instead of decaying to zero).
    """
    sorted_dates = np.sort(train_df["date"].unique())
    date_to_idx  = {pd.Timestamp(d): i for i, d in enumerate(sorted_dates)}
    n     = len(sorted_dates)
    tdx   = train_df["date"].map(date_to_idx).values
    delta = (n - 1) - tdx          # trading days from most recent training date
    w = np.exp(-np.log(2) * delta / half_life)
    return np.maximum(w, floor) if floor > 0 else w


# ── walk-forward split ────────────────────────────────────────────────────────

def walk_forward_splits(
    all_dates: np.ndarray,
    n_folds: int,
    val_days: int = VAL_DAYS,
    embargo_days: int = EMBARGO_DAYS,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Return (train_end, val_start, val_end) for each fold, oldest first.

    Expanding window: each fold's training set grows as val moves forward.
    Val windows are non-overlapping and separated from training by embargo_days.
    """
    MIN_TRAIN = 60
    needed = MIN_TRAIN + n_folds * (val_days + embargo_days)
    if len(all_dates) < needed:
        raise ValueError(
            f"Not enough dates: {n_folds} folds require at least {needed} trading days, "
            f"got {len(all_dates)}. Reduce n_folds or val_days."
        )

    splits = []
    for i in range(n_folds):
        # i=0 -> most recent val window; i=n_folds-1 -> oldest
        val_end_idx   = len(all_dates) - 1 - i * (val_days + embargo_days)
        val_start_idx = val_end_idx - val_days + 1
        train_end_idx = val_start_idx - embargo_days - 1

        if train_end_idx < MIN_TRAIN:
            break

        splits.append((
            pd.Timestamp(all_dates[train_end_idx]),
            pd.Timestamp(all_dates[val_start_idx]),
            pd.Timestamp(all_dates[val_end_idx]),
        ))

    return list(reversed(splits))   # chronological order: oldest -> newest


# ── main runner ───────────────────────────────────────────────────────────────

def run(cfg: dict, notes: str = "") -> dict:
    features   = cfg.get("features", FEATURE_COLUMNS)
    target_col = cfg.get("target", TARGET_COLUMN)
    top_k      = cfg.get("top_k", 50)
    n_folds    = cfg.get("n_folds", 5)

    print(f"\n{'='*60}")
    print(f"  {cfg['exp_name']}  [{cfg['model']}]  top_k={top_k}  folds={n_folds}")
    print(f"  features ({len(features)}): {features}")
    print(f"{'='*60}")

    # load data and build feature panel
    prices   = pd.read_parquet(DATA_DIR / "prices.parquet")
    index_df = pd.read_parquet(DATA_DIR / "index.parquet")
    index_df["date"] = pd.to_datetime(index_df["date"])
    industry_path = DATA_DIR / "industry.csv"
    industry_map  = pd.read_csv(industry_path, dtype={"stock_code": str}) if industry_path.exists() else None
    panel  = build_features(prices, industry_map=industry_map)

    missing = [f for f in features if f not in panel.columns]
    if missing:
        raise ValueError(f"Feature columns not found in panel — add them to features.py first: {missing}")

    # cap training pool to prevent future leakage when using --as-of
    trading_dates = np.sort(panel["date"].unique())
    as_of_ts  = panel["date"].max()
    as_of_idx = int(np.searchsorted(trading_dates, np.datetime64(as_of_ts)))
    cutoff    = pd.Timestamp(trading_dates[max(0, as_of_idx - FORWARD_HORIZON)])
    train_pool = training_frame(panel, max_date=cutoff, target=target_col)
    all_dates  = np.sort(train_pool["date"].unique())

    # walk-forward cross-validation
    splits = walk_forward_splits(all_dates, n_folds=n_folds)
    print(f">> Walk-Forward: {len(splits)} folds  val={VAL_DAYS}d  embargo={EMBARGO_DAYS}d")

    try:
        mod = importlib.import_module(f"models.{cfg['model']}")
    except ModuleNotFoundError:
        raise ValueError(
            f"Unknown model '{cfg['model']}'. "
            f"Add src/models/{cfg['model']}.py with make_model / fit / n_estimators_used."
        )

    fold_ics       = []
    fold_top20_ics = []
    fold_top50_ics = []
    fold_hit_rates = []
    best_iters     = []

    half_life      = cfg.get("half_life", None)
    weight_floor   = cfg.get("weight_floor", 0.0)
    rolling_window = cfg.get("rolling_window", None)   # trading-day window; None = expanding

    for k, (fold_train_end, fold_val_start, fold_val_end) in enumerate(splits):
        fold_train = train_pool[train_pool["date"] <= fold_train_end]
        if rolling_window is not None:
            _train_dates = np.sort(fold_train["date"].unique())
            if len(_train_dates) > rolling_window:
                _window_start = pd.Timestamp(_train_dates[-rolling_window])
                fold_train = fold_train[fold_train["date"] >= _window_start]
        fold_val   = train_pool[
            (train_pool["date"] >= fold_val_start) &
            (train_pool["date"] <= fold_val_end)
        ]
        sw = time_decay_weights(fold_train, half_life, weight_floor) if half_life else None
        fold_model = mod.fit(mod.make_model(cfg), fold_train, fold_val, features, target_col, sw)
        val_true   = fold_val[target_col].to_numpy()
        val_pred   = fold_model.predict(fold_val[features])
        val_dates  = fold_val["date"].to_numpy()

        fold_ic       = rank_ic(val_true, val_pred, val_dates)
        fold_top20_ic = rank_ic(val_true, val_pred, val_dates, top_pct=0.2)
        fold_top50_ic = rank_ic(val_true, val_pred, val_dates, top_pct=0.5)
        fold_hr       = hit_rate(val_true, val_pred, val_dates, top_k=top_k, actual_k=top_k*2)

        fold_ics.append(fold_ic)
        fold_top20_ics.append(fold_top20_ic)
        fold_top50_ics.append(fold_top50_ic)
        fold_hit_rates.append(fold_hr)
        best_iters.append(mod.n_estimators_used(fold_model))

        print(f"   fold {k+1}/{len(splits)}: "
              f"train <= {fold_train_end.date()}  "
              f"val [{fold_val_start.date()}, {fold_val_end.date()}]  "
              f"IC={fold_ic:.4f}  top20={fold_top20_ic:.4f}  "
              f"top50={fold_top50_ic:.4f}  hit={fold_hr:.3f}")

    mean_ic      = float(np.mean(fold_ics))
    std_ic       = float(np.std(fold_ics))
    mean_top20ic = float(np.mean(fold_top20_ics))
    mean_top50ic = float(np.mean(fold_top50_ics))
    mean_hr      = float(np.mean(fold_hit_rates))
    print(f">> Walk-Forward IC: {mean_ic:.4f} +/- {std_ic:.4f}"
          f"  top20_IC={mean_top20ic:.4f}  top50_IC={mean_top50ic:.4f}"
          f"  hit_rate={mean_hr:.3f}")

    # final model: last fold has the most training data and most recent val window.
    # For XGBoost, fix n_estimators to the mean best_iteration and disable early
    # stopping so the val set is not consumed during final training.
    final_train_end, final_val_start, final_val_end = splits[-1]
    train_df = train_pool[train_pool["date"] <= final_train_end]
    if rolling_window is not None:
        _final_dates = np.sort(train_df["date"].unique())
        if len(_final_dates) > rolling_window:
            _final_start = pd.Timestamp(_final_dates[-rolling_window])
            train_df = train_df[train_df["date"] >= _final_start]
    val_df   = train_pool[
        (train_pool["date"] >= final_val_start) &
        (train_pool["date"] <= final_val_end)
    ]
    print(f">> Final model: train {len(train_df):,} rows  "
          f"val {len(val_df):,} rows  (<= {final_train_end.date()})"
          + (f"  [rolling={rolling_window}d]" if rolling_window else ""))

    final_cfg = dict(cfg)
    if best_iters:
        final_cfg["n_estimators"]          = int(np.mean(best_iters))
        final_cfg["early_stopping_rounds"] = None
    sw_final = time_decay_weights(train_df, half_life, weight_floor) if half_life else None
    model = mod.fit(mod.make_model(final_cfg), train_df, None, features, target_col, sw_final)

    # rolling backtest with the final model
    bt = rolling_backtest(
        panel, model,
        start_date = cfg.get("bt_start", "2026-03-01"),
        end_date   = cfg.get("bt_end",   "2026-04-20"),
        top_k      = top_k,
        features   = features,
        index_df   = index_df,
    )
    bt_port_mean  = float(bt["portfolio_return"].mean())
    bt_bench_mean = float(bt["bench_return"].mean())
    bt_excess     = bt["excess_return"].dropna()
    bt_mean       = float(bt_excess.mean())
    bt_std        = float(bt_excess.std())
    bt_sharpe     = bt_mean / bt_std if bt_std > 0 else float("nan")
    print(f">> Backtest  port={bt_port_mean:.4f}  bench={bt_bench_mean:.4f}"
          f"  excess={bt_mean:.4f}  std={bt_std:.4f}  sharpe={bt_sharpe:.3f}")

    # generate submission file
    pred_df   = prediction_frame(panel)
    pred_df   = pred_df.assign(score=model.predict(pred_df[features]))
    weights   = build_portfolio(pred_df.set_index("stock_code")["score"], top_k=top_k)
    pred_date = pred_df["date"].iloc[0]

    out_path = OUTPUT_DIR / "submissions" / f"{cfg['exp_name']}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"stock_code": weights.index, "weight": weights.values}).to_csv(
        out_path, index=False)
    print(f">> Submission -> {out_path}")

    # log experiment
    params = {
        "exp_name":         cfg["exp_name"],
        "model":            cfg["model"],
        "as_of":            pred_date.date().isoformat(),
        "train_end":        final_train_end.date().isoformat(),
        "n_estimators":     mod.n_estimators_used(model),
        "max_depth":        cfg.get("max_depth", ""),
        "learning_rate":    cfg.get("learning_rate", ""),
        "subsample":        cfg.get("subsample", ""),
        "colsample_bytree": cfg.get("colsample_bytree", ""),
        "min_child_weight": cfg.get("min_child_weight", ""),
        "reg_lambda":       cfg.get("reg_lambda", ""),
        "features":         features,
        "top_k":            top_k,
        "out_file":         str(out_path),
    }
    results = {
        "val_rank_ic":    round(mean_ic, 6),
        "val_ic_std":     round(std_ic, 6),
        "val_top20_ic":   round(mean_top20ic, 6),
        "val_top50_ic":   round(mean_top50ic, 6),
        "val_hit_rate":   round(mean_hr, 6),
        "n_folds":        len(splits),
        "bt_mean_return": round(bt_port_mean, 6),
        "bt_bench_mean":  round(bt_bench_mean, 6),
        "bt_mean_excess": round(bt_mean, 6),
        "bt_std_return":  round(bt_std, 6),
        "bt_sharpe":      round(bt_sharpe, 4),
    }
    exp_id = log_experiment(params, results, notes=notes)
    print(f">> Logged: {exp_id}  ->  outputs/experiments.csv\n")
    return {"exp_id": exp_id, **results}


# ── edit CONFIG here to define your experiment ────────────────────────────────

CONFIG = {
    "exp_name":   "exp_018_xgb_decay_amplitude_topk30",
    "model":      "xgboost",
    "target":     TARGET_COLUMN,        # raw 3-day forward return

    "features":       FEATURE_COLUMNS,
    "n_folds":        10,
    "half_life":      120,              # slow decay; recent ~6 months emphasized
    "weight_floor":   0.5,              # old data keeps 50% baseline weight
    "rolling_window": None,

    # XGBoost hyperparams (matches baseline_xgboost.py)
    "n_estimators":          400,
    "early_stopping_rounds": None,      # consistent capacity across folds, no val consumption
    "max_depth":             5,
    "learning_rate":         0.05,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "min_child_weight":      10,
    "reg_lambda":            1.0,

    "top_k":    30,
    "bt_start": "2026-04-01",
    "bt_end":   "2026-04-30",
}

# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--notes", default="", help="free-text note appended to experiments.csv")
    args = p.parse_args()
    run(CONFIG, notes=args.notes)
