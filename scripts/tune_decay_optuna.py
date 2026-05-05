"""Optuna tune time-decay hyperparameters (half_life, floor) for baseline XGB.

Uses 3-fold walk-forward CV on pre-April data, optimizes top-50 alpha
(predicted top-50 mean return - all-stocks mean return).
Then runs daily April backtest comparing baseline / manual decay / Optuna decay.
"""
from __future__ import annotations
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
from pathlib import Path
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from features import FEATURE_COLUMNS, TARGET_COLUMN, build_features, training_frame
from portfolio import build_portfolio

DATA_DIR = Path('data')
TRAIN_CUTOFF = '2026-03-31'
N_FOLDS, VAL_DAYS, EMBARGO = 3, 10, 3
TUNE_TREES = 200
FINAL_TREES = 400
N_TRIALS = 25

# ── load + build features ───────────────────────────────────────────────────
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel = build_features(prices)

train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
print(f"Train rows: {len(train_df):,}  dates: {train_df.date.min().date()} ~ {train_df.date.max().date()}")

# ── helpers ─────────────────────────────────────────────────────────────────
def time_decay_floored(df, hl, floor):
    sd = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(sd)}
    n = len(sd)
    delta = (n - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

def make_xgb(n_est, seed=42):
    return xgb.XGBRegressor(
        n_estimators=n_est, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
        tree_method='hist', n_jobs=-1, random_state=seed)

def top_k_alpha(y_true, y_pred, dates, k=50):
    alphas = []
    for d in np.unique(dates):
        mask = dates == d
        if mask.sum() < k * 2:
            continue
        pred_d = y_pred[mask]; true_d = y_true[mask]
        top_idx = np.argpartition(pred_d, -k)[-k:]
        alphas.append(true_d[top_idx].mean() - true_d.mean())
    return float(np.mean(alphas))

# ── walk-forward folds for tuning ───────────────────────────────────────────
all_dates = np.sort(train_df['date'].unique())
splits = []
for i in range(N_FOLDS):
    val_end_idx   = len(all_dates) - 1 - i * (VAL_DAYS + EMBARGO)
    val_start_idx = val_end_idx - VAL_DAYS + 1
    train_end_idx = val_start_idx - EMBARGO - 1
    if train_end_idx < 60:
        break
    splits.append((all_dates[train_end_idx], all_dates[val_start_idx], all_dates[val_end_idx]))
splits = list(reversed(splits))
print(f"CV folds: {len(splits)}")
for i, (te, vs, ve) in enumerate(splits):
    print(f"  fold {i+1}: train<={pd.Timestamp(te).date()}  val=[{pd.Timestamp(vs).date()}, {pd.Timestamp(ve).date()}]")

# ── Optuna objective ────────────────────────────────────────────────────────
def objective(trial):
    hl    = trial.suggest_int('half_life', 30, 365)
    floor = trial.suggest_float('floor', 0.0, 0.95)
    fold_alphas = []
    for tr_end, val_start, val_end in splits:
        f_tr  = train_df[train_df['date'] <= tr_end]
        f_val = train_df[(train_df['date'] >= val_start) & (train_df['date'] <= val_end)]
        sw = time_decay_floored(f_tr, hl, floor)
        m = make_xgb(n_est=TUNE_TREES)
        m.fit(f_tr[FEATURE_COLUMNS], f_tr[TARGET_COLUMN], sample_weight=sw, verbose=False)
        pred = m.predict(f_val[FEATURE_COLUMNS])
        a = top_k_alpha(f_val[TARGET_COLUMN].values, pred, f_val['date'].values, k=50)
        fold_alphas.append(a)
    return float(np.mean(fold_alphas))

print(f"\n=== Optuna search (TPE, {N_TRIALS} trials, {TUNE_TREES} trees per fit) ===")
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_hl    = study.best_params['half_life']
best_floor = study.best_params['floor']
print(f"\nBest params: half_life={best_hl}d  floor={best_floor:.4f}")
print(f"Best CV top-50 alpha: {study.best_value:+.4f}")

print("\nTop 5 trials:")
df_trials = study.trials_dataframe().sort_values('value', ascending=False).head(5)
for _, r in df_trials.iterrows():
    print(f"  hl={int(r['params_half_life']):3d}d  floor={r['params_floor']:.3f}  alpha={r['value']:+.4f}")

# ── retrain final models on full pre-April data ─────────────────────────────
print(f"\nRetraining models with {FINAL_TREES} trees on full pre-April data...")
sw_best   = time_decay_floored(train_df, best_hl, best_floor)
sw_manual = time_decay_floored(train_df, 120, 0.5)

base_model    = make_xgb(n_est=FINAL_TREES)
base_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], verbose=False)

manual_model  = make_xgb(n_est=FINAL_TREES)
manual_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=sw_manual, verbose=False)

optuna_model  = make_xgb(n_est=FINAL_TREES)
optuna_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=sw_best, verbose=False)

# ── daily April backtest ────────────────────────────────────────────────────
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

def daily_bt(predict_fn, top_k=50):
    rows = []
    for d in all_trading:
        sd = pd.Timestamp(d)
        if sd.month != 4 or sd.year != 2026:
            continue
        sell_idx = date_to_idx[sd]; buy_idx = sell_idx - 3
        if buy_idx < 0:
            continue
        bd = pd.Timestamp(all_trading[buy_idx])
        pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
        if len(pred_df) < top_k:
            continue
        scores  = pd.Series(predict_fn(pred_df[FEATURE_COLUMNS]), index=pred_df['stock_code'].values)
        weights = build_portfolio(scores, top_k=top_k)
        buy_p   = panel_close.loc[bd].reindex(weights.index)
        sell_p  = panel_close.loc[sd].reindex(weights.index)
        valid   = (~buy_p.isna()) & (~sell_p.isna())
        w_v = weights[valid] / weights[valid].sum()
        port_ret  = (w_v * (sell_p[valid] / buy_p[valid] - 1)).sum()
        bench_ret = idx_close.loc[sd] / idx_close.loc[bd] - 1
        rows.append({'sell_date': sd, 'buy_date': bd,
                     'port_ret': float(port_ret), 'bench_ret': float(bench_ret),
                     'excess':   float(port_ret - bench_ret)})
    return pd.DataFrame(rows)

bt_base   = daily_bt(base_model.predict)
bt_manual = daily_bt(manual_model.predict)
bt_optuna = daily_bt(optuna_model.predict)

print()
header = f"{'sell':<12}{'buy':<12}{'baseline':>10}{'manual':>10}{'optuna':>10}{'CSI500':>10}{'opt-base':>11}"
print(header)
for i in range(len(bt_base)):
    rb = bt_base.iloc[i]; rm = bt_manual.iloc[i]; ro = bt_optuna.iloc[i]
    sd = rb['sell_date'].date(); bd = rb['buy_date'].date()
    diff = ro['port_ret'] - rb['port_ret']
    print(f"{str(sd):<12}{str(bd):<12}{rb['port_ret']:+10.4f}{rm['port_ret']:+10.4f}"
          f"{ro['port_ret']:+10.4f}{rb['bench_ret']:+10.4f}{diff:+11.4f}")

print()
print('=== April 2026 daily 3-day forward returns ===')
for label, bt in [
    ('Baseline XGB (no decay)               ', bt_base),
    ('Manual decay (hl=120, floor=0.500)    ', bt_manual),
    (f'Optuna decay (hl={best_hl}, floor={best_floor:.3f})  ', bt_optuna),
]:
    pm = bt['port_ret'].mean(); ex = bt['excess']
    sharpe = ex.mean() / ex.std() if ex.std() > 0 else float('nan')
    print(f"{label}  N={len(bt):2d}  mean_port={pm:+.4f}  "
          f"mean_excess={ex.mean():+.4f}  std={ex.std():.4f}  "
          f"win={(ex>0).mean():.2f}  sharpe={sharpe:.3f}")

print()
print('=== Pairwise vs baseline ===')
for label, bt in [('Manual vs baseline ', bt_manual),
                  ('Optuna vs baseline ', bt_optuna)]:
    diff = bt['port_ret'] - bt_base['port_ret']
    t = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else float('nan')
    print(f"{label}  mean_diff={diff.mean():+.4f}  "
          f"wins={(diff > 0).sum()}/{len(diff)}  t_stat={t:.2f}")
