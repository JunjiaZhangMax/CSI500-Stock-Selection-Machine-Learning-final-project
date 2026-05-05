"""Optuna tune time-decay (half_life, floor) DIRECTLY on April daily backtest.

WARNING: This is lookahead bias — tuning on the test set. Use only as an
upper-bound diagnostic: how good can pure time-decay possibly get in April?

Compares the April-overfit best params against:
  - Baseline (no decay)
  - Manual (hl=120, floor=0.5)
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
N_TREES = 400
N_TRIALS = 30

# ── load + build features ───────────────────────────────────────────────────
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel = build_features(prices)

train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
print(f"Train rows: {len(train_df):,}  cutoff: {TRAIN_CUTOFF}")

# ── decay weights ───────────────────────────────────────────────────────────
def time_decay_floored(df, hl, floor):
    sd = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(sd)}
    n = len(sd)
    delta = (n - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

def make_xgb(seed=42):
    return xgb.XGBRegressor(
        n_estimators=N_TREES, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10, reg_lambda=1.0,
        tree_method='hist', n_jobs=-1, random_state=seed)

# ── precompute April backtest infrastructure ────────────────────────────────
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

# Pre-extract April sell/buy date pairs (only computed once)
april_pairs = []
for d in all_trading:
    sd = pd.Timestamp(d)
    if sd.month != 4 or sd.year != 2026: continue
    sell_idx = date_to_idx[sd]; buy_idx = sell_idx - 3
    if buy_idx < 0: continue
    bd = pd.Timestamp(all_trading[buy_idx])
    april_pairs.append((sd, bd))
print(f"April backtest windows: {len(april_pairs)}")

# Pre-cache prediction frames (panel rows for each April buy date)
buy_pred_frames = {bd: panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
                   for _, bd in april_pairs}

def april_daily_bt(predict_fn, top_k=50):
    rows = []
    for sd, bd in april_pairs:
        pred_df = buy_pred_frames[bd]
        if len(pred_df) < top_k: continue
        scores  = pd.Series(predict_fn(pred_df[FEATURE_COLUMNS]),
                            index=pred_df['stock_code'].values)
        weights = build_portfolio(scores, top_k=top_k)
        buy_p   = panel_close.loc[bd].reindex(weights.index)
        sell_p  = panel_close.loc[sd].reindex(weights.index)
        valid   = (~buy_p.isna()) & (~sell_p.isna())
        w_v = weights[valid] / weights[valid].sum()
        port_ret  = float((w_v * (sell_p[valid]/buy_p[valid] - 1)).sum())
        bench_ret = float(idx_close.loc[sd]/idx_close.loc[bd] - 1)
        rows.append({'sell_date': sd, 'buy_date': bd,
                     'port_ret': port_ret, 'bench_ret': bench_ret,
                     'excess':   port_ret - bench_ret})
    return pd.DataFrame(rows)

# ── Optuna objective: maximize April Sharpe ─────────────────────────────────
trial_log = []

def objective(trial):
    hl    = trial.suggest_int('half_life', 30, 365)
    floor = trial.suggest_float('floor', 0.0, 0.95)
    sw = time_decay_floored(train_df, hl, floor)
    m = make_xgb()
    m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=sw, verbose=False)
    bt = april_daily_bt(m.predict)
    ex = bt['excess']
    sharpe = ex.mean() / ex.std() if ex.std() > 0 else 0.0
    trial_log.append({'trial': trial.number, 'hl': hl, 'floor': floor,
                       'mean_ex': ex.mean(), 'std_ex': ex.std(), 'sharpe': sharpe,
                       'win_rate': (ex > 0).mean()})
    print(f"  trial {trial.number:2d}: hl={hl:3d}d  floor={floor:.3f}  "
          f"mean_ex={ex.mean():+.4f}  sharpe={sharpe:+.3f}  win={(ex > 0).mean():.2f}")
    return sharpe

print(f"\n=== Optuna (TPE, {N_TRIALS} trials, target = April Sharpe) ===")
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best_hl    = study.best_params['half_life']
best_floor = study.best_params['floor']
print(f"\nBest April params: hl={best_hl}d  floor={best_floor:.4f}  Sharpe={study.best_value:.3f}")

# Top 5 by Sharpe
print("\nTop 5 trials by Sharpe:")
df_log = pd.DataFrame(trial_log).sort_values('sharpe', ascending=False).head(5)
for _, r in df_log.iterrows():
    print(f"  hl={int(r['hl']):3d}d  floor={r['floor']:.3f}  "
          f"mean_ex={r['mean_ex']:+.4f}  sharpe={r['sharpe']:+.3f}  win={r['win_rate']:.2f}")

# Top 5 by mean excess
print("\nTop 5 trials by mean_excess:")
df_log_ex = pd.DataFrame(trial_log).sort_values('mean_ex', ascending=False).head(5)
for _, r in df_log_ex.iterrows():
    print(f"  hl={int(r['hl']):3d}d  floor={r['floor']:.3f}  "
          f"mean_ex={r['mean_ex']:+.4f}  sharpe={r['sharpe']:+.3f}  win={r['win_rate']:.2f}")

# ── Side-by-side comparison ──────────────────────────────────────────────────
print(f"\n=== Compare baseline / manual / April-tuned ===")

base_model = make_xgb()
base_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], verbose=False)

manual_sw = time_decay_floored(train_df, 120, 0.5)
manual_model = make_xgb()
manual_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=manual_sw, verbose=False)

best_sw = time_decay_floored(train_df, best_hl, best_floor)
best_model = make_xgb()
best_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=best_sw, verbose=False)

bt_base   = april_daily_bt(base_model.predict)
bt_manual = april_daily_bt(manual_model.predict)
bt_tuned  = april_daily_bt(best_model.predict)

print()
header = f"{'sell':<12}{'buy':<12}{'baseline':>10}{'manual':>10}{'tuned':>10}{'CSI500':>10}"
print(header)
for i in range(len(bt_base)):
    rb = bt_base.iloc[i]; rm = bt_manual.iloc[i]; rt = bt_tuned.iloc[i]
    sd = rb['sell_date'].date(); bd = rb['buy_date'].date()
    print(f"{str(sd):<12}{str(bd):<12}{rb['port_ret']:+10.4f}{rm['port_ret']:+10.4f}"
          f"{rt['port_ret']:+10.4f}{rb['bench_ret']:+10.4f}")

print()
print('=== Summary (April daily 3-day forward returns) ===')
for label, bt in [
    ('Baseline XGB (no decay)                  ', bt_base),
    ('Manual decay (hl=120, floor=0.500)       ', bt_manual),
    (f'April-tuned (hl={best_hl}, floor={best_floor:.3f})  *overfit*', bt_tuned),
]:
    pm = bt['port_ret'].mean(); ex = bt['excess']
    sharpe = ex.mean() / ex.std() if ex.std() > 0 else float('nan')
    print(f"{label}  N={len(bt):2d}  mean_port={pm:+.4f}  "
          f"mean_excess={ex.mean():+.4f}  std={ex.std():.4f}  "
          f"win={(ex > 0).mean():.2f}  sharpe={sharpe:.3f}")

print()
print('=== Pairwise vs baseline ===')
for label, bt in [('Manual vs baseline       ', bt_manual),
                  ('April-tuned vs baseline  ', bt_tuned)]:
    diff = bt['port_ret'] - bt_base['port_ret']
    t = diff.mean() / (diff.std() / np.sqrt(len(diff))) if diff.std() > 0 else float('nan')
    print(f"{label}  mean_diff={diff.mean():+.4f}  "
          f"wins={(diff > 0).sum()}/{len(diff)}  t_stat={t:.2f}")
