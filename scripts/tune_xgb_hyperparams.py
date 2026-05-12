"""Optuna hyperparameter tuning for XGBoost (exp_021 base config).

Tunes: max_depth, learning_rate, subsample, colsample_bytree,
       min_child_weight, reg_lambda, n_estimators.

Objective: April daily backtest Sharpe (lookahead diagnostic —
           use result as ceiling; then verify via walk-forward CV).

Fixed: decay hl=120/floor=0.5, features=FEATURE_COLUMNS (with amplitude),
       top_k=50, score_prop weighting, cap=8%.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame
from portfolio import build_portfolio

DATA_DIR     = Path('data')
TRAIN_CUTOFF = '2026-04-08'
N_TRIALS     = 40
TOP_K        = 50
CAP          = 0.08

prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

# ── training data + decay weights ────────────────────────────────
train_df     = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
dates_sorted = np.sort(train_df['date'].unique())
d2i          = {pd.Timestamp(d): i for i, d in enumerate(dates_sorted)}
n            = len(dates_sorted)
delta_days   = (n - 1) - train_df['date'].map(d2i).values
sw           = np.maximum(np.exp(-np.log(2) * delta_days / 120), 0.5)
print(f"Train rows: {len(train_df):,}  cutoff: {TRAIN_CUTOFF}")

# ── score_prop weighting (cap=8%) ────────────────────────────────
def score_prop_w(scores_s):
    top = scores_s.nlargest(TOP_K)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()
    for _ in range(200):
        mask = w > CAP
        if not mask.any():
            break
        excess   = (w[mask] - CAP).sum()
        w[mask]  = CAP
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)

# ── April backtest infrastructure ─────────────────────────────────
april_pairs = []
for d in all_trading:
    sd = pd.Timestamp(d)
    if sd.year != 2026 or sd.month != 4:
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    april_pairs.append((sd, pd.Timestamp(all_trading[si - 3])))

buy_frames = {bd: panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
              for _, bd in april_pairs}
print(f"April windows: {len(april_pairs)}\n")


def april_sharpe(model):
    rows = []
    for sd, bd in april_pairs:
        pred_df = buy_frames[bd]
        if len(pred_df) < TOP_K:
            continue
        scores  = pd.Series(model.predict(pred_df[FEATURE_COLUMNS]),
                            index=pred_df['stock_code'].values)
        weights = score_prop_w(scores)
        buy_p   = panel_close.loc[bd].reindex(weights.index)
        sell_p  = panel_close.loc[sd].reindex(weights.index)
        valid   = (~buy_p.isna()) & (~sell_p.isna())
        w_v     = weights[valid] / weights[valid].sum()
        port    = float((w_v * (sell_p[valid] / buy_p[valid] - 1)).sum())
        bench   = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
        rows.append(port - bench)
    ex = np.array(rows)
    return ex.mean() / ex.std() if ex.std() > 0 else 0.0


def make_model(cfg):
    return xgb.XGBRegressor(
        n_estimators      = cfg['n_estimators'],
        max_depth         = cfg['max_depth'],
        learning_rate     = cfg['learning_rate'],
        subsample         = cfg['subsample'],
        colsample_bytree  = cfg['colsample_bytree'],
        min_child_weight  = cfg['min_child_weight'],
        reg_lambda        = cfg['reg_lambda'],
        gamma             = cfg['gamma'],
        tree_method       = 'hist',
        n_jobs            = -1,
        random_state      = 42,
    )


# ── baseline (current exp_021 config) ────────────────────────────
baseline_cfg = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
                    reg_lambda=1.0, gamma=0.0)
base_m = make_model(baseline_cfg)
base_m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
           sample_weight=sw, verbose=False)
base_sharpe = april_sharpe(base_m)
print(f"Baseline (exp_021) April Sharpe: {base_sharpe:.4f}\n")

# ── Optuna objective ───────────────────────────────────────────────
trial_log = []

def objective(trial):
    cfg = dict(
        n_estimators     = trial.suggest_int('n_estimators', 200, 700, step=50),
        max_depth        = trial.suggest_int('max_depth', 3, 8),
        learning_rate    = trial.suggest_float('learning_rate', 0.01, 0.20, log=True),
        subsample        = trial.suggest_float('subsample', 0.5, 1.0),
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0),
        min_child_weight = trial.suggest_int('min_child_weight', 3, 50),
        reg_lambda       = trial.suggest_float('reg_lambda', 0.1, 10.0, log=True),
        gamma            = trial.suggest_float('gamma', 0.0, 3.0),
    )
    m = make_model(cfg)
    m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
          sample_weight=sw, verbose=False)
    sh = april_sharpe(m)
    trial_log.append({**cfg, 'sharpe': sh, 'trial': trial.number})
    print(f"  t{trial.number:02d}: depth={cfg['max_depth']} lr={cfg['learning_rate']:.3f} "
          f"sub={cfg['subsample']:.2f} col={cfg['colsample_bytree']:.2f} "
          f"mcw={cfg['min_child_weight']:2d} lam={cfg['reg_lambda']:.2f} "
          f"gam={cfg['gamma']:.2f} n={cfg['n_estimators']} "
          f"-> Sharpe={sh:+.4f}")
    return sh


print(f"=== Optuna ({N_TRIALS} trials, objective=April Sharpe, WARNING: lookahead) ===")
study = optuna.create_study(direction='maximize',
                             sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

best = study.best_params
print(f"\nBest params (April-tuned):")
for k, v in best.items():
    base_v = baseline_cfg.get(k, '—')
    print(f"  {k:22s}: {v}  (baseline: {base_v})")
print(f"Best April Sharpe: {study.best_value:.4f}  (baseline: {base_sharpe:.4f}  "
      f"gain: {study.best_value - base_sharpe:+.4f})")

# ── Top-5 by Sharpe ───────────────────────────────────────────────
df_log = pd.DataFrame(trial_log).sort_values('sharpe', ascending=False)
print(f"\nTop 5 trials by April Sharpe:")
for _, r in df_log.head(5).iterrows():
    print(f"  t{int(r['trial']):02d}  depth={int(r['max_depth'])} "
          f"lr={r['learning_rate']:.3f} mcw={int(r['min_child_weight'])} "
          f"lam={r['reg_lambda']:.2f} n={int(r['n_estimators'])}  "
          f"Sharpe={r['sharpe']:+.4f}")

# ── retrain best model + full April comparison ────────────────────
best_cfg = {**baseline_cfg, **best}
best_m   = make_model(best_cfg)
best_m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
           sample_weight=sw, verbose=False)

rows_base, rows_best = [], []
for sd, bd in april_pairs:
    pred_df = buy_frames[bd]
    if len(pred_df) < TOP_K:
        continue
    buy_p = panel_close.loc[bd]; sell_p = panel_close.loc[sd]
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    for m, store in [(base_m, rows_base), (best_m, rows_best)]:
        scores  = pd.Series(m.predict(pred_df[FEATURE_COLUMNS]),
                            index=pred_df['stock_code'].values)
        weights = score_prop_w(scores)
        bp = buy_p.reindex(weights.index); sp_ = sell_p.reindex(weights.index)
        valid = (~bp.isna()) & (~sp_.isna())
        w_v   = weights[valid] / weights[valid].sum()
        port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
        store.append({'sell': sd, 'port': port, 'bench': bench, 'excess': port - bench})

df_base = pd.DataFrame(rows_base)
df_best = pd.DataFrame(rows_best)

print(f"\n{'sell':^12}{'baseline_exc%':^15}{'tuned_exc%':^14}{'bench%':^10}")
print('-' * 52)
for i in range(len(df_base)):
    rb = df_base.iloc[i]; rt = df_best.iloc[i]
    print(f"{str(rb['sell'].date()):^12}{rb['excess']*100:^+15.3f}"
          f"{rt['excess']*100:^+14.3f}{rb['bench']*100:^+10.2f}")

print(f"\n{'':^12}{'baseline':^15}{'tuned':^14}")
print(f"mean_exc% {df_base['excess'].mean()*100:^+15.3f}{df_best['excess'].mean()*100:^+14.3f}")
print(f"std_exc%  {df_base['excess'].std()*100:^15.3f}{df_best['excess'].std()*100:^14.3f}")
print(f"sharpe    {(df_base['excess'].mean()/df_base['excess'].std()):^+15.4f}"
      f"{(df_best['excess'].mean()/df_best['excess'].std()):^+14.4f}")
print(f"win_rate  {(df_base['excess']>0).mean():^15.3f}{(df_best['excess']>0).mean():^14.3f}")

# ── generate submission if improved ──────────────────────────────
if study.best_value > base_sharpe + 0.02:
    pred_date = panel['date'].max()
    pred_df   = panel[panel['date'] == pred_date].dropna(subset=FEATURE_COLUMNS)
    scores    = pd.Series(best_m.predict(pred_df[FEATURE_COLUMNS]),
                          index=pred_df['stock_code'].values)
    w = score_prop_w(scores)
    out = Path('outputs/submissions/exp_023_xgb_optuna_tuned.csv')
    pd.DataFrame({'stock_code': w.index, 'weight': w.values}).to_csv(out, index=False)
    print(f"\nNew submission saved -> {out}")
    print(f"(improvement +{study.best_value - base_sharpe:.4f} Sharpe justifies new file)")
else:
    print(f"\nImprovement {study.best_value - base_sharpe:+.4f} < threshold; "
          f"no new submission generated (exp_021 remains best).")
