"""Asymmetric Huber Loss with downside penalty for XGBoost.

Loss multiplier per sample:
  case1: residual <= 0  (underestimate)         -> M = 1
  case2: residual > 0,  y_true >= 0  (overest. winner)  -> M = alpha
  case3: residual > 0,  y_true <  0  (overest. loser)   -> M = alpha * (1 + beta*|y_true|)

Huber kernel:
  |r| <= delta  ->  0.5 * r^2            (quadratic, grad=r, hess=1)
  |r| >  delta  ->  delta*(|r|-0.5*delta) (linear,    grad=delta*sign(r), hess~eps)
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame
from portfolio import build_portfolio

DATA_DIR = Path('data')
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel       = build_features(prices)
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

TOP_K = 50
MAX_W = 0.10
TRAIN_CUTOFF = '2026-04-08'


# ── custom objective ──────────────────────────────────────────────
def make_asym_huber(alpha=1.5, beta=2.0, delta=0.03):
    """Return a custom XGBoost objective with asymmetric downside penalty."""
    def obj(y_pred: np.ndarray, dtrain: xgb.DMatrix):
        y_true   = dtrain.get_label()
        residual = y_pred - y_true          # positive = overestimate
        abs_r    = np.abs(residual)

        # Huber gradient and hessian
        in_quad = abs_r <= delta
        grad_h  = np.where(in_quad, residual, delta * np.sign(residual))
        hess_h  = np.where(in_quad, np.ones_like(residual), 1e-4 * np.ones_like(residual))

        # per-sample multiplier
        over       = residual > 0
        true_neg   = y_true < 0
        M          = np.ones(len(y_true))
        M[over & ~true_neg] = alpha
        M[over & true_neg]  = alpha * (1.0 + beta * np.abs(y_true[over & true_neg]))

        return (M * grad_h).astype(np.float32), (M * hess_h).astype(np.float32)
    return obj


# ── helpers ───────────────────────────────────────────────────────
def make_decay(df, hl=120, floor=0.5):
    ds    = np.sort(df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


def score_prop_w(scores_s):
    top = scores_s.nlargest(TOP_K)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()
    for _ in range(50):
        mask = w > MAX_W
        if not mask.any():
            break
        excess, w[mask] = (w[mask] - MAX_W).sum(), MAX_W
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)


# ── training data ─────────────────────────────────────────────────
train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
sw       = make_decay(train_df)
dtrain   = xgb.DMatrix(
    train_df[FEATURE_COLUMNS], label=train_df[TARGET_COLUMN], weight=sw)

BASE_PARAMS = dict(
    max_depth=5, eta=0.05, subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10, reg_lambda=1.0, tree_method='hist',
    nthread=-1, seed=42,
)
N_TREES = 400

# ── train regression baseline (exp_019 model) ────────────────────
print("Training baseline regression model...")
reg_params = dict(BASE_PARAMS, objective='reg:squarederror')
reg_bst    = xgb.train(reg_params, dtrain, num_boost_round=N_TREES, verbose_eval=False)
print(f"  Done. train rows: {len(train_df):,}")

# ── train asymmetric Huber models (grid) ─────────────────────────
configs = {
    'reg_sq (exp019)': reg_bst,
}
grid = [
    dict(alpha=1.5, beta=2.0, delta=0.03),   # base config
    dict(alpha=1.5, beta=2.0, delta=0.05),   # wider Huber band
    dict(alpha=2.0, beta=2.0, delta=0.03),   # stronger alpha
    dict(alpha=1.5, beta=3.0, delta=0.03),   # stronger beta
]
for cfg in grid:
    label = f"ah_a{cfg['alpha']}_b{cfg['beta']}_d{cfg['delta']}"
    print(f"Training {label} ...")
    obj = make_asym_huber(**cfg)
    bst = xgb.train(BASE_PARAMS, dtrain, num_boost_round=N_TREES,
                    obj=obj, verbose_eval=False)
    configs[label] = bst
print()

# ── April daily backtest (score_prop weighting for all) ──────────
april_days = [pd.Timestamp(d) for d in all_trading
              if pd.Timestamp(d).year == 2026 and pd.Timestamp(d).month == 4]

results = {k: [] for k in configs}

for sd in april_days:
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd      = pd.Timestamp(all_trading[si - 3])
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS).copy()
    if len(pred_df) < TOP_K:
        continue

    dtest   = xgb.DMatrix(pred_df[FEATURE_COLUMNS])
    buy_p   = panel_close.loc[bd]
    sell_p  = panel_close.loc[sd]
    bench   = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)

    act_all = (sell_p.reindex(pred_df['stock_code'])
               / buy_p.reindex(pred_df['stock_code']) - 1).values

    for name, bst in configs.items():
        raw_scores = bst.predict(dtest)
        scores_s   = pd.Series(raw_scores, index=pred_df['stock_code'].values)
        weights    = score_prop_w(scores_s)
        bp  = buy_p.reindex(weights.index)
        sp_ = sell_p.reindex(weights.index)
        valid = (~bp.isna()) & (~sp_.isna())
        w_v   = weights[valid] / weights[valid].sum()
        port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
        ic    = spearmanr(raw_scores, act_all)[0]
        results[name].append({'sell': sd, 'port': port,
                               'bench': bench, 'excess': port - bench, 'ic': ic})

dfs = {k: pd.DataFrame(v) for k, v in results.items()}

# ── per-day table ─────────────────────────────────────────────────
names = list(configs.keys())
hdr   = f"{'sell':^12}{'bench%':^8}"
for n in names:
    hdr += f"{n[:12]:^14}"
print(hdr)
print('-' * (20 + 14 * len(names)))

n_days = len(dfs[names[0]])
for i in range(n_days):
    sd  = dfs[names[0]].iloc[i]['sell']
    bm  = dfs[names[0]].iloc[i]['bench']
    row = f"{str(sd.date()):^12}{bm*100:^+8.2f}"
    for n in names:
        e = dfs[n].iloc[i]['excess'] * 100
        row += f"{e:^+14.3f}"
    print(row)

# ── summary ───────────────────────────────────────────────────────
print('\n' + '=' * (20 + 14 * len(names)))
print(f"{'metric':^20}", end='')
for n in names:
    print(f"{n[:13]:^14}", end='')
print()
print('-' * (20 + 14 * len(names)))

for label, fn in [
    ('mean_excess%',  lambda d: d['excess'].mean() * 100),
    ('std_excess%',   lambda d: d['excess'].std() * 100),
    ('sharpe',        lambda d: d['excess'].mean() / d['excess'].std()),
    ('win_rate',      lambda d: (d['excess'] > 0).mean()),
    ('max_win%',      lambda d: d['excess'].max() * 100),
    ('max_loss%',     lambda d: d['excess'].min() * 100),
    ('mean_IC',       lambda d: d['ic'].mean()),
]:
    print(f"  {label:^18}", end='')
    for n in names:
        print(f"{fn(dfs[n]):^+14.4f}", end='')
    print()

# ── downside analysis: how often does each model overestimate losers ──
print('\n=== 高估跌股频率分析 ===')
print('(预测>0 且实际<0 的比例，越低越好)')
for n, bst in configs.items():
    over_neg = 0; total = 0
    for sd in april_days:
        si = date_to_idx[sd]
        if si < 3: continue
        bd      = pd.Timestamp(all_trading[si - 3])
        pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
        if len(pred_df) < 10: continue
        dtest   = xgb.DMatrix(pred_df[FEATURE_COLUMNS])
        sell_p  = panel_close.loc[sd].reindex(pred_df['stock_code'])
        buy_p_  = panel_close.loc[bd].reindex(pred_df['stock_code'])
        act     = (sell_p / buy_p_ - 1).values
        preds   = bst.predict(dtest)
        mask    = ~np.isnan(act)
        over_neg += ((preds[mask] > 0) & (act[mask] < 0)).sum()
        total    += mask.sum()
    print(f"  {n:30s}: {over_neg/total:.4f} ({over_neg}/{total})")

# ── pairwise: best asym_huber vs exp_019 ─────────────────────────
best_name = max((n for n in names if n != 'reg_sq (exp019)'),
                key=lambda n: dfs[n]['excess'].mean())
ref_ex = dfs['reg_sq (exp019)']['excess'].values
alt_ex = dfs[best_name]['excess'].values
print(f'\n=== 最优 asym_huber ({best_name}) vs exp_019 ===')
print(f"  asym_huber 胜: {(alt_ex > ref_ex).sum()}/{n_days} 天")
print(f"  exp_019    胜: {(ref_ex > alt_ex).sum()}/{n_days} 天")
print(f"  平均差 (asym - exp019): {(alt_ex - ref_ex).mean()*100:+.4f}%")
