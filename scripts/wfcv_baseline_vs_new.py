"""Walk-Forward Cross-Validation: Original Baseline vs New Model.

Baseline (exp_001 style):
  - 14 features (no amplitude)
  - No time-decay (uniform sample weights)
  - top_k=50, rank-linear portfolio weighting, cap=10%
  - 3d forward return target

New Model (window2 config):
  - 16 features (+ amplitude_ma_20d / _rank)
  - Time-decay hl=60, floor=0.5
  - top_k=30, score-proportional weighting, cap=10%
  - 3d forward return target

Walk-forward: monthly retrain, daily 3d-hold evaluation, Oct 2025 - May 2026.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame

DATA_DIR = Path('data')

FEAT_BASE = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d', 'ret_60d',
    'vol_20d', 'volume_z_20d', 'turnover_ma_20d',
    'close_over_ma20', 'close_over_ma60', 'rsi_14',
    'ret_5d_rank', 'ret_20d_rank', 'vol_20d_rank',
]
FEAT_NEW = FEATURE_COLUMNS   # 16 features including amplitude

print("Loading data...")
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)

panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

# ── Portfolio construction helpers ────────────────────────────────

def rank_linear_w(scores_s, top_k=50, cap=0.10):
    """Rank-proportional weights (baseline style)."""
    top = scores_s.nlargest(top_k)
    ranks = np.arange(top_k, 0, -1, dtype=float)   # top rank = top_k
    w = ranks / ranks.sum()
    for _ in range(200):
        mask = w > cap
        if not mask.any():
            break
        excess   = (w[mask] - cap).sum()
        w[mask]  = cap
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)


def score_prop_w(scores_s, top_k=30, cap=0.10):
    """Score-proportional weights (new model style)."""
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()
    for _ in range(200):
        mask = w > cap
        if not mask.any():
            break
        excess   = (w[mask] - cap).sum()
        w[mask]  = cap
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)


def eval_port(weights, sd, bd):
    bp  = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    if valid.sum() == 0:
        return np.nan
    w_v = weights[valid] / weights[valid].sum()
    return float((w_v * (sp_[valid] / bp[valid] - 1)).sum())


def make_decay(df, hl, floor=0.5):
    ds    = np.sort(df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)


# ── Walk-forward setup ────────────────────────────────────────────
start_eval   = pd.Timestamp('2025-10-01')
end_eval     = pd.Timestamp('2026-05-08')
eval_dates   = [pd.Timestamp(d) for d in all_trading
                if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}   # ms -> (xgb_base, xgb_new)
print("Training monthly models (baseline + new)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr) < 5000:
        continue

    # Baseline: no decay
    tr_b = tr.dropna(subset=FEAT_BASE + [TARGET_COLUMN])
    xb   = make_xgb()
    xb.fit(tr_b[FEAT_BASE].values, tr_b[TARGET_COLUMN].values, verbose=False)

    # New: decay hl=60
    tr_n = tr.dropna(subset=FEAT_NEW + [TARGET_COLUMN])
    sw_n = make_decay(tr_n, hl=60)
    xn   = make_xgb()
    xn.fit(tr_n[FEAT_NEW].values, tr_n[TARGET_COLUMN].values,
           sample_weight=sw_n, verbose=False)

    model_cache[ms] = (xb, xn, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  "
          f"base_rows={len(tr_b):,}  new_rows={len(tr_n):,}")


def get_models(sell_date):
    cands = [ms for ms in model_cache if ms <= sell_date]
    return model_cache[max(cands)] if cands else None


# ── Daily evaluation ──────────────────────────────────────────────
records = []
for d in all_trading:
    sd = pd.Timestamp(d)
    if not (start_eval <= sd <= end_eval):
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd   = pd.Timestamp(all_trading[si - 3])
    mdls = get_models(sd)
    if mdls is None:
        continue
    xb, xn, _ = mdls

    pred_b = panel[panel['date'] == bd].dropna(subset=FEAT_BASE)
    pred_n = panel[panel['date'] == bd].dropna(subset=FEAT_NEW)
    if len(pred_b) < 50 or len(pred_n) < 30:
        continue

    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)

    # Baseline portfolio
    sc_b   = pd.Series(xb.predict(pred_b[FEAT_BASE].values),
                       index=pred_b['stock_code'].values)
    w_b    = rank_linear_w(sc_b, top_k=50, cap=0.10)
    port_b = eval_port(w_b, sd, bd)

    # IC: use sell-date panel where target is realised (not buy-date)
    pred_s_b = panel[panel['date'] == sd].dropna(subset=FEAT_BASE + [TARGET_COLUMN])
    if len(pred_s_b) > 10:
        sc_s_b = pd.Series(xb.predict(pred_s_b[FEAT_BASE].values),
                           index=pred_s_b['stock_code'].values)
        ic_b   = float(spearmanr(sc_s_b, pred_s_b[TARGET_COLUMN])[0])
    else:
        ic_b   = np.nan

    # New model portfolio
    sc_n   = pd.Series(xn.predict(pred_n[FEAT_NEW].values),
                       index=pred_n['stock_code'].values)
    w_n    = score_prop_w(sc_n, top_k=30, cap=0.10)
    port_n = eval_port(w_n, sd, bd)

    pred_s_n = panel[panel['date'] == sd].dropna(subset=FEAT_NEW + [TARGET_COLUMN])
    if len(pred_s_n) > 10:
        sc_s_n = pd.Series(xn.predict(pred_s_n[FEAT_NEW].values),
                           index=pred_s_n['stock_code'].values)
        ic_n   = float(spearmanr(sc_s_n, pred_s_n[TARGET_COLUMN])[0])
    else:
        ic_n   = np.nan

    if port_b is None or port_n is None or np.isnan(port_b) or np.isnan(port_n):
        continue

    records.append({
        'sell':     sd,
        'month':    sd.strftime('%Y-%m'),
        'bench':    bench,
        'port_b':   port_b,
        'excess_b': port_b - bench,
        'ic_b':     ic_b,
        'port_n':   port_n,
        'excess_n': port_n - bench,
        'ic_n':     ic_n,
    })

df = pd.DataFrame(records)

# ── Report ────────────────────────────────────────────────────────
SEP = '=' * 80
print()
print(SEP)
print("  Walk-Forward Results  Oct 2025 – May 2026  (monthly retrain, 3d hold)")
print(SEP)
print(f"{'month':^10}{'N':^5}{'baseline_exc%':^16}{'newmodel_exc%':^16}"
      f"{'delta_exc%':^13}{'bench%':^10}")
print('-' * 60)

for mo in sorted(df['month'].unique()):
    sub = df[df['month'] == mo]
    n   = len(sub)
    eb  = sub['excess_b'].mean() * 100
    en  = sub['excess_n'].mean() * 100
    bch = sub['bench'].mean()    * 100
    print(f"{mo:^10}{n:^5}{eb:^+16.3f}{en:^+16.3f}{en-eb:^+13.3f}{bch:^+10.3f}")

print('-' * 60)
metrics = [
    ('mean_exc%',  lambda c: df[c].mean()   * 100),
    ('std_exc%',   lambda c: df[c].std()    * 100),
    ('sharpe',     lambda c: df[c].mean()   / df[c].std()),
    ('win_rate',   lambda c: (df[c] > 0).mean()),
    ('port_mean%', lambda c: df[c.replace('excess','port')].mean() * 100),
    ('mean_IC',    lambda c: df[c.replace('excess','ic')].mean()),
]
print(f"\n{'metric':^14}{'baseline':^18}{'new_model':^18}{'improvement':^15}")
print('-' * 65)
for label, fn in metrics:
    vb = fn('excess_b')
    vn = fn('excess_n')
    delta_str = f"{vn-vb:+.4f}" if label not in ('win_rate','mean_IC') else f"{vn-vb:+.4f}"
    print(f"  {label:^12}  {vb:^+18.4f}{vn:^+18.4f}{delta_str:^15}")

# ── Regime-conditional breakdown ──────────────────────────────────
print()
print(SEP)
print("  Performance by Market Regime (index 20d log-return)")
print(SEP)
idx_log = np.log(idx_close)
idx_log = np.log(idx_close)
def _idx_ret20(sd):
    if sd not in idx_log.index:
        return np.nan
    loc = idx_log.index.get_loc(sd)
    if loc < 20:
        return np.nan
    return float(idx_log.iloc[loc] - idx_log.iloc[loc - 20])
df['idx_ret_20d'] = df['sell'].map(_idx_ret20).astype(float)

for label, lo, hi in [
    ('bull  (>+2%)',   0.02,  np.inf),
    ('neutral (+-2%)', -0.02,  0.02),
    ('bear  (<-2%)',  -np.inf, -0.02),
]:
    r = df['idx_ret_20d'].astype(float)
    if lo == -np.inf:
        mask = r <= hi
    elif hi == np.inf:
        mask = r > lo
    else:
        mask = (r > lo) & (r <= hi)
    sub  = df[mask]
    n    = len(sub)
    if n == 0:
        continue
    eb = sub['excess_b'].mean() * 100
    en = sub['excess_n'].mean() * 100
    print(f"  {label:^18}  N={n:3d}  baseline={eb:^+8.3f}%  "
          f"new={en:^+8.3f}%  delta={en-eb:^+7.3f}%")

# ── Drawdown / tail risk ──────────────────────────────────────────
print()
print(SEP)
print("  Tail Risk & Drawdown")
print(SEP)
for col, lbl in [('excess_b','baseline'), ('excess_n','new_model')]:
    s    = df[col] * 100
    worst5 = s.nsmallest(5).values
    print(f"  {lbl}:")
    print(f"    worst 5 days: {worst5.round(3)}")
    print(f"    max_loss: {s.min():+.3f}%   "
          f"pct_days<-1%: {(s<-1).mean()*100:.1f}%   "
          f"pct_days>+1%: {(s>1).mean()*100:.1f}%")

# ── IC stability ─────────────────────────────────────────────────
print()
print(SEP)
print("  Rank IC Stability (cross-sectional, daily)")
print(SEP)
for col, lbl in [('ic_b','baseline'), ('ic_n','new_model')]:
    ic = df[col]
    print(f"  {lbl}:  mean={ic.mean():+.4f}  std={ic.std():.4f}  "
          f"IR={ic.mean()/ic.std():+.3f}  "
          f"pct_positive={( ic>0).mean()*100:.1f}%")

# ── Methodology summary table ─────────────────────────────────────
print()
print(SEP)
print("  Methodology Comparison Summary")
print(SEP)
rows = [
    ("Target horizon",      "5-day return (original)", "3-day return"),
    ("Features",            "14 (no amplitude)",       "16 (+amplitude_ma_20d, _rank)"),
    ("Sample weights",      "Uniform (no decay)",      "Time-decay hl=60, floor=0.5"),
    ("Portfolio weighting", "Rank-linear",             "Score-proportional (iterative)"),
    ("Holdings (top_k)",    "50",                      "30"),
    ("Weight cap",          "10%",                     "10%"),
    ("Model",               "XGBoost (same)",          "XGBoost (same)"),
    ("Training cutoff",     "Fixed split",             "Expanding window (monthly)"),
]
print(f"  {'Dimension':^28}{'Baseline':^32}{'New Model':^32}")
print('  ' + '-' * 92)
for dim, base, new in rows:
    print(f"  {dim:^28}{base:^32}{new:^32}")
