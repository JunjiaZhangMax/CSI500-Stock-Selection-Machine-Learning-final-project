"""Walk-Forward validation: baseline vs new-feature configs.

Configs compared:
  A  baseline          FEATURE_COLUMNS, top50, hl=120, XGB only
  D  topk30_hl60       FEAT_NEW,        top30, hl=60,  XGB only
  E  topk30_ridge50    FEAT_NEW,        top30, hl=60,  XGB50/Ridge50

Walk-forward: monthly retrain, daily evaluation Oct 2025 - May 2026.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame

DATA_DIR = Path('data')

# ── Data + feature engineering ────────────────────────────────────
print("Loading data and building features...")
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)
idx_close = index_df.set_index('date')['close']
idx_log   = np.log(idx_close)
idx_r1    = idx_log.diff(1)

panel = panel.join(idx_log.diff(5).rename('idx_ret_5d'),  on='date')
panel = panel.join(idx_log.diff(20).rename('idx_ret_20d'), on='date')
panel['rel_ret_5d']  = panel['ret_5d']  - panel['idx_ret_5d']
panel['rel_ret_20d'] = panel['ret_20d'] - panel['idx_ret_20d']

print("Computing rolling beta (60d)...")
stock_r1 = panel.pivot_table(index='date', columns='stock_code', values='ret_1d')
idx_r1_a = idx_r1.reindex(stock_r1.index)
betas = {}
for code in stock_r1.columns:
    cov = stock_r1[code].rolling(60).cov(idx_r1_a)
    var = idx_r1_a.rolling(60).var()
    betas[code] = cov / var.replace(0, np.nan)
beta_df = pd.DataFrame(betas).stack().reset_index()
beta_df.columns = ['date', 'stock_code', 'beta_60d']
panel = panel.merge(beta_df, on=['date', 'stock_code'], how='left')

FEAT_NEW = FEATURE_COLUMNS + ['rel_ret_5d', 'rel_ret_20d', 'beta_60d']

panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

# ── Helpers ───────────────────────────────────────────────────────
def make_decay(df, hl=120, floor=0.5):
    ds    = np.sort(df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


def score_prop_w(scores_s, top_k=50, cap=0.08):
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


def eval_window(scores, sd, bd, top_k=50, cap=0.08):
    weights = score_prop_w(scores, top_k, cap)
    bp  = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    if valid.sum() == 0:
        return 0.0, float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    w_v   = weights[valid] / weights[valid].sum()
    port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    return port, bench


def _norm(s):
    s = s - s.mean()
    std = s.std()
    return s / std if std > 0 else s


def train_models(tr, cfg):
    feats = cfg['feats']
    df    = tr.dropna(subset=feats + [TARGET_COLUMN])
    sw    = make_decay(df, hl=cfg['hl'])
    X, y  = df[feats].values, df[TARGET_COLUMN].values
    xm = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)
    xm.fit(X, y, sample_weight=sw, verbose=False)
    rm = None
    if cfg['ridge'] > 0:
        rm = Pipeline([('sc', StandardScaler()), ('r', Ridge(1.0))])
        rm.fit(X, y, r__sample_weight=sw)
    return xm, rm


def predict_scores(xm, rm, pred_df, cfg):
    feats = cfg['feats']
    sub   = pred_df.dropna(subset=feats)
    if len(sub) < cfg['top_k']:
        return None
    sx = _norm(xm.predict(sub[feats].values))
    if rm is not None:
        sr = _norm(rm.predict(sub[feats].values))
        sc = pd.Series((1 - cfg['ridge']) * sx + cfg['ridge'] * sr,
                       index=sub['stock_code'].values)
    else:
        sc = pd.Series(sx, index=sub['stock_code'].values)
    return sc


# ── Config definitions ────────────────────────────────────────────
CONFIGS = [
    {'name': 'A_baseline',       'feats': FEATURE_COLUMNS, 'top_k': 50, 'hl': 120, 'ridge': 0.0},
    {'name': 'D_topk30_hl60',    'feats': FEAT_NEW,        'top_k': 30, 'hl':  60, 'ridge': 0.0},
    {'name': 'E_topk30_ridge50', 'feats': FEAT_NEW,        'top_k': 30, 'hl':  60, 'ridge': 0.5},
]

# ── Walk-forward monthly retrain ──────────────────────────────────
start_eval   = pd.Timestamp('2025-10-01')
end_eval     = pd.Timestamp('2026-05-08')
eval_dates   = [pd.Timestamp(d) for d in all_trading
                if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training monthly walk-forward models...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr) < 5000:
        continue
    month_models = {}
    for cfg in CONFIGS:
        xm, rm = train_models(tr, cfg)
        month_models[cfg['name']] = (xm, rm)
    model_cache[ms] = (month_models, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr):,}")


def get_models(sell_date):
    cands = [ms for ms in model_cache if ms <= sell_date]
    return model_cache[max(cands)] if cands else None


# ── Daily evaluation ──────────────────────────────────────────────
wf = {cfg['name']: [] for cfg in CONFIGS}

for d in all_trading:
    sd = pd.Timestamp(d)
    if not (start_eval <= sd <= end_eval):
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd      = pd.Timestamp(all_trading[si - 3])
    mdls    = get_models(sd)
    if mdls is None:
        continue
    month_models, _ = mdls
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < 30:
        continue

    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    for cfg in CONFIGS:
        name    = cfg['name']
        xm, rm  = month_models[name]
        sc      = predict_scores(xm, rm, pred_df, cfg)
        if sc is None:
            continue
        port, _ = eval_window(sc, sd, bd, cfg['top_k'])
        wf[name].append({
            'sell':   sd,
            'month':  sd.strftime('%Y-%m'),
            'port':   port,
            'bench':  bench,
            'excess': port - bench,
        })

wfs = {k: pd.DataFrame(v) for k, v in wf.items()}

# ── Report ────────────────────────────────────────────────────────
ref_col = 'A_baseline'
col_w   = 18
names   = [cfg['name'] for cfg in CONFIGS]
labels  = ['baseline', 'D:new+top30+hl60', 'E:+Ridge50']

print()
print("=" * (10 + 5 + col_w * len(CONFIGS)))
print("Walk-Forward  Oct 2025 - May 2026  (monthly retrain, 3d hold)")
print("=" * (10 + 5 + col_w * len(CONFIGS)))

header = f"{'month':^10}{'N':^5}"
for lbl in labels:
    header += f"{lbl+'_ex%':^{col_w}}"
header += f"{'bench%':^10}  r_mean"
print(header)
print('-' * len(header))

for mo in sorted(wfs[ref_col]['month'].unique()):
    n    = (wfs[ref_col]['month'] == mo).sum()
    row  = f"{mo:^10}{n:^5}"
    for name in names:
        sub = wfs[name][wfs[name]['month'] == mo]
        row += f"{sub['excess'].mean()*100:^+{col_w}.3f}"
    ref_sub = wfs[ref_col][wfs[ref_col]['month'] == mo]
    bench_m = ref_sub['bench'].mean() * 100
    r_m     = 0.5   # placeholder; regime not recomputed here
    row += f"{bench_m:^+10.3f}"
    print(row)

print('-' * len(header))
for label, fn in [
    ('mean_exc%',  lambda d: d['excess'].mean() * 100),
    ('std_exc%',   lambda d: d['excess'].std()  * 100),
    ('sharpe',     lambda d: d['excess'].mean() / d['excess'].std()),
    ('win_rate',   lambda d: (d['excess'] > 0).mean()),
    ('port_mean%', lambda d: d['port'].mean() * 100),
]:
    row = f"  {label:^8}   {'':^5}"
    for name in names:
        row += f"{fn(wfs[name]):^+{col_w}.4f}"
    print(row)

# ── Regime-conditional breakdown ─────────────────────────────────
print()
print("Performance by market regime (using index 20d return as proxy):")
ref = wfs[ref_col].copy()
ref['idx_ret_20d'] = ref['sell'].map(
    lambda d: float(idx_log.get(d, np.nan) - idx_log.shift(20).get(d, np.nan))
    if d in idx_log.index else np.nan
)

# compute regime properly
idx_log_s = idx_log.copy()
for name_col, mask_fn, desc in [
    ('bull (idx_ret20>2%)',  lambda r: r > 0.02,  'bull'),
    ('neutral (-2% to 2%)', lambda r: (r >= -0.02) & (r <= 0.02), 'neutral'),
    ('bear (idx_ret20<-2%)', lambda r: r < -0.02, 'bear'),
]:
    # compute idx_ret_20d for each sell date
    sell_dates = wfs[ref_col]['sell']
    idx_20d = {}
    for sd in sell_dates:
        if sd in idx_log.index:
            past = idx_log[idx_log.index <= sd]
            if len(past) >= 21:
                idx_20d[sd] = float(past.iloc[-1] - past.iloc[-21])
            else:
                idx_20d[sd] = 0.0

    print(f"\n  {name_col}:", end='')
    for name, lbl in zip(names, labels):
        sub = wfs[name].copy()
        sub['idx_20d'] = sub['sell'].map(idx_20d)
        mask = mask_fn(sub['idx_20d'].fillna(0))
        filtered = sub[mask]
        e = filtered['excess'].mean() * 100 if len(filtered) > 0 else np.nan
        print(f"  {lbl.split(':')[0]}={e:+.3f}%(N={len(filtered)})", end='')
    print()

# ── IC check: new features in walk-forward period ─────────────────
print()
print("=" * 60)
print("Feature IC check (walk-forward eval period only, Oct-May):")
from scipy.stats import spearmanr
eval_panel = panel[
    (panel['date'] >= start_eval) & (panel['date'] <= end_eval)
].dropna(subset=FEAT_NEW + [TARGET_COLUMN])

for feat in ['rel_ret_5d', 'rel_ret_20d', 'beta_60d', 'ret_20d', 'ret_5d']:
    daily_ic = eval_panel.groupby('date').apply(
        lambda g: spearmanr(g[feat].dropna(),
                            g[TARGET_COLUMN].reindex(g[feat].dropna().index))[0]
        if len(g[feat].dropna()) > 10 else np.nan,
        include_groups=False
    )
    mean_ic = daily_ic.mean()
    t_stat  = mean_ic / (daily_ic.std() / np.sqrt(daily_ic.count()))
    bar = '+' * int(abs(mean_ic) * 400) if mean_ic > 0 else '-' * int(abs(mean_ic) * 400)
    print(f"  {feat:22s}: IC={mean_ic:+.4f}  t={t_stat:+.2f}  {bar}")
