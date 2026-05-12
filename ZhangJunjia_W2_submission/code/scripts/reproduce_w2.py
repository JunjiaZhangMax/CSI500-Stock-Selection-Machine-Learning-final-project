"""Reproduce W2 submission: 3+5 ensemble with concentration boost (pow=2) + soft vol-adj.

Pipeline:
  1. Load CSI500 daily OHLCV data
  2. Build features via src/features.py
  3. Train two XGBoost models (3-day + 5-day forward return targets)
     - 60-day half-life time-decay sample weights
     - Cutoff: 5 trading days before as_of date (embargo for 5d target)
  4. Predict scores from each model on the as_of date (May 8, 2026)
  5. Combine: rank-percentile average (50% from 3d model, 50% from 5d model)
  6. Apply blacklist: exclude 002261 (regulatory warning May 7-8)
  7. Concentrated vol-adjusted weighting:
     - Take top 30 stocks by ensemble score
     - Amplify score differences: w_raw = score_norm^2  (concentration boost)
     - Soft vol adjustment: w / vol_20d^0.5  (less aggressive than pure 1/vol)
     - Iteratively cap individual weights at 10%
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import build_features, FEATURE_COLUMNS

# ── Config ────────────────────────────────────────────────────────
DATA_DIR    = ROOT / 'data'
OUT_PATH    = ROOT / 'submissions' / 'w2_023_pow2_softvol.csv'
AS_OF       = pd.Timestamp('2026-05-08')   # last trading day before W2 hold (May 11-15)
EMBARGO     = 5                             # 5-day target horizon
HALF_LIFE   = 60                            # faster decay (~3 months)
WEIGHT_FLOOR= 0.5
TOP_K       = 30
CAP         = 0.10
SCORE_POW   = 2.0                           # amplify top scores
VOL_POW     = 0.5                           # soft vol-adj (1/sqrt(vol))
WEIGHT_FLOOR_PORT = 0.001                   # min per-stock weight (positive constraint)

BLACKLIST = {'002261'}                      # 002261 — regulatory warning 2026-05-07/08

XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42,
)

# ── Data + features + extra targets ───────────────────────────────
print('Loading data...')
prices = pd.read_parquet(DATA_DIR / 'prices.parquet')
panel  = build_features(prices)

close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
# target_3d already in panel from features.py; only need to add target_5d
t5 = (close_piv.shift(-5) / close_piv - 1).stack().reset_index()
t5.columns = ['date', 'stock_code', 'target_5d']
panel = panel.merge(t5, on=['date', 'stock_code'], how='left')

all_dates = sorted(panel['date'].unique())
# Walk-forward training cutoff: monthly retrain logic
# Use the last available trading day before the start of as_of's month, then back EMBARGO days
month_start = pd.Timestamp(AS_OF.year, AS_OF.month, 1)
avail = [d for d in all_dates if pd.Timestamp(d) < month_start]
cutoff = pd.Timestamp(avail[-(EMBARGO + 1)])
print(f'  as_of={AS_OF.date()}  train_cutoff={cutoff.date()}')

# ── Helpers ───────────────────────────────────────────────────────
def make_decay_weights(df, hl, floor):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

def make_xgb():
    return xgb.XGBRegressor(**XGB_PARAMS)

# ── Train two XGBoost models (3d + 5d targets) ────────────────────
print('Training ensemble (target_3d, target_5d)...')
models = {}
for tgt in ['target_3d', 'target_5d']:
    tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS + [tgt])
    sw = make_decay_weights(tr, hl=HALF_LIFE, floor=WEIGHT_FLOOR)
    m = make_xgb()
    m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values, sample_weight=sw, verbose=False)
    models[tgt] = m
    print(f'  {tgt}: trained on {len(tr):,} rows')

# ── Predict + ensemble + blacklist ────────────────────────────────
pred = panel[panel['date'] == AS_OF].dropna(subset=FEATURE_COLUMNS)
X = pred[FEATURE_COLUMNS].values
codes = pred['stock_code'].values

s3 = pd.Series(models['target_3d'].predict(X), index=codes)
s5 = pd.Series(models['target_5d'].predict(X), index=codes)
vol_s = pred.set_index('stock_code')['vol_20d']

# Apply blacklist BEFORE ranking
keep = ~s3.index.isin(BLACKLIST)
s3 = s3[keep]; s5 = s5[keep]; vol_s = vol_s[s3.index]
print(f'  predictions: {len(s3)} stocks (after blacklist)')

# Rank-percentile ensemble (50/50 from each horizon)
ensemble_score = s3.rank(pct=True) * 0.5 + s5.rank(pct=True) * 0.5

# ── Concentrated vol-adj weighting ────────────────────────────────
def conc_weights(scores_s, vol_s, score_pow, vol_pow,
                 top_k, cap, floor):
    top = scores_s.nlargest(top_k)
    # Normalize scores to [0, 1] then amplify
    s_norm = (top.values - top.values.min()) / (top.values.max() - top.values.min() + 1e-9)
    s_norm = s_norm + 1e-3
    w = s_norm ** score_pow
    w = w / w.sum()
    # Soft vol adjustment
    vols = vol_s.reindex(top.index).values.astype(float)
    med = float(np.nanmedian(vols[vols > 0]))
    vols = np.where(np.isnan(vols) | (vols <= 0), med, vols)
    w = w / (vols ** vol_pow)
    w = w / w.sum()
    # Floor + iterative cap enforcement
    w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask] - cap).sum()
        w[mask] = cap
        free = w[~mask]
        free = np.maximum(free + excess * (free / free.sum()), floor)
        w[~mask] = free
        w /= w.sum()
    return pd.Series(w / w.sum(), index=top.index)

weights = conc_weights(
    ensemble_score, vol_s,
    score_pow=SCORE_POW, vol_pow=VOL_POW,
    top_k=TOP_K, cap=CAP, floor=WEIGHT_FLOOR_PORT,
).sort_values(ascending=False)

# ── Save ──────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out = weights.reset_index()
out.columns = ['stock_code', 'weight']
out.to_csv(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')
print(f'  N={len(weights)}  Sum={weights.sum():.6f}  Max={weights.max()*100:.2f}%  Min={weights.min()*100:.4f}%')
print(f'\nTop 10:')
for c, w in weights.head(10).items():
    print(f'  {c}  {w*100:6.2f}%')
