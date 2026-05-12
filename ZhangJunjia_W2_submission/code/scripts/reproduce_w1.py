"""Reproduce W1 submission: score-prop top50 with 8% cap (exp_021 baseline config).

Pipeline:
  1. Load CSI500 daily OHLCV data
  2. Build features via src/features.py (FEATURE_COLUMNS, includes amplitude)
  3. Train XGBoost on 3-day forward return target
     - 120-day half-life time-decay sample weights, floor 0.5
     - Hardcoded training cutoff: 2026-04-08 (matches original exp_021 setup)
  4. Predict scores for the as_of date (April 30, 2026 - last trading day before W1 hold)
  5. Apply score-proportional weighting:
     - Take top 50 stocks by predicted score
     - Weight ∝ (score - min_score)
     - Iteratively cap individual weights at 8%
"""
import sys, warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

# Add src to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / 'src'))
from features import build_features, FEATURE_COLUMNS

# ── Config (matches exp_021 baseline) ─────────────────────────────
DATA_DIR    = ROOT / 'data'
OUT_PATH    = ROOT / 'submissions' / 'w1_021_score_prop_cap8.csv'
AS_OF       = pd.Timestamp('2026-04-30')   # last trading day before W1 hold (May 6-8)
TRAIN_CUTOFF= pd.Timestamp('2026-04-08')   # hardcoded cutoff (matches exp_021)
HALF_LIFE   = 120                           # slow decay (~6 months)
WEIGHT_FLOOR= 0.5                           # old data keeps 50% baseline weight
TOP_K       = 50
CAP         = 0.08

XGB_PARAMS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, gamma=0.0,
    tree_method='hist', n_jobs=-1, random_state=42,
)

# ── Data + features ───────────────────────────────────────────────
print('Loading data...')
prices = pd.read_parquet(DATA_DIR / 'prices.parquet')
panel  = build_features(prices)
# build_features already provides target_3d = close(t+3)/close(t) - 1

print(f'  as_of={AS_OF.date()}  train_cutoff={TRAIN_CUTOFF.date()}')
cutoff = TRAIN_CUTOFF

# ── Time-decay sample weights ─────────────────────────────────────
def make_decay_weights(df, hl, floor):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

# ── Train XGBoost ─────────────────────────────────────────────────
print('Training XGBoost on 3-day target...')
tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS + ['target_3d'])
sw = make_decay_weights(tr, hl=HALF_LIFE, floor=WEIGHT_FLOOR)
model = xgb.XGBRegressor(**XGB_PARAMS)
model.fit(tr[FEATURE_COLUMNS].values, tr['target_3d'].values,
          sample_weight=sw, verbose=False)
print(f'  trained on {len(tr):,} rows')

# ── Predict for as_of date ────────────────────────────────────────
pred = panel[panel['date'] == AS_OF].dropna(subset=FEATURE_COLUMNS)
scores = pd.Series(model.predict(pred[FEATURE_COLUMNS].values),
                   index=pred['stock_code'].values)
print(f'  predicted {len(scores)} stocks')

# ── Score-proportional weighting with iterative cap ───────────────
def score_prop_weights(scores_s, top_k, cap):
    top = scores_s.nlargest(top_k)
    w = top.values - top.values.min() + 1e-6
    w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask] - cap).sum()
        w[mask] = cap
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)

weights = score_prop_weights(scores, TOP_K, CAP).sort_values(ascending=False)

# ── Save ──────────────────────────────────────────────────────────
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
out = weights.reset_index()
out.columns = ['stock_code', 'weight']
out.to_csv(OUT_PATH, index=False)
print(f'\nSaved: {OUT_PATH}')
print(f'  N={len(weights)}  Sum={weights.sum():.6f}  Max={weights.max()*100:.2f}%  Min={weights.min()*100:.4f}%')
