"""Test multi-target ensemble + regime-adaptive weights vs baseline.

Compares 4 strategies on Mar-Apr 2026 walk-forward:
  A) baseline:           5d-target  + vol-adj weights  (current w2_014/015)
  B) ensemble:           3d+5d+10d  + vol-adj weights
  C) regime-adaptive:    5d-target  + score-prop / vol-adj blend by regime
  D) ensemble + regime:  3d+5d+10d  + regime blend
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from features import build_features, FEATURE_COLUMNS

prices = pd.read_parquet('data/prices.parquet')
panel  = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

# ── add target_5d, target_10d (target_3d already in panel) ────────
for n, col in [(5, 'target_5d'), (10, 'target_10d')]:
    t = (close_piv.shift(-n)/close_piv - 1).stack().reset_index()
    t.columns = ['date', 'stock_code', col]
    panel = panel.merge(t, on=['date', 'stock_code'], how='left')

idx_df = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_close = idx_df.set_index('date')['close']
idx_log   = np.log(idx_close)
idx_vol20 = idx_log.diff().rolling(20).std() * np.sqrt(252)

all_dates = sorted(panel['date'].unique())
TOP_K, CAP, HL = 30, 0.10, 60

# ── helpers ───────────────────────────────────────────────────────
def make_decay(df, hl=60, floor=0.5):
    ds = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds)-1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2)*delta/hl), floor)

def make_xgb(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=seed)

def regime_score(buy_date):
    """0 = bear/choppy → vol-adj. 1 = strong bull → score-prop."""
    if buy_date not in idx_log.index: return 0.5
    loc = idx_log.index.get_loc(buy_date)
    if loc < 20: return 0.5
    ret20 = float(idx_log.iloc[loc] - idx_log.iloc[loc-20])
    # pivot at ret20 = +5%, smooth: < -3% → ~0, > +13% → ~1
    return float(1 / (1 + np.exp(-20*(ret20 - 0.05))))

def score_prop_w(scores_s, top_k=TOP_K, cap=CAP):
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6; w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        w[~mask] += excess * (w[~mask]/w[~mask].sum())
    return pd.Series(w/w.sum(), index=top.index)

def vol_adj_w(scores_s, vol_s, top_k=TOP_K, cap=CAP, floor=0.001):
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6; w /= w.sum()
    vols = vol_s.reindex(top.index).values.astype(float)
    med  = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    w = w/vols; w /= w.sum(); w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def regime_blend_w(scores_s, vol_s, regime, top_k=TOP_K, cap=CAP):
    """Blend score-prop and vol-adj based on regime score."""
    top = scores_s.nlargest(top_k)
    sub_scores = top
    w_sp = score_prop_w(sub_scores, top_k=top_k, cap=cap)
    w_va = vol_adj_w(sub_scores, vol_s, top_k=top_k, cap=cap)
    # align indices
    idx = w_sp.index
    w   = regime * w_sp.values + (1-regime) * w_va.reindex(idx).values
    # re-enforce cap
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free + excess*(free/free.sum()), 0.001)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=idx)

def port_ret(weights, buy_date, sell_date):
    bp = close_piv.loc[buy_date].reindex(weights.index)
    sp_ = close_piv.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna())&(~sp_.isna())
    if valid.sum()==0: return np.nan
    wv = weights[valid]/weights[valid].sum()
    return float((wv*(sp_[valid]/bp[valid]-1)).sum())

# ── train monthly models for each target ──────────────────────────
TARGETS = ['target_3d', 'target_5d', 'target_10d']
TARGET_HORIZONS = {'target_3d': 3, 'target_5d': 5, 'target_10d': 10}

start_eval = pd.Timestamp('2026-03-01')
end_eval   = pd.Timestamp('2026-05-08')
eval_dates = [pd.Timestamp(d) for d in all_dates
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

# model_cache[ms][target] = trained model
model_cache = {}
print('Training monthly models for each target horizon...')
for ms in month_starts:
    avail = [d for d in all_dates if pd.Timestamp(d) < ms]
    if len(avail) < 86: continue
    # use longest horizon embargo (10d) for safety
    cutoff = pd.Timestamp(avail[-11])
    model_cache[ms] = {'cutoff': cutoff, 'models': {}}
    for tgt in TARGETS:
        tr = panel[panel['date']<=cutoff].dropna(subset=FEATURE_COLUMNS+[tgt])
        if len(tr) < 5000: continue
        sw = make_decay(tr, hl=HL)
        m  = make_xgb()
        m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values, sample_weight=sw, verbose=False)
        model_cache[ms]['models'][tgt] = m
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}  rows={len(tr):,}  models={list(model_cache[ms]["models"].keys())}')

def get_models(d):
    cands = [ms for ms in model_cache if ms <= d]
    return model_cache[max(cands)] if cands else None

# ── walk-forward: 5-day hold ──────────────────────────────────────
print('\nRunning walk-forward (5-day hold)...')
records = []
for i, d in enumerate(all_dates):
    buy_date = pd.Timestamp(d)
    if not (start_eval <= buy_date <= end_eval): continue
    if i+5 >= len(all_dates): continue
    sell_date = pd.Timestamp(all_dates[i+5])
    cache = get_models(buy_date)
    if cache is None or len(cache['models']) < 3: continue
    pred_df = panel[panel['date']==buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K: continue
    X = pred_df[FEATURE_COLUMNS].values

    # individual model scores
    s5  = pd.Series(cache['models']['target_5d'].predict(X),  index=pred_df['stock_code'].values)
    s3  = pd.Series(cache['models']['target_3d'].predict(X),  index=pred_df['stock_code'].values)
    s10 = pd.Series(cache['models']['target_10d'].predict(X), index=pred_df['stock_code'].values)
    vol_s = pred_df.set_index('stock_code')['vol_20d']

    # ensemble = rank-average of three (normalize to roughly comparable scale)
    rank_avg = (s3.rank(pct=True) * 0.3 + s5.rank(pct=True) * 0.4 + s10.rank(pct=True) * 0.3)

    bench = float(idx_close.loc[sell_date]/idx_close.loc[buy_date]-1)
    reg   = regime_score(buy_date)

    row = {'buy': buy_date, 'month': buy_date.strftime('%Y-%m'),
           'bench': bench, 'regime': reg}

    # A) baseline: 5d + vol-adj
    wA = vol_adj_w(s5, vol_s)
    row['A_baseline'] = port_ret(wA, buy_date, sell_date) - bench

    # B) ensemble + vol-adj
    wB = vol_adj_w(rank_avg, vol_s)
    row['B_ensemble'] = port_ret(wB, buy_date, sell_date) - bench

    # C) 5d + regime-blend
    wC = regime_blend_w(s5, vol_s, reg)
    row['C_regime'] = port_ret(wC, buy_date, sell_date) - bench

    # D) ensemble + regime-blend
    wD = regime_blend_w(rank_avg, vol_s, reg)
    row['D_ensemble_regime'] = port_ret(wD, buy_date, sell_date) - bench

    records.append(row)

df = pd.DataFrame(records).dropna()
print(f'\nEvaluated {len(df)} 5-day windows\n')

# ── monthly breakdown ─────────────────────────────────────────────
strategies = ['A_baseline', 'B_ensemble', 'C_regime', 'D_ensemble_regime']
print(f'  {"month":^10}{"N":^4}{"regime":^9}', '  '.join(f'{s:^15}' for s in strategies))
print('-'*82)
for mo in sorted(df['month'].unique()):
    sub = df[df['month']==mo]
    reg = sub['regime'].mean()
    vals = '  '.join(f'{sub[s].mean()*100:^+15.3f}' for s in strategies)
    print(f'  {mo:^10}{len(sub):^4}{reg:^9.2f}  {vals}')
print('-'*82)

# ── summary metrics ───────────────────────────────────────────────
print(f'\n  {"metric":^14}', '  '.join(f'{s:^15}' for s in strategies))
print('-'*82)
for metric, fn in [
    ('mean_exc%',   lambda s: df[s].mean()*100),
    ('std%',        lambda s: df[s].std()*100),
    ('sharpe',      lambda s: df[s].mean()/df[s].std()),
    ('win_rate',    lambda s: (df[s]>0).mean()),
    ('max_loss%',   lambda s: df[s].min()*100),
    ('max_gain%',   lambda s: df[s].max()*100),
]:
    vals = '  '.join(f'{fn(s):^15.4f}' for s in strategies)
    print(f'  {metric:^14}  {vals}')

# ── regime-conditional analysis ───────────────────────────────────
print('\n  Regime-conditional excess return (mean %)')
print('-'*82)
for label, mask in [
    ('regime>=0.7 (bull)',     df['regime']>=0.7),
    ('0.3<=regime<0.7 (mid)',  (df['regime']>=0.3)&(df['regime']<0.7)),
    ('regime<0.3 (choppy)',    df['regime']<0.3),
]:
    sub = df[mask]
    if len(sub) < 2: continue
    vals = '  '.join(f'{sub[s].mean()*100:^+15.3f}' for s in strategies)
    print(f'  {label:^22} N={len(sub):<3} {vals}')

# ── current Window 2 prediction ───────────────────────────────────
print('\n' + '='*82)
print('  Window 2 prediction — buy May 8, sell May 15 (5 trading days)')
print('='*82)
buy2  = pd.Timestamp('2026-05-08')
cache = get_models(buy2)
print(f'  Using model cohort: cutoff={cache["cutoff"].date()}')

pred = panel[panel['date']==buy2].dropna(subset=FEATURE_COLUMNS)
X    = pred[FEATURE_COLUMNS].values
s5   = pd.Series(cache['models']['target_5d'].predict(X),  index=pred['stock_code'].values)
s3   = pd.Series(cache['models']['target_3d'].predict(X),  index=pred['stock_code'].values)
s10  = pd.Series(cache['models']['target_10d'].predict(X), index=pred['stock_code'].values)
vol_s = pred.set_index('stock_code')['vol_20d']
rank_avg = (s3.rank(pct=True)*0.3 + s5.rank(pct=True)*0.4 + s10.rank(pct=True)*0.3)
reg = regime_score(buy2)
print(f'  Regime score: {reg:.3f}  ({"strong bull" if reg>0.7 else "mid" if reg>0.3 else "choppy/bear"})')

print(f'\n  Top 30 stocks per strategy (intersect):')
for label, w in [
    ('A_baseline (5d+vol)',         vol_adj_w(s5, vol_s)),
    ('B_ensemble (3+5+10 + vol)',   vol_adj_w(rank_avg, vol_s)),
    ('C_regime  (5d + blend)',      regime_blend_w(s5, vol_s, reg)),
    ('D_ens+regime (best)',         regime_blend_w(rank_avg, vol_s, reg)),
]:
    top5 = w.nlargest(5)
    s5_str = ' '.join(f'{c}:{v*100:.1f}%' for c, v in top5.items())
    print(f'  {label:<32} {s5_str}')

# Save the recommended portfolio
print(f'\n  Saving D (ensemble+regime) as candidate w2_016...')
wD = regime_blend_w(rank_avg, vol_s, reg)
out = wD.sort_values(ascending=False).reset_index()
out.columns = ['stock_code', 'weight']
out_path = Path('outputs/submissions/w2_016_ensemble_regime_candidate.csv')
out.to_csv(out_path, index=False)
print(f'  Saved: {out_path}')
print(f'  Sum={out["weight"].sum():.6f}  Max={out["weight"].max()*100:.2f}%  N={len(out)}')
