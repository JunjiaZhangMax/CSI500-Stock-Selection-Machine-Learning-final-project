"""Walk-forward validation: ensemble+vol-adj WITH vs WITHOUT rally penalty (P3).

Compares 3 strategies on Mar-Apr 2026 walk-forward:
  A) baseline:                5d-target + vol-adj  (current w2_015)
  B) ensemble:                3d+5d+10d + vol-adj  (current w2_017)
  E) ensemble + rally penalty 3d+5d+10d + vol-adj + P3 composite  (candidate w2_018)
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from features import build_features, FEATURE_COLUMNS

prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

for n, col in [(5,'target_5d'),(10,'target_10d')]:
    t=(close_piv.shift(-n)/close_piv-1).stack().reset_index()
    t.columns=['date','stock_code',col]
    panel = panel.merge(t, on=['date','stock_code'], how='left')

# Pre-compute rally indicators
ret_3d_piv = close_piv.pct_change(3)
ret_5d_piv = close_piv.pct_change(5)

idx_df = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date']); idx_close = idx_df.set_index('date')['close']
all_dates = sorted(panel['date'].unique())

TOP_K, CAP, HL = 30, 0.10, 60

def make_decay(df, hl=60, floor=0.5):
    ds=np.sort(df['date'].unique()); d2i={pd.Timestamp(d):i for i,d in enumerate(ds)}
    return np.maximum(np.exp(-np.log(2)*((len(ds)-1)-df['date'].map(d2i).values)/hl), floor)

def make_xgb():
    return xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=10,reg_lambda=1.0,tree_method='hist',n_jobs=-1,random_state=42)

def vol_adj_w(scores_s, vol_s, top_k=TOP_K, cap=CAP, floor=0.001):
    top=scores_s.nlargest(top_k); w=top.values-top.values.min()+1e-6; w/=w.sum()
    vols=vol_s.reindex(top.index).values.astype(float)
    med=float(np.nanmedian(vols[vols>0])); vols=np.where(np.isnan(vols)|(vols<=0),med,vols)
    w=w/vols; w/=w.sum(); w=np.maximum(w,floor); w/=w.sum()
    for _ in range(500):
        mask=w>cap
        if not mask.any(): break
        excess=(w[mask]-cap).sum(); w[mask]=cap
        free=w[~mask]; free=np.maximum(free+excess*(free/free.sum()),floor); w[~mask]=free; w/=w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def norm01(x, lo, hi): return np.clip((x-lo)/(hi-lo), 0, 1)

def rally_penalty_w(base_w, snap_features, buy_date):
    """Apply P3 composite rally penalty to base weights, then re-cap."""
    codes = base_w.index.tolist()
    r5  = np.array([float(ret_5d_piv.loc[buy_date, c]*100) if c in ret_5d_piv.columns else 0 for c in codes])
    rsi = np.array([float(snap_features.loc[c,'rsi_14']) if c in snap_features.index else 50 for c in codes])
    pma = np.array([float(snap_features.loc[c,'close_over_ma20']) if c in snap_features.index else 1.0 for c in codes])
    ob = 0.5*norm01(r5,0,20) + 0.3*norm01(rsi,50,85) + 0.2*norm01(pma,1.0,1.20)
    penalty = 1 - 0.6*ob
    w = base_w.values * penalty
    w = np.maximum(w, 0.001); w /= w.sum()
    for _ in range(500):
        mask = w > CAP
        if not mask.any(): break
        excess=(w[mask]-CAP).sum(); w[mask]=CAP
        free=w[~mask]; free=np.maximum(free+excess*(free/free.sum()),0.001); w[~mask]=free; w/=w.sum()
    return pd.Series(w/w.sum(), index=codes)

def port_ret(weights, buy_date, sell_date):
    bp=close_piv.loc[buy_date].reindex(weights.index); sp_=close_piv.loc[sell_date].reindex(weights.index)
    valid=(~bp.isna())&(~sp_.isna())
    if valid.sum()==0: return np.nan
    wv=weights[valid]/weights[valid].sum()
    return float((wv*(sp_[valid]/bp[valid]-1)).sum())

# ── Train monthly models for each target ──────────────────────────
TARGETS = ['target_3d','target_5d','target_10d']
start_eval=pd.Timestamp('2026-03-01'); end_eval=pd.Timestamp('2026-05-08')
eval_dates=[pd.Timestamp(d) for d in all_dates if start_eval<=pd.Timestamp(d)<=end_eval]
month_starts=sorted(set(d.replace(day=1) for d in eval_dates))

model_cache={}
print('Training monthly models for each target...')
for ms in month_starts:
    avail=[d for d in all_dates if pd.Timestamp(d)<ms]
    if len(avail)<86: continue
    cutoff=pd.Timestamp(avail[-11])
    model_cache[ms]={'cutoff':cutoff,'models':{}}
    for tgt in TARGETS:
        tr=panel[panel['date']<=cutoff].dropna(subset=FEATURE_COLUMNS+[tgt])
        if len(tr)<5000: continue
        m=make_xgb(); m.fit(tr[FEATURE_COLUMNS].values,tr[tgt].values,sample_weight=make_decay(tr),verbose=False)
        model_cache[ms]['models'][tgt]=m
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}  rows={len(tr):,}')

def get_models(d):
    cands=[ms for ms in model_cache if ms<=d]
    return model_cache[max(cands)] if cands else None

# ── Walk-forward ──────────────────────────────────────────────────
print('\nRunning walk-forward...')
records=[]
for i, d in enumerate(all_dates):
    buy_date=pd.Timestamp(d)
    if not(start_eval<=buy_date<=end_eval): continue
    if i+5>=len(all_dates): continue
    sell_date=pd.Timestamp(all_dates[i+5])
    cache=get_models(buy_date)
    if cache is None or len(cache['models'])<3: continue

    pred_df=panel[panel['date']==buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df)<TOP_K: continue
    snap=pred_df.set_index('stock_code')
    X=pred_df[FEATURE_COLUMNS].values

    s5  = pd.Series(cache['models']['target_5d'].predict(X),  index=pred_df['stock_code'].values)
    s3  = pd.Series(cache['models']['target_3d'].predict(X),  index=pred_df['stock_code'].values)
    s10 = pd.Series(cache['models']['target_10d'].predict(X), index=pred_df['stock_code'].values)
    vol_s=pred_df.set_index('stock_code')['vol_20d']
    rank_avg = (s3.rank(pct=True)*0.3 + s5.rank(pct=True)*0.4 + s10.rank(pct=True)*0.3)

    bench=float(idx_close.loc[sell_date]/idx_close.loc[buy_date]-1)

    # market regime tag
    loc=idx_close.index.get_loc(buy_date)
    if loc>=20:
        log_idx=np.log(idx_close)
        ret20=float(log_idx.iloc[loc]-log_idx.iloc[loc-20])
        regime=float(1/(1+np.exp(-20*(ret20-0.05))))
    else: regime=0.5

    # average ret_5d of selected stocks (for diagnostic)
    wA=vol_adj_w(s5, vol_s)
    wB=vol_adj_w(rank_avg, vol_s)
    wE=rally_penalty_w(wB, snap, buy_date)

    avg_r5_B = float(np.mean([ret_5d_piv.loc[buy_date,c]*100 if c in ret_5d_piv.columns else 0
                              for c in wB.index]))

    records.append({
        'buy': buy_date, 'month': buy_date.strftime('%Y-%m'),
        'bench': bench, 'regime': regime, 'avg_r5_B': avg_r5_B,
        'A_baseline':       port_ret(wA, buy_date, sell_date) - bench,
        'B_ensemble':       port_ret(wB, buy_date, sell_date) - bench,
        'E_ens_penalty':    port_ret(wE, buy_date, sell_date) - bench,
    })

df=pd.DataFrame(records).dropna()
print(f'\nEvaluated {len(df)} 5-day windows\n')

strategies=['A_baseline','B_ensemble','E_ens_penalty']

print(f'  {"month":^10}{"N":^4}{"regime":^9}{"avg_r5%":^9}', '  '.join(f'{s:^16}' for s in strategies))
print('-'*82)
for mo in sorted(df['month'].unique()):
    sub=df[df['month']==mo]
    vals='  '.join(f'{sub[s].mean()*100:^+16.3f}' for s in strategies)
    print(f'  {mo:^10}{len(sub):^4}{sub["regime"].mean():^9.2f}{sub["avg_r5_B"].mean():^+9.2f}  {vals}')
print('-'*82)

print(f'\n  {"metric":^14}', '  '.join(f'{s:^16}' for s in strategies))
print('-'*70)
for metric, fn in [
    ('mean_exc%',  lambda s: df[s].mean()*100),
    ('std%',       lambda s: df[s].std()*100),
    ('sharpe',     lambda s: df[s].mean()/df[s].std()),
    ('win_rate',   lambda s: (df[s]>0).mean()),
    ('max_loss%',  lambda s: df[s].min()*100),
    ('max_gain%',  lambda s: df[s].max()*100),
]:
    vals='  '.join(f'{fn(s):^16.4f}' for s in strategies)
    print(f'  {metric:^14}  {vals}')

# Sub-group: high-rally regime (when avg_r5 > 5% in selected portfolio)
print('\n  Conditional on avg ret_5d of selected stocks (B portfolio):')
print('-'*70)
for label, mask in [
    ('avg_r5 > 8% (frothy)',  df['avg_r5_B']>8),
    ('5% < avg_r5 <= 8%',     (df['avg_r5_B']>5)&(df['avg_r5_B']<=8)),
    ('0% < avg_r5 <= 5%',     (df['avg_r5_B']>0)&(df['avg_r5_B']<=5)),
    ('avg_r5 <= 0% (cooling)',df['avg_r5_B']<=0),
]:
    sub=df[mask]
    if len(sub)<2: continue
    vals='  '.join(f'{sub[s].mean()*100:^+16.3f}' for s in strategies)
    print(f'  {label:^22} N={len(sub):<3}  {vals}')
