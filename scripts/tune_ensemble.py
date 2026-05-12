"""Hyperparameter tuning for ensemble weighting.

Tests 14 combinations on Mar-Apr 5-day walk-forward:
- Single horizons: 3d, 5d, 7d, 10d
- Pairs: 3+5, 5+10, 3+10
- Triples with various weights
- Plus: raw vs rank averaging, seed-ensemble variant
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from features import build_features, FEATURE_COLUMNS

prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

# Add 5d/7d/10d targets (3d already in panel)
for n,col in [(5,'target_5d'),(7,'target_7d'),(10,'target_10d')]:
    t=(close_piv.shift(-n)/close_piv-1).stack().reset_index()
    t.columns=['date','stock_code',col]
    panel = panel.merge(t, on=['date','stock_code'], how='left')

idx_df = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date']=pd.to_datetime(idx_df['date']); idx_close=idx_df.set_index('date')['close']
all_dates=sorted(panel['date'].unique())

TOP_K, CAP, HL = 30, 0.10, 60
TARGETS = ['target_3d','target_5d','target_7d','target_10d']

def make_decay(df, hl=60, floor=0.5):
    ds=np.sort(df['date'].unique()); d2i={pd.Timestamp(d):i for i,d in enumerate(ds)}
    return np.maximum(np.exp(-np.log(2)*((len(ds)-1)-df['date'].map(d2i).values)/hl), floor)

def make_xgb(seed=42):
    return xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=10,reg_lambda=1.0,tree_method='hist',n_jobs=-1,random_state=seed)

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

def port_ret(weights, buy_date, sell_date):
    bp=close_piv.loc[buy_date].reindex(weights.index); sp_=close_piv.loc[sell_date].reindex(weights.index)
    valid=(~bp.isna())&(~sp_.isna())
    if valid.sum()==0: return np.nan
    wv=weights[valid]/weights[valid].sum()
    return float((wv*(sp_[valid]/bp[valid]-1)).sum())

# ── Train monthly models for each target ──────────────────────────
start_eval=pd.Timestamp('2026-03-01'); end_eval=pd.Timestamp('2026-05-08')
eval_dates=[pd.Timestamp(d) for d in all_dates if start_eval<=pd.Timestamp(d)<=end_eval]
month_starts=sorted(set(d.replace(day=1) for d in eval_dates))

print('Training monthly models for each target horizon (3d, 5d, 7d, 10d)...')
model_cache={}
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
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}')

def get_models(d):
    cands=[ms for ms in model_cache if ms<=d]
    return model_cache[max(cands)] if cands else None

# ── Define configurations to test ─────────────────────────────────
# Each config: (label, weights_dict, use_rank)
CONFIGS = [
    # Single horizon baselines
    ('3d only',           {'target_3d':1.0},                                    False),
    ('5d only [BASELINE]',{'target_5d':1.0},                                    False),
    ('7d only',           {'target_7d':1.0},                                    False),
    ('10d only',          {'target_10d':1.0},                                   False),
    # Pairs
    ('3+5 equal',         {'target_3d':0.5,'target_5d':0.5},                    True),
    ('5+10 equal',        {'target_5d':0.5,'target_10d':0.5},                   True),
    ('5+7 equal',         {'target_5d':0.5,'target_7d':0.5},                    True),
    ('3+10 equal',        {'target_3d':0.5,'target_10d':0.5},                   True),
    # Triples
    ('3+5+10 equal',      {'target_3d':1/3,'target_5d':1/3,'target_10d':1/3},   True),
    ('3+5+10 (cur)',      {'target_3d':0.3,'target_5d':0.4,'target_10d':0.3},   True),
    ('3+5+10 5d-heavy',   {'target_3d':0.2,'target_5d':0.6,'target_10d':0.2},   True),
    ('3+5+10 long-bias',  {'target_3d':0.2,'target_5d':0.3,'target_10d':0.5},   True),
    ('3+5+10 short-bias', {'target_3d':0.5,'target_5d':0.3,'target_10d':0.2},   True),
    # Quad
    ('3+5+7+10 equal',    {'target_3d':0.25,'target_5d':0.25,'target_7d':0.25,'target_10d':0.25}, True),
    # 5d-centered
    ('5+7+10 5d-heavy',   {'target_5d':0.5,'target_7d':0.25,'target_10d':0.25}, True),
    ('5d only RAW',       {'target_5d':1.0},                                    False),  # sanity
]

# ── Walk-forward ──────────────────────────────────────────────────
print('\nRunning walk-forward...\n')
records=[]
for i, d in enumerate(all_dates):
    buy_date=pd.Timestamp(d)
    if not(start_eval<=buy_date<=end_eval): continue
    if i+5>=len(all_dates): continue
    sell_date=pd.Timestamp(all_dates[i+5])
    cache=get_models(buy_date)
    if cache is None or len(cache['models'])<4: continue

    pred_df=panel[panel['date']==buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df)<TOP_K: continue
    X=pred_df[FEATURE_COLUMNS].values
    codes=pred_df['stock_code'].values
    vol_s=pred_df.set_index('stock_code')['vol_20d']

    # cache predictions per target
    preds={tgt: pd.Series(cache['models'][tgt].predict(X), index=codes) for tgt in TARGETS}

    bench=float(idx_close.loc[sell_date]/idx_close.loc[buy_date]-1)
    row={'buy':buy_date,'month':buy_date.strftime('%Y-%m'),'bench':bench}

    for label, weights, use_rank in CONFIGS:
        score = pd.Series(0.0, index=codes)
        for tgt, wt in weights.items():
            s = preds[tgt]
            if use_rank: s = s.rank(pct=True)
            score = score + wt * s
        w = vol_adj_w(score, vol_s)
        row[label] = port_ret(w, buy_date, sell_date) - bench

    records.append(row)

df=pd.DataFrame(records).dropna()
print(f'Evaluated {len(df)} 5-day windows\n')

# ── Summary ───────────────────────────────────────────────────────
results=[]
for label, _, _ in CONFIGS:
    s=df[label]
    results.append({
        'config': label,
        'mean%':  s.mean()*100,
        'std%':   s.std()*100,
        'sharpe': s.mean()/s.std(),
        'win%':   (s>0).mean()*100,
        'minl%':  s.min()*100,
        'maxg%':  s.max()*100,
    })

res=pd.DataFrame(results).sort_values('sharpe', ascending=False)
print('='*88)
print('  Ensemble configurations ranked by Sharpe')
print('='*88)
print(f'  {"config":<22} {"mean%":>8} {"std%":>7} {"sharpe":>8} {"win%":>7} {"min_loss%":>10} {"max_gain%":>10}')
print('-'*88)
for _, r in res.iterrows():
    marker = ' BEST' if r['sharpe']==res['sharpe'].max() else ''
    print(f'  {r["config"]:<22} {r["mean%"]:>+8.3f} {r["std%"]:>7.3f} {r["sharpe"]:>8.4f} {r["win%"]:>6.1f}% {r["minl%"]:>+10.3f} {r["maxg%"]:>+10.3f}{marker}')

# ── Pick top-3 and break down by month ────────────────────────────
top3 = res.head(3)['config'].tolist()
print('\n  Monthly breakdown for top-3 configurations:')
print('-'*65)
print(f'  {"month":^8}{"N":^4}', '  '.join(f'{c:^18}' for c in top3))
for mo in sorted(df['month'].unique()):
    sub=df[df['month']==mo]
    vals='  '.join(f'{sub[c].mean()*100:^+18.3f}' for c in top3)
    print(f'  {mo:^8}{len(sub):^4}  {vals}')
