"""Test concentration variants - amplifying top scores to create 'main force' positions.

All variants use 3+5 ensemble + vol_pow=0.5 (your preferred soft vol-adj).
Difference is how aggressively top scores get amplified before vol-adj.
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from features import build_features, FEATURE_COLUMNS

BLACKLIST = {'002261'}

prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
t5 = (close_piv.shift(-5)/close_piv-1).stack().reset_index()
t5.columns = ['date','stock_code','target_5d']
panel = panel.merge(t5, on=['date','stock_code'], how='left')

idx_df = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date']=pd.to_datetime(idx_df['date']); idx_close=idx_df.set_index('date')['close']
all_dates = sorted(panel['date'].unique())

TOP_K, CAP, HL = 30, 0.10, 60

def make_decay(df, hl=60, floor=0.5):
    ds=np.sort(df['date'].unique()); d2i={pd.Timestamp(d):i for i,d in enumerate(ds)}
    return np.maximum(np.exp(-np.log(2)*((len(ds)-1)-df['date'].map(d2i).values)/hl), floor)

def make_xgb():
    return xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=10,reg_lambda=1.0,tree_method='hist',n_jobs=-1,random_state=42)

def conc_w(scores_s, vol_s, score_pow=1.0, vol_pow=0.5,
           top_k=TOP_K, cap=CAP, floor=0.001):
    """score_pow > 1 amplifies top-end concentration."""
    top = scores_s.nlargest(top_k)
    # Normalize scores to [0, 1]
    s_norm = (top.values - top.values.min()) / (top.values.max() - top.values.min() + 1e-9)
    s_norm = s_norm + 1e-3  # tiny offset so min isn't exactly 0
    # Amplify
    w = s_norm ** score_pow
    w = w / w.sum()
    # Vol adjustment
    vols = vol_s.reindex(top.index).values.astype(float)
    med = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    w = w / (vols ** vol_pow)
    w = w / w.sum()
    w = np.maximum(w, floor); w /= w.sum()
    # Cap enforcement
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def tiered_w(scores_s, vol_s, top_tier=5, mid_tier=10,
             top_pct=0.50, mid_pct=0.40, top_k=TOP_K, cap=CAP, floor=0.005):
    """Explicit 3-tier: top N get top_pct of weight evenly, middle N get mid_pct, rest get floor."""
    top = scores_s.nlargest(top_k)
    n = len(top)
    # Vol-adjust within each tier
    vols = vol_s.reindex(top.index).values.astype(float)
    med = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    inv_vol = 1.0/np.sqrt(vols)

    w = np.zeros(n)
    # Top tier
    iv_top = inv_vol[:top_tier]; w[:top_tier] = top_pct * iv_top / iv_top.sum()
    # Mid tier
    iv_mid = inv_vol[top_tier:top_tier+mid_tier]; w[top_tier:top_tier+mid_tier] = mid_pct * iv_mid / iv_mid.sum()
    # Tail tier
    rest = 1.0 - top_pct - mid_pct
    iv_tail = inv_vol[top_tier+mid_tier:]; w[top_tier+mid_tier:] = rest * iv_tail / iv_tail.sum()

    w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def port_ret(weights, buy_date, sell_date):
    bp=close_piv.loc[buy_date].reindex(weights.index); sp_=close_piv.loc[sell_date].reindex(weights.index)
    valid=(~bp.isna())&(~sp_.isna())
    if valid.sum()==0: return np.nan
    wv=weights[valid]/weights[valid].sum()
    return float((wv*(sp_[valid]/bp[valid]-1)).sum())

# ── Train monthly models ──────────────────────────────────────────
start_eval=pd.Timestamp('2026-03-01'); end_eval=pd.Timestamp('2026-05-08')
eval_dates=[pd.Timestamp(d) for d in all_dates if start_eval<=pd.Timestamp(d)<=end_eval]
month_starts=sorted(set(d.replace(day=1) for d in eval_dates))

print('Training monthly models (3d, 5d)...')
model_cache={}
for ms in month_starts:
    avail=[d for d in all_dates if pd.Timestamp(d)<ms]
    if len(avail)<86: continue
    cutoff=pd.Timestamp(avail[-6])
    model_cache[ms]={'cutoff':cutoff,'models':{}}
    for tgt in ['target_3d','target_5d']:
        tr=panel[panel['date']<=cutoff].dropna(subset=FEATURE_COLUMNS+[tgt])
        if len(tr)<5000: continue
        m=make_xgb(); m.fit(tr[FEATURE_COLUMNS].values,tr[tgt].values,sample_weight=make_decay(tr),verbose=False)
        model_cache[ms]['models'][tgt]=m
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}')

def get_models(d):
    cands=[ms for ms in model_cache if ms<=d]
    return model_cache[max(cands)] if cands else None

# ── Walk-forward ──────────────────────────────────────────────────
print('\nWalk-forward...')
records=[]
for i, d in enumerate(all_dates):
    buy_date=pd.Timestamp(d)
    if not(start_eval<=buy_date<=end_eval): continue
    if i+5>=len(all_dates): continue
    sell_date=pd.Timestamp(all_dates[i+5])
    cache=get_models(buy_date)
    if cache is None or len(cache['models'])<2: continue

    pred_df=panel[panel['date']==buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df)<TOP_K: continue
    X=pred_df[FEATURE_COLUMNS].values

    s3=pd.Series(cache['models']['target_3d'].predict(X), index=pred_df['stock_code'].values)
    s5=pd.Series(cache['models']['target_5d'].predict(X), index=pred_df['stock_code'].values)
    vol_s=pred_df.set_index('stock_code')['vol_20d']
    score=s3.rank(pct=True)*0.5 + s5.rank(pct=True)*0.5

    bench=float(idx_close.loc[sell_date]/idx_close.loc[buy_date]-1)
    row={'buy':buy_date,'month':buy_date.strftime('%Y-%m'),'bench':bench}

    variants = [
        ('V0_pow1_pow1.0',  conc_w(score, vol_s, score_pow=1, vol_pow=1.0)),  # current w2_020
        ('V5_pow1_pow0.5',  conc_w(score, vol_s, score_pow=1, vol_pow=0.5)),  # w2_022
        ('V8_pow2_pow0.5',  conc_w(score, vol_s, score_pow=2, vol_pow=0.5)),  # mild concentrate
        ('V9_pow3_pow0.5',  conc_w(score, vol_s, score_pow=3, vol_pow=0.5)),  # strong
        ('V10_pow5_pow0.5', conc_w(score, vol_s, score_pow=5, vol_pow=0.5)),  # very strong
        ('V11_tier_50_40_10', tiered_w(score, vol_s, top_tier=5, mid_tier=10, top_pct=0.50, mid_pct=0.40)),
        ('V12_tier_55_35_10', tiered_w(score, vol_s, top_tier=5, mid_tier=10, top_pct=0.55, mid_pct=0.35)),
    ]

    for label, w in variants:
        row[label] = port_ret(w, buy_date, sell_date) - bench

    records.append(row)

df=pd.DataFrame(records).dropna()
print(f'\nEvaluated {len(df)} windows\n')

strats=[v[0] for v in variants]
print(f'  {"month":^8}{"N":^4}', '  '.join(f'{s[:13]:^14}' for s in strats))
print('-'*120)
for mo in sorted(df['month'].unique()):
    sub=df[df['month']==mo]
    vals='  '.join(f'{sub[s].mean()*100:^+14.3f}' for s in strats)
    print(f'  {mo:^8}{len(sub):^4}  {vals}')
print('-'*120)

print(f'\n  {"metric":^14}', '  '.join(f'{s[:13]:^14}' for s in strats))
print('-'*120)
for metric, fn in [
    ('mean_exc%',  lambda s: df[s].mean()*100),
    ('std%',       lambda s: df[s].std()*100),
    ('sharpe',     lambda s: df[s].mean()/df[s].std()),
    ('win_rate',   lambda s: (df[s]>0).mean()),
    ('max_loss%',  lambda s: df[s].min()*100),
    ('max_gain%',  lambda s: df[s].max()*100),
]:
    vals='  '.join(f'{fn(s):^14.4f}' for s in strats)
    print(f'  {metric:^14}  {vals}')

# ── Generate W2 candidates ────────────────────────────────────────
print('\n' + '='*120)
print('  W2 candidates (concentration test)')
print('='*120)

buy2=pd.Timestamp('2026-05-08')
avail=[d for d in all_dates if pd.Timestamp(d)<pd.Timestamp('2026-05-01')]
cutoff=pd.Timestamp(avail[-6])

models={}
for tgt in ['target_3d','target_5d']:
    tr=panel[panel['date']<=cutoff].dropna(subset=FEATURE_COLUMNS+[tgt])
    sw=make_decay(tr); m=make_xgb()
    m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values, sample_weight=sw, verbose=False)
    models[tgt]=m

pred=panel[panel['date']==buy2].dropna(subset=FEATURE_COLUMNS); X=pred[FEATURE_COLUMNS].values
s3=pd.Series(models['target_3d'].predict(X), index=pred['stock_code'].values)
s5=pd.Series(models['target_5d'].predict(X), index=pred['stock_code'].values)
vol_s=pred.set_index('stock_code')['vol_20d']
keep=~s3.index.isin(BLACKLIST)
s3=s3[keep]; s5=s5[keep]; vol_s=vol_s[s3.index]
score=s3.rank(pct=True)*0.5 + s5.rank(pct=True)*0.5

w_v8  = conc_w(score, vol_s, score_pow=2, vol_pow=0.5).sort_values(ascending=False)
w_v9  = conc_w(score, vol_s, score_pow=3, vol_pow=0.5).sort_values(ascending=False)
w_v10 = conc_w(score, vol_s, score_pow=5, vol_pow=0.5).sort_values(ascending=False)
w_v11 = tiered_w(score, vol_s, top_tier=5, mid_tier=10, top_pct=0.50, mid_pct=0.40).sort_values(ascending=False)

print(f'\n  Top 10 concentration comparison:')
print(f'  {"rank":^4} {"code":^8} {"V0_curr":^9} {"V5_w2_022":^10} {"V8_pow2":^9} {"V9_pow3":^9} {"V10_pow5":^9} {"V11_tier":^10}')
print('-'*90)

w_v0_path = 'outputs/submissions/w2_020_3d5d_voladj_blacklist002261.csv'
w_v0 = pd.read_csv(w_v0_path, dtype={'stock_code':str})
w_v0['stock_code']=w_v0['stock_code'].str.zfill(6)
w_v0_dict = w_v0.set_index('stock_code')['weight'].to_dict()
w_v5_path = 'outputs/submissions/w2_022_softvol_blacklist002261.csv'
w_v5 = pd.read_csv(w_v5_path, dtype={'stock_code':str})
w_v5['stock_code']=w_v5['stock_code'].str.zfill(6)
w_v5_dict = w_v5.set_index('stock_code')['weight'].to_dict()

ranked = w_v9.sort_values(ascending=False)
for i, code in enumerate(ranked.index[:12], 1):
    v0  = w_v0_dict.get(code, 0)*100
    v5  = w_v5_dict.get(code, 0)*100
    v8  = w_v8.get(code, 0)*100
    v9  = w_v9.get(code, 0)*100
    v10 = w_v10.get(code, 0)*100
    v11 = w_v11.get(code, 0)*100
    print(f'  #{i:<3} {code:^8} {v0:^9.2f} {v5:^10.2f} {v8:^9.2f} {v9:^9.2f} {v10:^9.2f} {v11:^10.2f}')

print(f'\n  Concentration metrics:')
print(f'  {"variant":^14} {"max%":^7} {"top1-3%":^9} {"top1-5%":^9} {"top1-10%":^10} {"bot10%":^8}')
for label, w in [('V0_curr', pd.Series(w_v0_dict)), ('V5_w2_022', pd.Series(w_v5_dict)),
                 ('V8_pow2', w_v8), ('V9_pow3', w_v9), ('V10_pow5', w_v10), ('V11_tier', w_v11)]:
    s = w.sort_values(ascending=False)
    print(f'  {label:^14} {s.iloc[0]*100:^7.2f} {s.iloc[:3].sum()*100:^9.2f} {s.iloc[:5].sum()*100:^9.2f} {s.iloc[:10].sum()*100:^10.2f} {s.iloc[-10:].sum()*100:^8.2f}')

# Save best balanced (V9 = pow3) and aggressive (V10 = pow5)
for label, w, name in [
    ('V8_pow2', w_v8, 'w2_023_pow2_softvol'),
    ('V9_pow3', w_v9, 'w2_024_pow3_softvol'),
    ('V10_pow5', w_v10, 'w2_025_pow5_softvol'),
    ('V11_tier', w_v11, 'w2_026_tier_50_40_10'),
]:
    out=w.reset_index(); out.columns=['stock_code','weight']
    out_path=Path(f'outputs/submissions/{name}.csv')
    out.to_csv(out_path, index=False)
    print(f'  Saved: {out_path.name}')
