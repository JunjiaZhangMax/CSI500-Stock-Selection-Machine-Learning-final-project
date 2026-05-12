"""Compare softened vol-adj (vol_pow=0.5) vs baseline (vol_pow=1.0).

Both use cap=10%, top_k=30, 3+5 ensemble. Only difference: vol exponent.
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
idx_df['date'] = pd.to_datetime(idx_df['date']); idx_close = idx_df.set_index('date')['close']
all_dates = sorted(panel['date'].unique())

TOP_K, CAP, HL = 30, 0.10, 60

def make_decay(df, hl=60, floor=0.5):
    ds=np.sort(df['date'].unique()); d2i={pd.Timestamp(d):i for i,d in enumerate(ds)}
    return np.maximum(np.exp(-np.log(2)*((len(ds)-1)-df['date'].map(d2i).values)/hl), floor)

def make_xgb():
    return xgb.XGBRegressor(n_estimators=400,max_depth=5,learning_rate=0.05,subsample=0.8,
        colsample_bytree=0.8,min_child_weight=10,reg_lambda=1.0,tree_method='hist',n_jobs=-1,random_state=42)

def vol_adj_w(scores_s, vol_s, vol_pow=1.0, top_k=TOP_K, cap=CAP, floor=0.001):
    top=scores_s.nlargest(top_k); w=top.values-top.values.min()+1e-6; w/=w.sum()
    vols=vol_s.reindex(top.index).values.astype(float)
    med=float(np.nanmedian(vols[vols>0])); vols=np.where(np.isnan(vols)|(vols<=0),med,vols)
    w=w/(vols**vol_pow); w/=w.sum(); w=np.maximum(w,floor); w/=w.sum()
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
    for label, vp in [('V0_pow1.0', 1.0), ('V5_pow0.5', 0.5),
                      ('V6_pow0.3', 0.3), ('V7_pow0.0', 0.0)]:
        w = vol_adj_w(score, vol_s, vol_pow=vp)
        row[label] = port_ret(w, buy_date, sell_date) - bench
    records.append(row)

df=pd.DataFrame(records).dropna()
print(f'\nEvaluated {len(df)} windows\n')

strats=['V0_pow1.0','V5_pow0.5','V6_pow0.3','V7_pow0.0']
# V0 = full vol-adj (current w2_020)
# V5 = soft vol-adj (sqrt)
# V6 = very soft (cube root-ish)
# V7 = no vol-adj (pure score-prop)

print(f'  {"month":^8}{"N":^4}', '  '.join(f'{s:^14}' for s in strats))
print('-'*76)
for mo in sorted(df['month'].unique()):
    sub=df[df['month']==mo]
    vals='  '.join(f'{sub[s].mean()*100:^+14.3f}' for s in strats)
    print(f'  {mo:^8}{len(sub):^4}  {vals}')
print('-'*76)

print(f'\n  {"metric":^14}', '  '.join(f'{s:^14}' for s in strats))
print('-'*76)
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

# ── Generate W2 candidate (V5) ────────────────────────────────────
print('\n' + '='*76)
print('  W2 candidate (V5 = pow 0.5, cap 10%)')
print('='*76)

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

w_v5 = vol_adj_w(score, vol_s, vol_pow=0.5).sort_values(ascending=False)
out=w_v5.reset_index(); out.columns=['stock_code','weight']
out_path=Path('outputs/submissions/w2_022_softvol_blacklist002261.csv')
out.to_csv(out_path, index=False)
print(f'\n  Saved → {out_path}')
print(f'  Sum={w_v5.sum():.6f}  Max={w_v5.max()*100:.2f}%  N={len(w_v5)}')

# Compare side-by-side
w_v0_path = 'outputs/submissions/w2_020_3d5d_voladj_blacklist002261.csv'
w_v0 = pd.read_csv(w_v0_path, dtype={'stock_code':str})
w_v0['stock_code'] = w_v0['stock_code'].str.zfill(6)
w_v0_dict = w_v0.set_index('stock_code')['weight'].to_dict()

print()
print(f'  {"code":^8} {"V0_w%":^7} {"V5_w%":^7} {"diff_pp":^9}  {"raw_pred%":^10} {"vol%"}')
print('-'*60)
ranked = w_v5.sort_values(ascending=False)
raw_pred = ((s3+s5)/2)*100
for code in ranked.index:
    v0w = w_v0_dict.get(code, 0)*100
    v5w = w_v5[code]*100
    rp  = float(raw_pred.get(code, 0))
    vl  = float(vol_s.get(code, 0))*100
    diff = v5w - v0w
    print(f'  {code:^8} {v0w:^7.2f} {v5w:^7.2f} {diff:^+9.2f}  {rp:^+10.2f} {vl:^.1f}')
