"""Compare v5d / v10d / v20d vol-adjustment on Window-1 return and April stability."""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from features import build_features, FEATURE_COLUMNS

prices   = pd.read_parquet('data/prices.parquet')
panel    = build_features(prices)

# ── build_features already provides: vol_5d, target_3d ───────────
# only need to add vol_10d and target_5d
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
ret_daily = close_piv.pct_change()

v10_piv = ret_daily.rolling(10).std() * np.sqrt(252)
v10_lng = v10_piv.stack().reset_index(); v10_lng.columns = ['date','stock_code','vol_10d']
panel = panel.merge(v10_lng, on=['date','stock_code'], how='left')

t5 = (close_piv.shift(-5)/close_piv - 1).stack().reset_index()
t5.columns = ['date','stock_code','target_5d']
panel = panel.merge(t5, on=['date','stock_code'], how='left')

all_dates = sorted(panel['date'].unique())
idx_df    = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_close = idx_df.set_index('date')['close']

TOP_K, CAP, HL = 30, 0.10, 60

def make_decay(df, hl=60, floor=0.5):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds)-1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2)*delta/hl), floor)

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)

def vol_adj_w(scores_s, vol_s, top_k=TOP_K, cap=CAP, floor=0.001):
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6; w /= w.sum()
    vols = vol_s.reindex(top.index).values.astype(float)
    med  = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    w = w/vols; w /= w.sum()
    w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free+excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def port_ret(weights, buy_date, sell_date):
    bp  = close_piv.loc[buy_date].reindex(weights.index)
    sp_ = close_piv.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna())&(~sp_.isna())
    if valid.sum()==0: return np.nan
    wv = weights[valid]/weights[valid].sum()
    return float((wv*(sp_[valid]/bp[valid]-1)).sum())

SEP = '='*65
VOL_COLS = [('vol_5d', 'v5d'), ('vol_10d', 'v10d'), ('vol_20d', 'v20d')]
LABELS   = ['v5d', 'v10d', 'v20d']

# ═══════════════════════════════════════════════════════════════════
# TEST 1: Window 1 — 3d-target, buy=Apr30, sell=May6 (+3 trading days)
# ═══════════════════════════════════════════════════════════════════
print(SEP)
print('TEST 1: Window 1  (3d-target | buy=Apr30 | sell=May6)')
print(SEP)

buy1  = pd.Timestamp('2026-04-30')
sell1 = pd.Timestamp('2026-05-08')   # 3 trading days after Apr30 (May6/7/8)

avail   = [d for d in all_dates if pd.Timestamp(d) < buy1]
cutoff1 = pd.Timestamp(avail[-6])
print(f'Cutoff: {cutoff1.date()}')

tr1 = panel[panel['date']<=cutoff1].dropna(subset=FEATURE_COLUMNS+['target_3d'])
sw1 = make_decay(tr1, hl=HL)
m1  = make_xgb()
m1.fit(tr1[FEATURE_COLUMNS].values, tr1['target_3d'].values, sample_weight=sw1, verbose=False)
print(f'Trained on {len(tr1):,} rows (3d-target)\n')

pred1   = panel[panel['date']==buy1].dropna(subset=FEATURE_COLUMNS)
scores1 = pd.Series(m1.predict(pred1[FEATURE_COLUMNS].values), index=pred1['stock_code'].values)
bench1  = float(idx_close.loc[sell1]/idx_close.loc[buy1]-1)

print(f'  {"variant":^8}  {"portfolio":^12}  {"excess":^12}  {"top5"}')
print('-'*65)
for vol_col, label in VOL_COLS:
    vs = pred1.set_index('stock_code')[vol_col]
    w  = vol_adj_w(scores1, vs)
    p  = port_ret(w, buy1, sell1)
    top5 = '  '.join(w.nlargest(5).index.tolist())
    print(f'  {label:^8}  {p*100:^+12.2f}  {(p-bench1)*100:^+12.2f}  {top5}')
print(f'  {"bench":^8}  {bench1*100:^+12.2f}')

# Actual submission
w1_sub = pd.read_csv('outputs/submissions/w1_019_score_prop_top50_SUBMITTED.csv',
                     dtype={'stock_code': str})
w1_sub['stock_code'] = w1_sub['stock_code'].str.zfill(6)
w1_w   = w1_sub.set_index('stock_code')['weight']
p_sub  = port_ret(w1_w, buy1, sell1)
print(f'  {"ACTUAL":^8}  {p_sub*100:^+12.2f}  {(p_sub-bench1)*100:^+12.2f}  (score-prop top50 submitted)')
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 2: April stability — 5d-target, monthly retrain
# ═══════════════════════════════════════════════════════════════════
print(SEP)
print('TEST 2: March–April stability  (5d-target, monthly retrain)')
print(SEP)

start_eval = pd.Timestamp('2026-03-01')
end_eval   = pd.Timestamp('2026-05-08')
eval_dates = [pd.Timestamp(d) for d in all_dates
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
for ms in month_starts:
    avail = [d for d in all_dates if pd.Timestamp(d) < ms]
    if len(avail) < 86: continue
    cutoff = pd.Timestamp(avail[-6])
    tr = panel[panel['date']<=cutoff].dropna(subset=FEATURE_COLUMNS+['target_5d'])
    if len(tr) < 5000: continue
    sw = make_decay(tr, hl=HL); m = make_xgb()
    m.fit(tr[FEATURE_COLUMNS].values, tr['target_5d'].values,
          sample_weight=sw, verbose=False)
    model_cache[ms] = (m, cutoff)
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}  rows={len(tr):,}')

def get_model(d):
    cands = [ms for ms in model_cache if ms <= d]
    return model_cache[max(cands)] if cands else None

records = []
for i, d in enumerate(all_dates):
    buy_date = pd.Timestamp(d)
    if not (start_eval <= buy_date <= end_eval): continue
    if i+5 >= len(all_dates): continue
    sell_date = pd.Timestamp(all_dates[i+5])
    mdl = get_model(buy_date)
    if mdl is None: continue
    m, _ = mdl
    pred_df = panel[panel['date']==buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K: continue
    scores  = pd.Series(m.predict(pred_df[FEATURE_COLUMNS].values),
                        index=pred_df['stock_code'].values)
    bench   = float(idx_close.loc[sell_date]/idx_close.loc[buy_date]-1)
    row = {'buy': buy_date, 'month': buy_date.strftime('%Y-%m'), 'bench': bench}
    for vol_col, label in VOL_COLS:
        vs = pred_df.set_index('stock_code')[vol_col]
        w  = vol_adj_w(scores, vs)
        p  = port_ret(w, buy_date, sell_date)
        row[label] = (p - bench) if not np.isnan(p) else np.nan
    records.append(row)

df = pd.DataFrame(records).dropna()
print(f'\nEvaluated {len(df)} 5-day windows\n')

labels = LABELS
print(f'  {"month":^10}{"N":^5}', '  '.join(f'{"exc_"+l+"%":^12}' for l in labels))
print('-'*58)
for mo in sorted(df['month'].unique()):
    sub = df[df['month']==mo]
    vals = '  '.join(f'{sub[l].mean()*100:^+12.3f}' for l in labels)
    print(f'  {mo:^10}{len(sub):^5}  {vals}')
print('-'*58)

print(f'\n  {"metric":^14}', '  '.join(f'{l:^12}' for l in labels))
print('-'*54)
for metric, fn in [
    ('mean_exc%',  lambda l: df[l].mean()*100),
    ('std%',       lambda l: df[l].std()*100),
    ('sharpe',     lambda l: df[l].mean()/df[l].std()),
    ('win_rate',   lambda l: (df[l]>0).mean()),
    ('max_loss%',  lambda l: df[l].min()*100),
]:
    vals = '  '.join(f'{fn(l):^12.4f}' for l in labels)
    print(f'  {metric:^14}  {vals}')
