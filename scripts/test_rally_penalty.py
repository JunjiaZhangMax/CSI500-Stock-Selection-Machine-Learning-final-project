"""Apply rally-exhaustion penalty to w2_017 ensemble portfolio.

Tests 4 penalty schemes against the current w2_017 weights:
  P1: linear penalty on ret_5d above 5%
  P2: exponential penalty using ret_3d + ret_5d
  P3: composite (ret_5d, RSI, close/MA20) — weighted overbought score
  P4: per-stock cap reduction proportional to recent rally
"""
import sys, warnings; warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
import numpy as np, pandas as pd, xgboost as xgb
from pathlib import Path
from features import build_features, FEATURE_COLUMNS

prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

idx_df    = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_close = idx_df.set_index('date')['close']

as_of = pd.Timestamp('2026-05-08')

# ── Compute recent rally indicators for all stocks ────────────────
ret_3d_piv = close_piv.pct_change(3)
ret_5d_piv = close_piv.pct_change(5)
ret_10d_piv = close_piv.pct_change(10)

# ── Load w2_017 ───────────────────────────────────────────────────
sub = pd.read_csv('outputs/submissions/w2_017_ensemble_voladj_candidate.csv',
                  dtype={'stock_code': str})
sub['stock_code'] = sub['stock_code'].str.zfill(6)

snap = panel[panel['date']==as_of].set_index('stock_code')

print('='*88)
print(f'  w2_017 — recent rally diagnostics  (as of {as_of.date()})')
print('='*88)
print(f'  {"code":^8} {"w%":^6} {"ret3d":^7} {"ret5d":^8} {"ret10d":^8} {"rsi14":^7} {"px/ma20":^8} {"vol20":^7}  flag')
print('-'*88)

rows = []
for _, r in sub.iterrows():
    s = r['stock_code']; w = r['weight']*100
    r3  = float(ret_3d_piv.loc[as_of, s] * 100) if s in ret_3d_piv.columns else np.nan
    r5  = float(ret_5d_piv.loc[as_of, s] * 100) if s in ret_5d_piv.columns else np.nan
    r10 = float(ret_10d_piv.loc[as_of, s]* 100) if s in ret_10d_piv.columns else np.nan
    rsi = float(snap.loc[s, 'rsi_14']) if s in snap.index else np.nan
    pma = float(snap.loc[s, 'close_over_ma20']) if s in snap.index else np.nan
    v20 = float(snap.loc[s, 'vol_20d']*100) if s in snap.index else np.nan

    flags = []
    if r3 > 10:  flags.append('R3+')
    if r5 > 15:  flags.append('R5++')
    elif r5 > 10: flags.append('R5+')
    if rsi > 75: flags.append('RSI++')
    elif rsi > 70: flags.append('RSI+')
    if pma > 1.15: flags.append('MA++')
    elif pma > 1.10: flags.append('MA+')

    flag_str = ' '.join(flags) if flags else ''
    print(f'  {s:^8} {w:^6.2f} {r3:^+7.1f} {r5:^+8.1f} {r10:^+8.1f} {rsi:^7.1f} {pma:^8.3f} {v20:^7.1f}  {flag_str}')
    rows.append({'code': s, 'w0': r['weight'], 'r3': r3, 'r5': r5, 'r10': r10,
                 'rsi': rsi, 'pma': pma, 'v20': v20})

df = pd.DataFrame(rows)
print()
print(f'  Heavy rally count (ret_5d>10%):  {(df["r5"]>10).sum()}/30')
print(f'  Heavy rally count (ret_3d>10%):  {(df["r3"]>10).sum()}/30')
print(f'  RSI>70 count:                    {(df["rsi"]>70).sum()}/30')

# ═══════════════════════════════════════════════════════════════════
# Apply 4 penalty schemes
# ═══════════════════════════════════════════════════════════════════
def cap_normalize(w, cap=0.10, floor=0.001):
    w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]; free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return w / w.sum()

w0 = df['w0'].values

# P1: linear on ret_5d above 5%
penalty1 = np.maximum(0.30, 1 - 2.0 * np.maximum(0, df['r5'].values/100 - 0.05))
w1 = cap_normalize(w0 * penalty1)

# P2: exponential on combined ret_3d + ret_5d
combined = np.maximum(0, df['r3'].values/100 - 0.03) + np.maximum(0, df['r5'].values/100 - 0.05)
penalty2 = np.exp(-4.0 * combined)
w2 = cap_normalize(w0 * penalty2)

# P3: composite overbought score
# normalize each indicator to 0-1 (1 = most overbought)
def norm01(x, lo, hi):
    return np.clip((x - lo)/(hi - lo), 0, 1)
ob_r5  = norm01(df['r5'].values,   0,  20)   # 0-20%
ob_rsi = norm01(df['rsi'].values, 50,  85)   # 50-85
ob_pma = norm01(df['pma'].values,  1.0, 1.20)  # 0% to 20% above MA
overbought = 0.5*ob_r5 + 0.3*ob_rsi + 0.2*ob_pma
penalty3 = 1 - 0.6 * overbought   # max 60% reduction
w3 = cap_normalize(w0 * penalty3)

# P4: per-stock cap reduction
per_cap = 0.10 * (1 - 0.4 * ob_r5)   # cap shrinks from 10% → 6% for top movers
w4 = w0.copy()
for _ in range(500):
    mask = w4 > per_cap
    if not mask.any(): break
    excess = (w4[mask] - per_cap[mask]).sum(); w4[mask] = per_cap[mask]
    free = w4[~mask]; free = free + excess*(free/free.sum())
    w4[~mask] = free
w4 = cap_normalize(w4)

# ── compare ───────────────────────────────────────────────────────
out = pd.DataFrame({
    'code': df['code'], 'r5%': df['r5'], 'rsi': df['rsi'],
    'w0_orig': w0*100,
    'P1_linear':       w1*100,
    'P2_exp':          w2*100,
    'P3_composite':    w3*100,
    'P4_dynamic_cap':  w4*100,
}).sort_values('w0_orig', ascending=False)

print('\n' + '='*88)
print('  Penalty effect on top 15 (sorted by original weight)')
print('='*88)
print(f'  {"code":^8} {"r5%":^7} {"rsi":^6} {"w0":^7} {"P1":^7} {"P2":^7} {"P3":^7} {"P4":^7}')
print('-'*88)
for _, r in out.head(15).iterrows():
    print(f'  {r["code"]:^8} {r["r5%"]:^+7.1f} {r["rsi"]:^6.1f} '
          f'{r["w0_orig"]:^7.2f} {r["P1_linear"]:^7.2f} {r["P2_exp"]:^7.2f} '
          f'{r["P3_composite"]:^7.2f} {r["P4_dynamic_cap"]:^7.2f}')

print('\n  Concentration metrics:')
print(f'  {"scheme":^16} {"max%":^8} {"top5%":^8} {"HHI":^8} {"on_movers%"}')
for label, w in [('w0_original', w0), ('P1_linear', w1), ('P2_exp', w2),
                 ('P3_composite', w3), ('P4_dynamic_cap', w4)]:
    movers_mask = df['r5'].values > 10
    on_movers = float(w[movers_mask].sum())*100 if movers_mask.any() else 0
    hhi = float((w**2).sum())*10000  # Herfindahl, scaled
    print(f'  {label:^16} {w.max()*100:^8.2f} {sorted(w, reverse=True)[:5][-1]*100:^8.2f}'
          f' {hhi:^8.0f} {on_movers:^10.1f}')

# Save P3 (composite) as new candidate — most principled
new_w = pd.Series(w3, index=df['code'].values).sort_values(ascending=False)
out_path = Path('outputs/submissions/w2_018_ensemble_voladj_rallypenalty.csv')
out_save = new_w.reset_index(); out_save.columns = ['stock_code','weight']
out_save.to_csv(out_path, index=False)
print(f'\n  Saved P3 (composite penalty) → {out_path}')
print(f'  Sum={new_w.sum():.6f}  Max={new_w.max()*100:.2f}%  Min={new_w.min()*100:.4f}%')
