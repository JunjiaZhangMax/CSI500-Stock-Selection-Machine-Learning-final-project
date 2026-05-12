"""Figure 3: Cumulative portfolio return vs CSI500 over the walk-forward period.

Replays the exact w2_023 recipe (monthly retrain, 3+5 ensemble, vol_pow=0.5,
score_pow=2, top-30, cap 10%, 002261 blacklist) over Oct 2025 - May 2026.

Cumulative curves are built by chaining 5-trading-day holding windows
(buy at close, sell 5 days later) so the equity curves are comparable.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from features import build_features, FEATURE_COLUMNS

BLACKLIST = {'002261'}
TOP_K, CAP = 30, 0.10
HL = 60
SCORE_POW, VOL_POW = 2.0, 0.5

# ── Data ─────────────────────────────────────────────────────────
prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

t5 = (close_piv.shift(-5)/close_piv - 1).stack().reset_index()
t5.columns = ['date','stock_code','target_5d']
panel = panel.merge(t5, on=['date','stock_code'], how='left')

idx_df   = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_close = idx_df.set_index('date')['close']

all_dates = sorted(pd.to_datetime(panel['date'].unique()))

# ── Helpers ──────────────────────────────────────────────────────
def make_decay(df, hl=HL, floor=0.5):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds)-1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2)*delta/hl), floor)

def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42,
    )

def conc_w(scores_s, vol_s, score_pow=SCORE_POW, vol_pow=VOL_POW,
           top_k=TOP_K, cap=CAP, floor=0.001):
    top = scores_s.nlargest(top_k)
    s_norm = (top.values - top.values.min()) / (top.values.max() - top.values.min() + 1e-9)
    s_norm = s_norm + 1e-3
    w = s_norm ** score_pow
    w = w / w.sum()
    vols = vol_s.reindex(top.index).values.astype(float)
    med = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    w = w / (vols ** vol_pow)
    w = w / w.sum()
    w = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]
        free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free
        w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def port_ret(weights, buy_date, sell_date):
    bp = close_piv.loc[buy_date].reindex(weights.index)
    sp = close_piv.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna()) & (~sp.isna())
    if valid.sum() == 0: return np.nan
    wv = weights[valid] / weights[valid].sum()
    return float((wv * (sp[valid]/bp[valid] - 1)).sum())

# ── Walk-forward window ──────────────────────────────────────────
start_eval = pd.Timestamp('2025-10-01')
end_eval   = pd.Timestamp('2026-05-08')
eval_dates = [d for d in all_dates if start_eval <= d <= end_eval]
month_starts = sorted({d.replace(day=1) for d in eval_dates})

print('Training monthly models (target_3d, target_5d)...')
model_cache = {}
for ms in month_starts:
    avail = [d for d in all_dates if d < ms]
    if len(avail) < 86: continue
    cutoff = avail[-6]   # 5-day embargo
    model_cache[ms] = {'cutoff': cutoff, 'models': {}}
    for tgt in ['target_3d', 'target_5d']:
        tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS + [tgt])
        if len(tr) < 5000: continue
        m = make_xgb()
        m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values,
              sample_weight=make_decay(tr), verbose=False)
        model_cache[ms]['models'][tgt] = m
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}')

def get_models(d):
    cands = [ms for ms in model_cache if ms <= d]
    return model_cache[max(cands)] if cands else None

# ── Non-overlapping 5-day windows ────────────────────────────────
print('\nWalk-forward (non-overlapping 5-day windows)...')
records = []
i = 0
date_idx = {d: k for k, d in enumerate(all_dates)}
# Find the first eval date in all_dates
first_eval_idx = next(k for k, d in enumerate(all_dates) if d >= start_eval)

k = first_eval_idx
while k + 5 < len(all_dates):
    buy_date  = all_dates[k]
    sell_date = all_dates[k+5]
    if buy_date > end_eval: break
    cache = get_models(buy_date)
    if cache is None or len(cache['models']) < 2:
        k += 5; continue

    pred_df = panel[panel['date'] == buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        k += 5; continue
    X  = pred_df[FEATURE_COLUMNS].values
    s3 = pd.Series(cache['models']['target_3d'].predict(X), index=pred_df['stock_code'].values)
    s5 = pd.Series(cache['models']['target_5d'].predict(X), index=pred_df['stock_code'].values)
    vol_s = pred_df.set_index('stock_code')['vol_20d']
    keep = ~s3.index.isin(BLACKLIST)
    s3 = s3[keep]; s5 = s5[keep]; vol_s = vol_s[s3.index]
    score = s3.rank(pct=True)*0.5 + s5.rank(pct=True)*0.5

    w = conc_w(score, vol_s)
    r_port  = port_ret(w, buy_date, sell_date)
    r_bench = float(idx_close.loc[sell_date]/idx_close.loc[buy_date] - 1)

    records.append({
        'buy':  buy_date,
        'sell': sell_date,
        'r_port':  r_port,
        'r_bench': r_bench,
        'excess':  r_port - r_bench,
    })
    k += 5

df = pd.DataFrame(records).dropna()
print(f'\nWalk-forward: {len(df)} non-overlapping windows')

# Compounded equity curves, anchored at the buy_date of the first window
df = df.sort_values('buy').reset_index(drop=True)
df['eq_port']  = (1 + df['r_port']).cumprod()
df['eq_bench'] = (1 + df['r_bench']).cumprod()

# Pre-pend an anchor row at value 1.0 on first buy_date
anchor = pd.DataFrame([{
    'sell': df['buy'].iloc[0], 'eq_port': 1.0, 'eq_bench': 1.0,
}])
plot_df = pd.concat([anchor[['sell','eq_port','eq_bench']],
                     df[['sell','eq_port','eq_bench']]], ignore_index=True)

# ── Styling ──────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.edgecolor':    '#444',
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#ebebeb',
    'grid.linestyle':    '-',
    'grid.linewidth':    0.6,
    'axes.axisbelow':    True,
    'xtick.color':       '#333',
    'ytick.color':       '#333',
})

COLOR_PORT  = '#1a6e7a'   # deep teal
COLOR_BENCH = '#52b788'   # sage green
COLOR_GAP   = '#a8d5c8'   # mint fill

fig, ax = plt.subplots(figsize=(11.5, 5.8), dpi=150)

dates = pd.to_datetime(plot_df['sell']).dt.to_pydatetime()
eq_p  = (plot_df['eq_port']  - 1) * 100
eq_b  = (plot_df['eq_bench'] - 1) * 100

# Shaded outperformance area
ax.fill_between(dates, eq_b, eq_p, where=(eq_p >= eq_b),
                color=COLOR_GAP, alpha=0.55, linewidth=0,
                label='Outperformance')

ax.plot(dates, eq_p, color=COLOR_PORT, lw=2.4,
        label=f'w2_023 portfolio  ({len(df)} non-overlapping 5-day windows)')
ax.plot(dates, eq_b, color=COLOR_BENCH, lw=2.0, ls='--',
        label='CSI500 benchmark')

# Annotate final values
fin_p, fin_b = eq_p.iloc[-1], eq_b.iloc[-1]
ax.annotate(f'+{fin_p:.1f}%',
            xy=(dates[-1], fin_p),
            xytext=(8, 0), textcoords='offset points',
            fontsize=10, color=COLOR_PORT, fontweight='bold',
            va='center')
ax.annotate(f'+{fin_b:.1f}%',
            xy=(dates[-1], fin_b),
            xytext=(8, 0), textcoords='offset points',
            fontsize=10, color=COLOR_BENCH,
            va='center')

# Horizontal zero line
ax.axhline(0, color='#888', lw=0.7, ls=':')

ax.set_ylabel('Cumulative return (%)', fontsize=11, color='#222', labelpad=8)
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')

ax.set_title('Figure 3.  Cumulative return — w2_023 portfolio vs CSI500  '
             '(Oct 2025 – May 2026, walk-forward)',
             fontsize=13, color='#1a3a6b', pad=14, loc='left', fontweight='bold')

ax.legend(loc='upper left', frameon=False, fontsize=10,
          handletextpad=0.5, labelspacing=0.4)

# Give the right side a bit of breathing room for the annotations
ax.set_xlim(dates[0], dates[-1] + pd.Timedelta(days=8))

plt.tight_layout()
out_dir = Path('outputs/figures')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'fig3_cumulative_return.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {out_path}')

# Save the data table
df.to_csv(out_dir / 'fig3_cumulative_return.csv', index=False)
print(f'Saved data: {out_dir / "fig3_cumulative_return.csv"}')

# Summary
print('\nSummary:')
print(f'  Portfolio  cumulative return: {fin_p:+.2f}%')
print(f'  CSI500     cumulative return: {fin_b:+.2f}%')
print(f'  Outperformance:               {fin_p - fin_b:+.2f} pp')
print(f'  Win rate (excess > 0):        {(df["excess"]>0).mean()*100:.1f}%')
print(f'  Mean per-window excess:       {df["excess"].mean()*100:+.2f}%')
