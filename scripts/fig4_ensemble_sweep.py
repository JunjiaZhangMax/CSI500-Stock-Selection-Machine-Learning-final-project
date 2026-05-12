"""Figure 4: Ensemble horizon sweep — Sharpe vs std% scatter.

Replays the tune_ensemble.py walk-forward (Mar-Apr 2026, 5-day non-overlapping
windows, monthly retrain) and plots each configuration as a point in the
(Sharpe, std%) plane.  Lower std% + higher Sharpe = bottom-right corner = better.
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

from features import build_features, FEATURE_COLUMNS

# ── Data ─────────────────────────────────────────────────────────
prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')

for n, col in [(5, 'target_5d'), (7, 'target_7d'), (10, 'target_10d')]:
    t = (close_piv.shift(-n)/close_piv - 1).stack().reset_index()
    t.columns = ['date','stock_code', col]
    panel = panel.merge(t, on=['date','stock_code'], how='left')

idx_df = pd.read_parquet('data/index.parquet').sort_values('date')
idx_df['date'] = pd.to_datetime(idx_df['date'])
idx_close = idx_df.set_index('date')['close']
all_dates = sorted(pd.to_datetime(panel['date'].unique()))

TOP_K, CAP, HL = 30, 0.10, 60
TARGETS = ['target_3d', 'target_5d', 'target_7d', 'target_10d']

def make_decay(df, hl=HL, floor=0.5):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    return np.maximum(np.exp(-np.log(2)*((len(ds)-1)-df['date'].map(d2i).values)/hl), floor)

def make_xgb(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=seed,
    )

def vol_adj_w(scores_s, vol_s, top_k=TOP_K, cap=CAP, floor=0.001):
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6
    w   = w / w.sum()
    vols = vol_s.reindex(top.index).values.astype(float)
    med = float(np.nanmedian(vols[vols>0]))
    vols = np.where(np.isnan(vols)|(vols<=0), med, vols)
    w   = w / vols; w /= w.sum()
    w   = np.maximum(w, floor); w /= w.sum()
    for _ in range(500):
        mask = w > cap
        if not mask.any(): break
        excess = (w[mask]-cap).sum(); w[mask] = cap
        free = w[~mask]
        free = np.maximum(free + excess*(free/free.sum()), floor)
        w[~mask] = free; w /= w.sum()
    return pd.Series(w/w.sum(), index=top.index)

def port_ret(weights, buy_date, sell_date):
    bp = close_piv.loc[buy_date].reindex(weights.index)
    sp = close_piv.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna()) & (~sp.isna())
    if valid.sum() == 0: return np.nan
    wv = weights[valid]/weights[valid].sum()
    return float((wv*(sp[valid]/bp[valid]-1)).sum())

# ── Train monthly models for each target ─────────────────────────
start_eval = pd.Timestamp('2026-03-01')
end_eval   = pd.Timestamp('2026-05-08')
eval_dates = [d for d in all_dates if start_eval <= d <= end_eval]
month_starts = sorted({d.replace(day=1) for d in eval_dates})

print('Training monthly models per target (3d, 5d, 7d, 10d)...')
model_cache = {}
for ms in month_starts:
    avail = [d for d in all_dates if d < ms]
    if len(avail) < 86: continue
    cutoff = avail[-11]
    model_cache[ms] = {'cutoff': cutoff, 'models': {}}
    for tgt in TARGETS:
        tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS+[tgt])
        if len(tr) < 5000: continue
        m = make_xgb()
        m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values,
              sample_weight=make_decay(tr), verbose=False)
        model_cache[ms]['models'][tgt] = m
    print(f'  {ms.strftime("%Y-%m")}: cutoff={cutoff.date()}')

def get_models(d):
    cands = [ms for ms in model_cache if ms <= d]
    return model_cache[max(cands)] if cands else None

# ── Configurations to sweep ──────────────────────────────────────
CONFIGS = [
    ('3d only',         {'target_3d': 1.0},                                   False),
    ('5d only',         {'target_5d': 1.0},                                   False),
    ('7d only',         {'target_7d': 1.0},                                   False),
    ('10d only',        {'target_10d': 1.0},                                  False),
    ('3+5 equal',       {'target_3d': 0.5, 'target_5d': 0.5},                 True),
    ('5+7 equal',       {'target_5d': 0.5, 'target_7d': 0.5},                 True),
    ('5+10 equal',      {'target_5d': 0.5, 'target_10d': 0.5},                True),
    ('3+10 equal',      {'target_3d': 0.5, 'target_10d': 0.5},                True),
    ('3+5+10 equal',    {'target_3d': 1/3, 'target_5d': 1/3, 'target_10d': 1/3}, True),
    ('3+5+10 5d-heavy', {'target_3d': 0.2, 'target_5d': 0.6, 'target_10d': 0.2}, True),
    ('3+5+10 long-bias',{'target_3d': 0.2, 'target_5d': 0.3, 'target_10d': 0.5}, True),
    ('3+5+7+10 equal',  {'target_3d': 0.25,'target_5d': 0.25,'target_7d': 0.25,'target_10d': 0.25}, True),
]

# ── Walk-forward ─────────────────────────────────────────────────
print('\nWalk-forward...')
records = []
for i, d in enumerate(all_dates):
    buy_date = pd.Timestamp(d)
    if not (start_eval <= buy_date <= end_eval): continue
    if i + 5 >= len(all_dates): continue
    sell_date = pd.Timestamp(all_dates[i+5])
    cache = get_models(buy_date)
    if cache is None or len(cache['models']) < 4: continue

    pred_df = panel[panel['date'] == buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K: continue
    X     = pred_df[FEATURE_COLUMNS].values
    codes = pred_df['stock_code'].values
    vol_s = pred_df.set_index('stock_code')['vol_20d']

    preds = {tgt: pd.Series(cache['models'][tgt].predict(X), index=codes) for tgt in TARGETS}

    bench = float(idx_close.loc[sell_date]/idx_close.loc[buy_date] - 1)
    row = {'buy': buy_date}
    for label, wmap, use_rank in CONFIGS:
        score = pd.Series(0.0, index=codes)
        for tgt, wt in wmap.items():
            s = preds[tgt]
            if use_rank: s = s.rank(pct=True)
            score = score + wt * s
        w = vol_adj_w(score, vol_s)
        row[label] = port_ret(w, buy_date, sell_date) - bench
    records.append(row)

df = pd.DataFrame(records).dropna()
print(f'  {len(df)} windows evaluated')

# Aggregate
results = []
for label, _, _ in CONFIGS:
    s = df[label]
    results.append({
        'config': label,
        'mean%':  s.mean()*100,
        'std%':   s.std()*100,
        'sharpe': s.mean()/s.std(),
    })
res = pd.DataFrame(results).sort_values('sharpe', ascending=False).reset_index(drop=True)
print('\n' + res.to_string(index=False, float_format='%.4f'))

# ── Plot ─────────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.edgecolor':    '#5b6770',
    'axes.linewidth':    0.9,
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.color':        '#eef0f2',
    'grid.linestyle':    '-',
    'grid.linewidth':    0.7,
    'axes.axisbelow':    True,
    'xtick.color':       '#333',
    'ytick.color':       '#333',
})

# Drop near-duplicate variants in the crowded triple/quad cluster
KEEP = {
    '3d only', '5d only', '7d only', '10d only',
    '3+5 equal', '5+7 equal', '5+10 equal',
    '3+5+10 equal', '3+5+7+10 equal',
}
res_plot = res[res['config'].isin(KEEP)].reset_index(drop=True)

# Size = number of distinct horizons used (1..4)
def n_horizons(label):
    return 1 if '+' not in label else label.count('+') + 1

# Blue → green → yellow gradient over horizon count
COLOR_BY_N = {
    1: '#f4c95d',   # warm yellow  (single horizon — simplest)
    2: '#79c267',   # leafy green
    3: '#2a9d8f',   # teal
    4: '#1f5f8b',   # deep blue   (most complex)
}
SIZE_BY_N = {1: 180, 2: 380, 3: 580, 4: 760}
COLOR_FRONTIER = '#264653'
COLOR_SELECTED_RING = '#bc4749'

fig, ax = plt.subplots(figsize=(11.5, 7.0), dpi=150)

# Efficient frontier on the kept points
sorted_pts = res_plot.sort_values('sharpe', ascending=False).reset_index(drop=True)
frontier, min_std = [], np.inf
for _, r in sorted_pts.iterrows():
    if r['std%'] < min_std:
        frontier.append((r['sharpe'], r['std%']))
        min_std = r['std%']
fx = [p[0] for p in frontier]
fy = [p[1] for p in frontier]
ax.plot(fx, fy, color=COLOR_FRONTIER, lw=1.4, ls=(0, (6, 4)), alpha=0.6,
        zorder=1, label='Efficient frontier')

# Bubble scatter (size = #horizons, color = #horizons gradient)
for _, r in res_plot.iterrows():
    n = n_horizons(r['config'])
    is_sel = (r['config'] == '3+5 equal')
    ax.scatter(r['sharpe'], r['std%'],
               s=SIZE_BY_N[n], color=COLOR_BY_N[n],
               alpha=0.78,
               edgecolors=COLOR_SELECTED_RING if is_sel else 'white',
               linewidths=2.2 if is_sel else 1.2,
               zorder=4)

# Manual label placement to avoid overlap
LABEL_OFFSETS = {
    '3+5 equal':       ( 22,   0, 'left',  'center'),
    '5+7 equal':       (  0, -22, 'center', 'top'),
    '5+10 equal':      (  0, -22, 'center', 'top'),
    '3+5+10 equal':    (  0,  24, 'center', 'bottom'),
    '3+5+7+10 equal':  (-22,  20, 'right', 'bottom'),
    '3d only':         ( 18,   0, 'left',  'center'),
    '5d only':         (-18,   0, 'right', 'center'),
    '10d only':        (-16,  -8, 'right', 'top'),
    '7d only':         ( 16,  -2, 'left',  'center'),
}
for _, r in res_plot.iterrows():
    is_sel = (r['config'] == '3+5 equal')
    dx, dy, ha, va = LABEL_OFFSETS.get(r['config'], (12, 6, 'left', 'bottom'))
    ax.annotate(r['config'],
                xy=(r['sharpe'], r['std%']),
                xytext=(dx, dy), textcoords='offset points',
                fontsize=10.5 if is_sel else 9.8,
                color=COLOR_SELECTED_RING if is_sel else '#333',
                fontweight='bold' if is_sel else 'normal',
                ha=ha, va=va)

# Axes
ax.set_xlabel('Walk-forward Sharpe  (mean / std of per-window excess return)',
              fontsize=11.5, color='#222', labelpad=10)
ax.set_ylabel('Volatility of excess return — std (%)',
              fontsize=11.5, color='#222', labelpad=10)

x_min, x_max = res_plot['sharpe'].min(), res_plot['sharpe'].max()
y_min, y_max = res_plot['std%'].min(),   res_plot['std%'].max()
x_pad = (x_max - x_min) * 0.13
y_pad = (y_max - y_min) * 0.18
ax.set_xlim(x_min - x_pad, x_max + x_pad)
ax.set_ylim(y_min - y_pad*0.85, y_max + y_pad*0.9)

# "better" arrow toward bottom-right
ax.annotate('better',
            xy=(x_max + x_pad*0.45, y_min - y_pad*0.55),
            xytext=(x_max - x_pad*0.05, y_min - y_pad*0.05),
            arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0),
            fontsize=10, color='#888', style='italic',
            ha='center', va='center')

ax.set_title('Figure 4.  Ensemble horizon sweep — risk-return scatter  '
             '(Mar–Apr 2026 walk-forward)',
             fontsize=13.5, color='#264653', pad=14, loc='left', fontweight='bold')

# ── Size-key legend in the BOTTOM-LEFT (like the reference figure) ──
# Build a small in-axes legend box showing the 4 bubble sizes
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

key_xfrac, key_yfrac = 0.018, 0.025          # bottom-left in axes coords
key_w, key_h         = 0.22, 0.30
# Background box
bg = FancyBboxPatch((key_xfrac, key_yfrac), key_w, key_h,
                    transform=ax.transAxes,
                    boxstyle='round,pad=0.012,rounding_size=0.012',
                    facecolor='white', edgecolor='#c8d0d6', linewidth=0.9,
                    zorder=6, alpha=0.97)
ax.add_patch(bg)

ax.text(key_xfrac + key_w/2, key_yfrac + key_h - 0.035, 'Horizons used',
        transform=ax.transAxes, ha='center', va='center',
        fontsize=10.5, color='#264653', fontweight='bold', zorder=7)

# 4 rows: marker + label
rows_y = np.linspace(key_yfrac + key_h - 0.085, key_yfrac + 0.035, 4)
LABELS = {1: '1  horizon', 2: '2  horizons', 3: '3  horizons', 4: '4  horizons'}
for i, n in enumerate([1, 2, 3, 4]):
    # marker (scatter in axes coords)
    ax.scatter([key_xfrac + 0.04], [rows_y[i]],
               s=SIZE_BY_N[n] * 0.55, color=COLOR_BY_N[n],
               alpha=0.85, edgecolors='white', linewidths=1.0,
               transform=ax.transAxes, zorder=7, clip_on=False)
    ax.text(key_xfrac + 0.085, rows_y[i], LABELS[n],
            transform=ax.transAxes, ha='left', va='center',
            fontsize=9.5, color='#333', family='monospace', zorder=7)

# Frontier legend (small, top-right, unobtrusive)
frontier_handle = Line2D([], [], color=COLOR_FRONTIER, lw=1.4,
                         ls=(0, (6, 4)), label='Efficient frontier')
selected_handle = Line2D([], [], marker='o', markersize=11,
                         markeredgecolor=COLOR_SELECTED_RING,
                         markerfacecolor='#2a9d8f',
                         markeredgewidth=2.0, linestyle='None',
                         label='Selected  (3+5 equal)')
ax.legend(handles=[frontier_handle, selected_handle],
          loc='upper right', frameon=False, fontsize=10,
          handletextpad=0.7, labelspacing=0.5)

out_dir = Path('outputs/figures')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'fig4_ensemble_sweep.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {out_path}')

res.to_csv(out_dir / 'fig4_ensemble_sweep.csv', index=False)
print(f'Saved data: {out_dir / "fig4_ensemble_sweep.csv"}')
