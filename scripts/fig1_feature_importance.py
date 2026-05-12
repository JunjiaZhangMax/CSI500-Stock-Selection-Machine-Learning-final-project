"""Figure 1: Feature importance bar chart for the final w2_023 ensemble.

Trains the same two XGBoost models used for w2_023 (target_3d, target_5d),
averages their gain-based feature importances, and renders a horizontal bar
chart.
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

# ── Train the two w2_023 models ───────────────────────────────────
prices    = pd.read_parquet('data/prices.parquet')
panel     = build_features(prices)
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
t5 = (close_piv.shift(-5)/close_piv-1).stack().reset_index()
t5.columns = ['date','stock_code','target_5d']
panel = panel.merge(t5, on=['date','stock_code'], how='left')

all_dates = sorted(panel['date'].unique())
month_start = pd.Timestamp(2026, 5, 1)
avail = [d for d in all_dates if pd.Timestamp(d) < month_start]
cutoff = pd.Timestamp(avail[-6])  # 5-day embargo
print(f'Training models with cutoff {cutoff.date()}...')

def make_decay(df, hl=60, floor=0.5):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

XGB = dict(n_estimators=400, max_depth=5, learning_rate=0.05,
           subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
           reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)

importances = {}
for tgt in ['target_3d', 'target_5d']:
    tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS + [tgt])
    sw = make_decay(tr)
    m = xgb.XGBRegressor(**XGB)
    m.fit(tr[FEATURE_COLUMNS].values, tr[tgt].values, sample_weight=sw, verbose=False)
    # Gain-based importance
    booster = m.get_booster()
    gain = booster.get_score(importance_type='gain')
    # XGBoost names are f0, f1, ...; map back
    imp = np.array([gain.get(f'f{i}', 0.0) for i in range(len(FEATURE_COLUMNS))])
    imp = imp / imp.sum()
    importances[tgt] = imp
    print(f'  {tgt}: top-3 = {[FEATURE_COLUMNS[i] for i in np.argsort(imp)[::-1][:3]]}')

# Average importance across both models
imp_avg = (importances['target_3d'] + importances['target_5d']) / 2

# Build dataframe sorted by avg importance
df_imp = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'imp_3d':  importances['target_3d'],
    'imp_5d':  importances['target_5d'],
    'imp_avg': imp_avg,
}).sort_values('imp_avg', ascending=True)

# ── Styling ───────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif':  ['Times New Roman', 'DejaVu Serif'],
    'axes.edgecolor':   '#444',
    'axes.linewidth':   0.8,
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

# Blue-green palette
COLOR_3D  = '#1a7a8a'   # teal-blue   (target_3d)
COLOR_5D  = '#52b788'   # sage green  (target_5d)

# Sort descending for left-to-right readability
df_plot = df_imp.sort_values('imp_avg', ascending=False).reset_index(drop=True)

n_feat = len(df_plot)
fig, ax = plt.subplots(figsize=(13.0, 5.8), dpi=150)

x      = np.arange(n_feat)
bar_w  = 0.36

bars3 = ax.bar(x - bar_w/2, df_plot['imp_3d']*100, bar_w,
               color=COLOR_3D, label='target_3d', edgecolor='white', linewidth=0.5)
bars5 = ax.bar(x + bar_w/2, df_plot['imp_5d']*100, bar_w,
               color=COLOR_5D, label='target_5d', edgecolor='white', linewidth=0.5)

# Value labels above each bar
for bars in [bars3, bars5]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.06,
                f'{h:.1f}', ha='center', va='bottom', fontsize=7.5, color='#444')

# X-axis: feature names rotated
ax.set_xticks(x)
ax.set_xticklabels(df_plot['feature'], rotation=38, ha='right',
                   fontfamily='monospace', fontsize=9)
ax.set_ylabel('Gain-based importance (%)', fontsize=11, color='#222', labelpad=8)
y_max = max(df_plot['imp_3d'].max(), df_plot['imp_5d'].max()) * 100
ax.set_ylim(0, y_max * 1.20)

# Title block
ax.set_title('Figure 1.  XGBoost feature importance — w2_023 ensemble',
             fontsize=13, color='#1a3a6b', pad=14, loc='left', fontweight='bold')

# Legend
ax.legend(loc='upper right', frameon=False, fontsize=10,
          handletextpad=0.4, labelspacing=0.4)

# Tight layout + save
plt.tight_layout()
out_dir = Path('outputs/figures')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'fig1_feature_importance.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'\nSaved: {out_path}')

# Also save the data table
df_imp.iloc[::-1].to_csv(out_dir / 'fig1_feature_importance.csv', index=False)
print(f'Saved data: {out_dir / "fig1_feature_importance.csv"}')
