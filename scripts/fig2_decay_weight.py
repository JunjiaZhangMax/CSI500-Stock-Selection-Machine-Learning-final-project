"""Figure 2: Time-decay sample weights and per-day training contribution.

X-axis: calendar dates (~ Jan 2026 to Apr 2026, ending at the w2_023 cutoff).
Left  Y-axis (bars):  per-day contribution to training =
                      (#stocks with target_3d on that day) * weight(day) under hl=60.
Right Y-axis (lines): the decay weight curve W(t) for hl=60 (used) and
                      hl=120 (alternative), with the floor=0.5 horizontal reference.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates

from features import build_features, FEATURE_COLUMNS

# ── Load panel ───────────────────────────────────────────────────
prices = pd.read_parquet('data/prices.parquet')
panel  = build_features(prices)

# w2_023 cutoff used in scripts/test_concentration.py
all_dates = sorted(pd.to_datetime(panel['date'].unique()))
end_date  = pd.Timestamp('2026-04-23')         # = avail-before-May1 [-6]
start_x   = pd.Timestamp('2026-01-01')

# Trading days in window
tr_dates = [d for d in all_dates if start_x <= d <= end_date]
if not tr_dates:
    raise SystemExit('No trading dates in window')

# All training dates up to cutoff, ordered (needed for delta indexing
# exactly as make_decay does it in the training scripts)
train_dates = [d for d in all_dates if d <= end_date]
last_idx    = len(train_dates) - 1
date_to_idx = {d: i for i, d in enumerate(train_dates)}

HL_MAIN, HL_ALT, FLOOR = 60, 120, 0.5
def W(delta, hl):
    return np.maximum(np.exp(-np.log(2) * delta / hl), FLOOR)

# Per-day stock count (rows with non-null target_3d, the actual training mask)
mask  = panel.dropna(subset=FEATURE_COLUMNS + ['target_3d'])
count = mask.groupby('date').size().reindex(tr_dates, fill_value=0)

deltas = np.array([last_idx - date_to_idx[d] for d in tr_dates])
w60    = W(deltas, HL_MAIN)
w120   = W(deltas, HL_ALT)
contrib = count.values * w60       # bar height = effective sample count

# ── Styling ──────────────────────────────────────────────────────
mpl.rcParams.update({
    'font.family':       'serif',
    'font.serif':        ['Times New Roman', 'DejaVu Serif'],
    'axes.edgecolor':    '#444',
    'axes.linewidth':    0.8,
    'axes.spines.top':   False,
    'axes.grid':         True,
    'grid.color':        '#ebebeb',
    'grid.linestyle':    '-',
    'grid.linewidth':    0.6,
    'axes.axisbelow':    True,
    'xtick.color':       '#333',
    'ytick.color':       '#333',
})

COLOR_BAR  = '#a8d5c8'   # mint  (bars)
COLOR_W60  = '#1a6e7a'   # deep teal (used)
COLOR_W120 = '#52b788'   # sage green (alt)
COLOR_FLOOR= '#c0392b'   # brick red (floor)

fig, ax = plt.subplots(figsize=(11.0, 5.4), dpi=150)

# Bars on the LEFT axis: per-day effective sample count
bar_dates = [d.to_pydatetime() for d in tr_dates]
ax.bar(bar_dates, contrib, width=1.0,
       color=COLOR_BAR, alpha=0.85, edgecolor='white', linewidth=0.3,
       label='Effective samples / day  ( #stocks × W(60) )')
ax.set_ylabel('Effective samples per day', fontsize=11, color='#222', labelpad=8)
ax.set_ylim(0, contrib.max() * 1.18)

# Right axis: weight curves
ax2 = ax.twinx()
ax2.plot(bar_dates, w60,  color=COLOR_W60,  lw=2.2,
         label=f'W(t),  half-life = {HL_MAIN} d   (w2_023)')
ax2.plot(bar_dates, w120, color=COLOR_W120, lw=2.2, ls='--',
         label=f'W(t),  half-life = {HL_ALT} d   (alternative)')
ax2.axhline(FLOOR, color=COLOR_FLOOR, lw=1.2, ls=':',
            label=f'floor = {FLOOR}')

ax2.set_ylim(0.0, 1.08)
ax2.set_ylabel('Sample weight  W(t)', fontsize=11, color='#222', labelpad=8)
ax2.spines['top'].set_visible(False)
ax2.grid(False)

# X-axis: monthly ticks
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
ax.set_xlim(bar_dates[0], bar_dates[-1])
for tk in ax.get_xticklabels():
    tk.set_rotation(0)
    tk.set_ha('center')

# Annotate the cutoff
ax.axvline(end_date.to_pydatetime(), color='#444', lw=0.8, ls=(0, (4, 4)))
ax.text(end_date.to_pydatetime(), contrib.max() * 1.12,
        f'  cutoff {end_date.date()}', fontsize=9, color='#444',
        va='center', ha='left')

# Title + legend (combined from both axes)
ax.set_title('Figure 2.  Time-decay sample weights and per-day training contribution',
             fontsize=13, color='#1a3a6b', pad=14, loc='left', fontweight='bold')

handles1, labels1 = ax.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax.legend(handles1 + handles2, labels1 + labels2,
          loc='lower left', frameon=False, fontsize=9.5,
          handletextpad=0.5, labelspacing=0.4)

plt.tight_layout()
out_dir = Path('outputs/figures')
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / 'fig2_decay_weight.png'
plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')

# Save the underlying numbers
out_df = pd.DataFrame({
    'date': tr_dates,
    'n_stocks': count.values,
    'delta_days': deltas,
    'W_hl60':   w60,
    'W_hl120':  w120,
    'eff_samples_hl60': contrib,
})
out_df.to_csv(out_dir / 'fig2_decay_weight.csv', index=False)
print(f'Saved data: {out_dir / "fig2_decay_weight.csv"}')
