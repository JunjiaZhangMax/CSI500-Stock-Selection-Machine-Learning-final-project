"""April 2026 daily backtest for exp_019 (score_prop) vs rank_linear.
Uses the final exp_017 model (train cutoff 2026-04-08, same as submission).
Each April trading day: sell=day, buy=3 trading days earlier.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from scipy.stats import spearmanr
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame
from portfolio import build_portfolio

DATA_DIR = Path('data')
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

TOP_K = 50
MAX_W = 0.10

# --- train final model (same as exp_019 submission) ---
train_df     = training_frame(panel, max_date='2026-04-08', target=TARGET_COLUMN)
dates_sorted = np.sort(train_df['date'].unique())
d2i   = {pd.Timestamp(d): i for i, d in enumerate(dates_sorted)}
n     = len(dates_sorted)
delta = (n - 1) - train_df['date'].map(d2i).values
sw    = np.maximum(np.exp(-np.log(2) * delta / 120), 0.5)
model = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)
model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN], sample_weight=sw, verbose=False)
print("Model trained (cutoff 2026-04-08).\n")


def score_prop_w(scores_series):
    top = scores_series.nlargest(TOP_K)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()
    for _ in range(50):
        mask = w > MAX_W
        if not mask.any():
            break
        excess   = (w[mask] - MAX_W).sum()
        w[mask]  = MAX_W
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    w /= w.sum()
    return pd.Series(w, index=top.index)


def rank_linear_w(scores_series):
    return build_portfolio(scores_series, top_k=TOP_K)


# --- April daily loop ---
april_days = [pd.Timestamp(d) for d in all_trading
              if pd.Timestamp(d).year == 2026 and pd.Timestamp(d).month == 4]

rows = []
for sd in april_days:
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd = pd.Timestamp(all_trading[si - 3])
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS).copy()
    if len(pred_df) < TOP_K:
        continue

    scores  = pd.Series(model.predict(pred_df[FEATURE_COLUMNS]),
                        index=pred_df['stock_code'].values)
    buy_p   = panel_close.loc[bd]
    sell_p  = panel_close.loc[sd]
    bench   = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)

    # rank IC (all stocks)
    act_all = (sell_p.reindex(scores.index) / buy_p.reindex(scores.index) - 1).dropna()
    sc_all  = scores.reindex(act_all.index).dropna()
    common  = act_all.index.intersection(sc_all.index)
    ic = spearmanr(sc_all[common], act_all[common])[0] if len(common) > 10 else np.nan

    for label, wt_fn in [('score_prop', score_prop_w), ('rank_linear', rank_linear_w)]:
        weights = wt_fn(scores)
        bp = buy_p.reindex(weights.index)
        sp_ = sell_p.reindex(weights.index)
        valid = (~bp.isna()) & (~sp_.isna())
        w_v   = weights[valid] / weights[valid].sum()
        port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
        # top-10 concentration
        top10_w = weights.nlargest(10).sum()
        rows.append({
            'sell': sd, 'buy': bd, 'strategy': label,
            'port': port, 'bench': bench, 'excess': port - bench,
            'ic': ic, 'top10_conc': top10_w,
        })

df = pd.DataFrame(rows)
df_sp = df[df['strategy'] == 'score_prop'].reset_index(drop=True)
df_rl = df[df['strategy'] == 'rank_linear'].reset_index(drop=True)

# --- per-day table ---
print(f"{'sell':^12}{'buy':^12}{'bench%':^8} | "
      f"{'sp_port%':^10}{'sp_exc%':^10} | "
      f"{'rl_port%':^10}{'rl_exc%':^10} | "
      f"{'IC':^8}{'sp_top10':^9}")
print('-' * 90)
for i in range(len(df_sp)):
    r  = df_sp.iloc[i]
    rl = df_rl.iloc[i]
    sp_mark = 'W' if r['excess'] > 0 else 'L'
    rl_mark = 'W' if rl['excess'] > 0 else 'L'
    print(f"{str(r['sell'].date()):^12}{str(r['buy'].date()):^12}{r['bench']*100:^+8.2f} | "
          f"{r['port']*100:^+10.3f}{r['excess']*100:^+9.3f}{sp_mark} | "
          f"{rl['port']*100:^+10.3f}{rl['excess']*100:^+9.3f}{rl_mark} | "
          f"{r['ic']:^+8.3f}{r['top10_conc']:^9.3f}")

# --- summary ---
print('\n' + '='*90)
print(f"{'指标':^22} {'score_prop':^16} {'rank_linear':^16}")
print('-'*56)
for label, fn in [
    ('mean_port%',   lambda d: d['port'].mean()*100),
    ('mean_excess%', lambda d: d['excess'].mean()*100),
    ('std_excess%',  lambda d: d['excess'].std()*100),
    ('sharpe',       lambda d: d['excess'].mean()/d['excess'].std() if d['excess'].std()>0 else 0),
    ('win_rate',     lambda d: (d['excess']>0).mean()),
    ('max_win%',     lambda d: d['excess'].max()*100),
    ('max_loss%',    lambda d: d['excess'].min()*100),
    ('mean_IC',      lambda d: d['ic'].mean()),
]:
    print(f"  {label:^20}  {fn(df_sp):^+16.4f}  {fn(df_rl):^+16.4f}")

# --- week-by-week ---
print('\n月内分段超额 (score_prop):')
segments = [
    ('前5日 (4/01~4/09)',  '2026-04-01', '2026-04-09'),
    ('中5日 (4/10~4/17)',  '2026-04-10', '2026-04-17'),
    ('后7日 (4/20~4/30)',  '2026-04-20', '2026-04-30'),
]
for name, s, e in segments:
    sub = df_sp[(df_sp['sell'] >= s) & (df_sp['sell'] <= e)]
    ex  = sub['excess']
    print(f"  {name}: mean={ex.mean()*100:+.3f}%  "
          f"win={(ex>0).mean():.2f}  N={len(sub)}")

# --- worst & best days detail ---
print('\n最佳3天持仓明细 (score_prop):')
for _, row in df_sp.nlargest(3, 'excess').iterrows():
    sd = row['sell']; bd = row['buy']
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    scores  = pd.Series(model.predict(pred_df[FEATURE_COLUMNS]),
                        index=pred_df['stock_code'].values)
    weights = score_prop_w(scores)
    bp = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    ret = (sp_[valid] / bp[valid] - 1)
    top5 = weights[valid].nlargest(5)
    detail = '  '.join(f"{c}:{ret[c]*100:+.1f}%({w*100:.1f}%)" for c, w in top5.items())
    print(f"  {sd.date()} (超额{row['excess']*100:+.2f}%): {detail}")

print('\n最差3天持仓明细 (score_prop):')
for _, row in df_sp.nsmallest(3, 'excess').iterrows():
    sd = row['sell']; bd = row['buy']
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    scores  = pd.Series(model.predict(pred_df[FEATURE_COLUMNS]),
                        index=pred_df['stock_code'].values)
    weights = score_prop_w(scores)
    bp = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    ret = (sp_[valid] / bp[valid] - 1)
    top5 = weights[valid].nlargest(5)
    detail = '  '.join(f"{c}:{ret[c]*100:+.1f}%({w*100:.1f}%)" for c, w in top5.items())
    print(f"  {sd.date()} (超额{row['excess']*100:+.2f}%): {detail}")
