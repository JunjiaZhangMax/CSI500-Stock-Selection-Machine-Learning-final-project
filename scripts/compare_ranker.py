"""Compare rank:pairwise XGBRanker vs exp_019 (regression + score_prop).

rank:pairwise treats each trading date as a query group and learns
to rank stocks by 3-day forward return within each day. Predictions
are ranking scores (magnitude less meaningful than order).

Tests both weighting schemes on each model:
  - regression + score_prop   (exp_019 baseline)
  - regression + rank_linear
  - ranker    + rank_linear   (natural for ranking output)
  - ranker    + score_prop    (treats ranker score magnitudes as signal)
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
panel       = build_features(prices)
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

TOP_K = 50
MAX_W = 0.10
TRAIN_CUTOFF = '2026-04-08'

# ── time-decay weights ────────────────────────────────────────────
def make_decay(df, hl=120, floor=0.5):
    ds = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)

# ── weighting schemes ─────────────────────────────────────────────
def score_prop_w(scores_s):
    top = scores_s.nlargest(TOP_K)
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

def rank_linear_w(scores_s):
    return build_portfolio(scores_s, top_k=TOP_K)

# ── train regression model (exp_017/exp_019 base) ────────────────
print("Training regression model (XGBRegressor)...")
train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
sw_reg   = make_decay(train_df)

reg_model = xgb.XGBRegressor(
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)
reg_model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
              sample_weight=sw_reg, verbose=False)
print(f"  Regression model ready.  train rows: {len(train_df):,}")

# ── train rank:pairwise model ─────────────────────────────────────
print("Training rank:pairwise model (XGBRanker)...")
# sort by date so groups are contiguous
train_rank = train_df.sort_values('date').reset_index(drop=True)
sw_rank    = make_decay(train_rank)

# group sizes: number of stocks per trading date
group_sizes = train_rank.groupby('date', sort=True).size().values

# XGBRanker sample_weight must be per-group (one per query), not per sample
# Use mean decay weight within each date as the group weight
train_rank['_sw'] = sw_rank
group_weights = train_rank.groupby('date', sort=True)['_sw'].mean().values
train_rank.drop(columns=['_sw'], inplace=True)

ranker = xgb.XGBRanker(
    objective='rank:pairwise',
    n_estimators=400, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)
ranker.fit(
    train_rank[FEATURE_COLUMNS],
    train_rank[TARGET_COLUMN],      # raw return as relevance label
    group=group_sizes,
    sample_weight=group_weights,    # one weight per query group
    verbose=False,
)
print(f"  Ranker model ready.\n")

# ── April daily backtest ─────────────────────────────────────────
april_days = [pd.Timestamp(d) for d in all_trading
              if pd.Timestamp(d).year == 2026 and pd.Timestamp(d).month == 4]

configs = [
    ('reg+score_prop',  reg_model,  score_prop_w),
    ('reg+rank_linear', reg_model,  rank_linear_w),
    ('ranker+rank_lin', ranker,     rank_linear_w),
    ('ranker+score_p',  ranker,     score_prop_w),
]

results = {name: [] for name, _, _ in configs}

for sd in april_days:
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd = pd.Timestamp(all_trading[si - 3])
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS).copy()
    if len(pred_df) < TOP_K:
        continue

    buy_p  = panel_close.loc[bd]
    sell_p = panel_close.loc[sd]
    bench  = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)

    # IC (same for all configs, based on regressor scores as reference)
    scores_reg = pd.Series(reg_model.predict(pred_df[FEATURE_COLUMNS]),
                           index=pred_df['stock_code'].values)
    act_all = (sell_p.reindex(scores_reg.index) / buy_p.reindex(scores_reg.index) - 1).dropna()
    common  = scores_reg.reindex(act_all.index).dropna().index.intersection(act_all.index)
    ic_reg  = spearmanr(scores_reg[common], act_all[common])[0]

    scores_rnk = pd.Series(ranker.predict(pred_df[FEATURE_COLUMNS]),
                           index=pred_df['stock_code'].values)
    ic_rnk = spearmanr(scores_rnk[common], act_all[common])[0]

    for name, mdl, wt_fn in configs:
        scores = pd.Series(mdl.predict(pred_df[FEATURE_COLUMNS]),
                           index=pred_df['stock_code'].values)
        weights = wt_fn(scores)
        bp  = buy_p.reindex(weights.index)
        sp_ = sell_p.reindex(weights.index)
        valid = (~bp.isna()) & (~sp_.isna())
        w_v   = weights[valid] / weights[valid].sum()
        port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
        ic    = ic_reg if 'reg' in name else ic_rnk
        results[name].append({
            'sell': sd, 'port': port, 'bench': bench,
            'excess': port - bench, 'ic': ic,
        })

dfs = {k: pd.DataFrame(v) for k, v in results.items()}

# ── per-day table ────────────────────────────────────────────────
print(f"{'sell':^12}{'bench%':^8} | "
      f"{'reg+sp':^10}{'reg+rl':^10} | "
      f"{'rnk+rl':^10}{'rnk+sp':^10} | "
      f"{'IC_reg':^8}{'IC_rnk':^8}")
print('-' * 90)
n = len(dfs['reg+score_prop'])
for i in range(n):
    rows = {k: dfs[k].iloc[i] for k in dfs}
    sd   = rows['reg+score_prop']['sell']
    bm   = rows['reg+score_prop']['bench']
    def fmt(k):
        e = rows[k]['excess']*100
        return f"{e:^+10.3f}"
    ic_r = rows['reg+score_prop']['ic']
    ic_k = rows['ranker+rank_lin']['ic']
    print(f"{str(sd.date()):^12}{bm*100:^+8.2f} | "
          f"{fmt('reg+score_prop')}{fmt('reg+rank_linear')} | "
          f"{fmt('ranker+rank_lin')}{fmt('ranker+score_p')} | "
          f"{ic_r:^+8.3f}{ic_k:^+8.3f}")

# ── summary ──────────────────────────────────────────────────────
print('\n' + '='*75)
print(f"{'指标':^20}", end='')
for k in dfs:
    print(f"{k:^16}", end='')
print()
print('-'*75)

for label, fn in [
    ('mean_excess%',  lambda d: d['excess'].mean()*100),
    ('std_excess%',   lambda d: d['excess'].std()*100),
    ('sharpe',        lambda d: d['excess'].mean()/d['excess'].std()),
    ('win_rate',      lambda d: (d['excess']>0).mean()),
    ('max_win%',      lambda d: d['excess'].max()*100),
    ('max_loss%',     lambda d: d['excess'].min()*100),
    ('mean_IC',       lambda d: d['ic'].mean()),
]:
    print(f"  {label:^18}", end='')
    for k in dfs:
        print(f"{fn(dfs[k]):^+16.4f}", end='')
    print()

# ── IC comparison ────────────────────────────────────────────────
ic_reg_mean = np.mean([r['ic'] for r in results['reg+score_prop']])
ic_rnk_mean = np.mean([r['ic'] for r in results['ranker+rank_lin']])
print(f"\n  Regression IC mean : {ic_reg_mean:+.4f}")
print(f"  Ranker     IC mean : {ic_rnk_mean:+.4f}")
print(f"  IC advantage       : {'ranker' if ic_rnk_mean > ic_reg_mean else 'regression'} "
      f"by {abs(ic_rnk_mean - ic_reg_mean):.4f}")

# ── pairwise win count ────────────────────────────────────────────
print(f"\n  逐日对决 (exp_019 reg+sp vs ranker+rl):")
dsp = dfs['reg+score_prop']['excess'].values
drl = dfs['ranker+rank_lin']['excess'].values
print(f"    reg+score_prop 胜 : {(dsp > drl).sum()}/{n} 天")
print(f"    ranker+rank_lin 胜: {(drl > dsp).sum()}/{n} 天")
print(f"    平均差 (sp - rnk) : {(dsp - drl).mean()*100:+.4f}%")
