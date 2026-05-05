"""Long-term walk-forward stability analysis: score_prop vs rank_linear."""
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


def train_model(tr_df):
    dates_sorted = np.sort(tr_df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(dates_sorted)}
    n     = len(dates_sorted)
    delta = (n - 1) - tr_df['date'].map(d2i).values
    sw    = np.maximum(np.exp(-np.log(2) * delta / 120), 0.5)
    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)
    m.fit(tr_df[FEATURE_COLUMNS], tr_df[TARGET_COLUMN], sample_weight=sw, verbose=False)
    return m


def score_prop_weights(scores_series):
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


def rank_linear_weights(scores_series):
    return build_portfolio(scores_series, top_k=TOP_K)


def eval_window(model, sell_date, buy_date, wt_fn):
    pred_df = panel[panel['date'] == buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        return None
    scores  = pd.Series(model.predict(pred_df[FEATURE_COLUMNS]),
                        index=pred_df['stock_code'].values)
    weights = wt_fn(scores)
    buy_p   = panel_close.loc[buy_date].reindex(weights.index)
    sell_p  = panel_close.loc[sell_date].reindex(weights.index)
    valid   = (~buy_p.isna()) & (~sell_p.isna())
    if valid.sum() < 20:
        return None
    w_v  = weights[valid] / weights[valid].sum()
    port = float((w_v * (sell_p[valid] / buy_p[valid] - 1)).sum())
    bench = float(idx_close.loc[sell_date] / idx_close.loc[buy_date] - 1)
    ic = spearmanr(
        scores.reindex(buy_p[valid].index).fillna(0),
        (sell_p[valid] / buy_p[valid] - 1)
    )[0]
    return {'port': port, 'bench': bench, 'excess': port - bench, 'ic': ic}


# --- walk-forward: retrain monthly ---
start_eval = pd.Timestamp('2025-10-01')
end_eval   = pd.Timestamp('2026-04-30')
eval_dates = [pd.Timestamp(d) for d in all_trading
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training walk-forward models (monthly refit)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr_df  = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr_df) < 5000:
        continue
    model_cache[ms] = (train_model(tr_df), cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr_df):,}")


def get_model(sell_date):
    candidates = [ms for ms in model_cache if ms <= sell_date]
    if not candidates:
        return None, None
    return model_cache[max(candidates)]


# --- daily backtest ---
rows_sp, rows_rl = [], []
for d in all_trading:
    sd = pd.Timestamp(d)
    if sd < start_eval or sd > end_eval:
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd = pd.Timestamp(all_trading[si - 3])
    m, _ = get_model(sd)
    if m is None:
        continue
    r_sp = eval_window(m, sd, bd, score_prop_weights)
    r_rl = eval_window(m, sd, bd, rank_linear_weights)
    if r_sp:
        rows_sp.append({'date': sd, 'month': sd.strftime('%Y-%m'), **r_sp})
    if r_rl:
        rows_rl.append({'date': sd, 'month': sd.strftime('%Y-%m'), **r_rl})

df_sp = pd.DataFrame(rows_sp)
df_rl = pd.DataFrame(rows_rl)

print(f"\n{'='*72}")
print(f"Walk-forward backtest  {start_eval.date()} ~ {end_eval.date()}  N={len(df_sp)} windows")
print(f"{'='*72}")

# monthly breakdown
print(f"\n{'月份':^10} {'sp超额%':^12} {'rl超额%':^12} {'sp胜率':^8} {'rl胜率':^8} {'IC':^8} {'N':^5}")
print('-'*68)
for mo in sorted(df_sp['month'].unique()):
    sp = df_sp[df_sp['month'] == mo]
    rl = df_rl[df_rl['month'] == mo]
    ic = sp['ic'].mean()
    print(f"  {mo}  {sp['excess'].mean()*100:^+12.3f}{rl['excess'].mean()*100:^+12.3f}"
          f"{(sp['excess']>0).mean():^8.2f}{(rl['excess']>0).mean():^8.2f}"
          f"{ic:^+8.3f}{len(sp):^5}")

# overall
print(f"\n{'='*55}")
print(f"{'指标':^20}  {'score_prop':^14}  {'rank_linear':^14}")
print(f"{'-'*55}")
metrics = [
    ('mean_excess%', lambda x: x['excess'].mean()*100),
    ('std_excess%',  lambda x: x['excess'].std()*100),
    ('sharpe',       lambda x: x['excess'].mean()/x['excess'].std() if x['excess'].std() > 0 else 0),
    ('win_rate',     lambda x: (x['excess'] > 0).mean()),
    ('max_loss%',    lambda x: x['excess'].min()*100),
    ('mean_IC',      lambda x: x['ic'].mean()),
]
for label, fn in metrics:
    print(f"  {label:^20}{fn(df_sp):^+14.4f}{fn(df_rl):^+14.4f}")


def max_losing_streak(series):
    cur = mx = 0
    for v in series:
        cur = cur + 1 if v < 0 else 0
        mx  = max(mx, cur)
    return mx

print(f"\n  最长连续负超额: score_prop={max_losing_streak(df_sp['excess'])}天"
      f"  rank_linear={max_losing_streak(df_rl['excess'])}天")

# bull vs bear
for name, df in [('score_prop', df_sp), ('rank_linear', df_rl)]:
    bull = df[df['bench'] > 0]
    bear = df[df['bench'] <= 0]
    print(f"\n  {name} 按市场方向:")
    print(f"    上涨日(N={len(bull)}): mean_excess={bull['excess'].mean()*100:+.3f}%  "
          f"win={(bull['excess']>0).mean():.2f}")
    print(f"    下跌日(N={len(bear)}): mean_excess={bear['excess'].mean()*100:+.3f}%  "
          f"win={(bear['excess']>0).mean():.2f}")

# cumulative excess return curve (text)
print(f"\n  累计超额收益走势 (score_prop vs rank_linear):")
print(f"  {'月份':^10} {'sp累计%':^12} {'rl累计%':^12} {'基准累计%':^12}")
cum_sp = cum_rl = cum_bm = 0.0
for mo in sorted(df_sp['month'].unique()):
    sp = df_sp[df_sp['month'] == mo]
    rl = df_rl[df_rl['month'] == mo]
    cum_sp += sp['excess'].sum() * 100
    cum_rl += rl['excess'].sum() * 100
    cum_bm += sp['bench'].mean() * 100 * len(sp)
    print(f"  {mo:^10}{cum_sp:^+12.2f}{cum_rl:^+12.2f}{cum_bm:^+12.2f}")
