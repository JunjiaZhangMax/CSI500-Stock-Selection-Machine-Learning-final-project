"""Walk-forward validation: 3d-target vs 5d-target, evaluated on 5-day holding windows.

Matches the actual competition Window 2 setup:
  - buy at close(t), hold 5 trading days, sell at close(t+5)
  - monthly retrain (expanding window)
  - both models use: top_k=30, hl=60, cap=10%, score-prop weighting
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame

DATA_DIR = Path('data')
TOP_K, CAP, HL = 30, 0.10, 60

print("Loading data...")
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)
idx_close = index_df.set_index('date')['close']

# Add target_5d
print("Computing target_5d...")
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
t5 = (close_piv.shift(-5) / close_piv - 1.0).stack().reset_index()
t5.columns = ['date', 'stock_code', 'target_5d']
panel = panel.merge(t5, on=['date', 'stock_code'], how='left')

TARGET_3D = TARGET_COLUMN   # 'target_3d'
TARGET_5D = 'target_5d'

panel_close = close_piv
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}

# ── Helpers ───────────────────────────────────────────────────────
def make_decay(df, hl=60, floor=0.5):
    ds    = np.sort(df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


def score_prop_w(scores_s):
    top = scores_s.nlargest(TOP_K)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()
    for _ in range(200):
        mask = w > CAP
        if not mask.any():
            break
        excess   = (w[mask] - CAP).sum()
        w[mask]  = CAP
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)


def eval_5d(weights, buy_date, sell_date):
    bp  = panel_close.loc[buy_date].reindex(weights.index)
    sp_ = panel_close.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    if valid.sum() == 0:
        return np.nan
    w_v = weights[valid] / weights[valid].sum()
    return float((w_v * (sp_[valid] / bp[valid] - 1)).sum())


# ── Walk-forward: monthly retrain ─────────────────────────────────
start_eval   = pd.Timestamp('2025-10-01')
end_eval     = pd.Timestamp('2026-05-08')
eval_dates   = [pd.Timestamp(d) for d in all_trading
                if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training monthly models (3d-target and 5d-target)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    # 3d model: cutoff = available[-4] (3d embargo)
    # 5d model: cutoff = available[-6] (5d embargo)
    if len(available) < 6:
        continue
    cutoff_3d = pd.Timestamp(available[-4])
    cutoff_5d = pd.Timestamp(available[-6])

    tr3 = panel[panel['date'] <= cutoff_3d].dropna(subset=FEATURE_COLUMNS + [TARGET_3D])
    tr5 = panel[panel['date'] <= cutoff_5d].dropna(subset=FEATURE_COLUMNS + [TARGET_5D])
    if len(tr3) < 5000 or len(tr5) < 5000:
        continue

    sw3 = make_decay(tr3, hl=HL)
    m3  = make_xgb()
    m3.fit(tr3[FEATURE_COLUMNS].values, tr3[TARGET_3D].values,
           sample_weight=sw3, verbose=False)

    sw5 = make_decay(tr5, hl=HL)
    m5  = make_xgb()
    m5.fit(tr5[FEATURE_COLUMNS].values, tr5[TARGET_5D].values,
           sample_weight=sw5, verbose=False)

    model_cache[ms] = (m3, m5, cutoff_3d, cutoff_5d)
    print(f"  {ms.strftime('%Y-%m')}: 3d-cutoff={cutoff_3d.date()}({len(tr3):,}rows) "
          f"5d-cutoff={cutoff_5d.date()}({len(tr5):,}rows)")


def get_models(date):
    cands = [ms for ms in model_cache if ms <= date]
    return model_cache[max(cands)] if cands else None


# ── Daily evaluation on 5-day holding windows ─────────────────────
# buy_date = t, sell_date = t+5 trading days
records = []
for i, d in enumerate(all_trading):
    buy_date = pd.Timestamp(d)
    if not (start_eval <= buy_date <= end_eval):
        continue
    sell_idx = i + 5
    if sell_idx >= len(all_trading):
        continue
    sell_date = pd.Timestamp(all_trading[sell_idx])

    # Use model trained before buy_date's month
    mdls = get_models(buy_date)
    if mdls is None:
        continue
    m3, m5, _, _ = mdls

    pred_df = panel[panel['date'] == buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        continue

    bench = float(idx_close.loc[sell_date] / idx_close.loc[buy_date] - 1)

    sc3 = pd.Series(m3.predict(pred_df[FEATURE_COLUMNS].values),
                    index=pred_df['stock_code'].values)
    sc5 = pd.Series(m5.predict(pred_df[FEATURE_COLUMNS].values),
                    index=pred_df['stock_code'].values)

    w3 = score_prop_w(sc3)
    w5 = score_prop_w(sc5)

    p3 = eval_5d(w3, buy_date, sell_date)
    p5 = eval_5d(w5, buy_date, sell_date)

    if np.isnan(p3) or np.isnan(p5):
        continue

    # IC: how well does prediction at buy_date rank 5d returns?
    tgt5_at_buy = pred_df.set_index('stock_code')[TARGET_5D].dropna()
    if len(tgt5_at_buy) > 10:
        ic3 = float(spearmanr(sc3.reindex(tgt5_at_buy.index).dropna(),
                               tgt5_at_buy.reindex(sc3.reindex(tgt5_at_buy.index).dropna().index))[0])
        ic5 = float(spearmanr(sc5.reindex(tgt5_at_buy.index).dropna(),
                               tgt5_at_buy.reindex(sc5.reindex(tgt5_at_buy.index).dropna().index))[0])
    else:
        ic3 = ic5 = np.nan

    records.append({
        'buy':      buy_date,
        'sell':     sell_date,
        'month':    buy_date.strftime('%Y-%m'),
        'bench':    bench,
        'port_3d':  p3, 'excess_3d': p3 - bench,
        'port_5d':  p5, 'excess_5d': p5 - bench,
        'ic3':      ic3, 'ic5': ic5,
    })

df = pd.DataFrame(records)
print(f"\nEvaluated {len(df)} 5-day windows\n")

# ── Report ────────────────────────────────────────────────────────
SEP = '=' * 75
print(SEP)
print("  5-Day Hold Walk-Forward  Oct 2025 – May 2026  (monthly retrain)")
print(SEP)
print(f"{'month':^10}{'N':^5}{'3d-tgt_exc%':^15}{'5d-tgt_exc%':^15}"
      f"{'delta':^10}{'bench%':^10}")
print('-' * 55)
for mo in sorted(df['month'].unique()):
    sub = df[df['month'] == mo]
    e3  = sub['excess_3d'].mean() * 100
    e5  = sub['excess_5d'].mean() * 100
    bch = sub['bench'].mean()     * 100
    print(f"{mo:^10}{len(sub):^5}{e3:^+15.3f}{e5:^+15.3f}{e5-e3:^+10.3f}{bch:^+10.3f}")
print('-' * 55)

print(f"\n{'metric':^16}{'3d-target':^18}{'5d-target':^18}{'winner':^10}")
print('-' * 62)
for label, c3, c5, higher_is_better in [
    ('mean_excess%',  df['excess_3d'].mean()*100,       df['excess_5d'].mean()*100,       True),
    ('std_excess%',   df['excess_3d'].std()*100,        df['excess_5d'].std()*100,         False),
    ('sharpe',        df['excess_3d'].mean()/df['excess_3d'].std(),
                      df['excess_5d'].mean()/df['excess_5d'].std(),                        True),
    ('win_rate',      (df['excess_3d']>0).mean(),       (df['excess_5d']>0).mean(),        True),
    ('port_mean%',    df['port_3d'].mean()*100,         df['port_5d'].mean()*100,          True),
    ('max_loss%',     df['excess_3d'].min()*100,        df['excess_5d'].min()*100,         False),
    ('mean_IC(5d)',   df['ic3'].mean(),                 df['ic5'].mean(),                  True),
]:
    better = (c5 > c3) if higher_is_better else (c5 < c3)
    winner = '5d ***' if better else '3d'
    print(f"  {label:^14}  {c3:^+18.4f}{c5:^+18.4f}{winner:^10}")

# ── Regime breakdown ──────────────────────────────────────────────
idx_log = np.log(idx_close)
def idx_ret20(d):
    if d not in idx_log.index: return np.nan
    loc = idx_log.index.get_loc(d)
    return float(idx_log.iloc[loc] - idx_log.iloc[max(0, loc-20)]) if loc >= 20 else np.nan

df['idx_ret_20d'] = df['buy'].map(idx_ret20).astype(float)

print(f"\n{SEP}")
print("  Performance by Market Regime (at buy date)")
print(SEP)
print(f"{'regime':^22}{'N':^5}{'3d-tgt_exc%':^15}{'5d-tgt_exc%':^15}{'delta':^10}{'bench%':^10}")
print('-' * 77)
for label, lo, hi in [
    ('strong bull (>+3%)',   0.03, np.inf),
    ('bull (+1% to +3%)',    0.01, 0.03),
    ('neutral (±1%)',       -0.01, 0.01),
    ('bear (-1% to -3%)',   -0.03, -0.01),
    ('strong bear (<-3%)',  -np.inf, -0.03),
]:
    r = df['idx_ret_20d'].astype(float)
    mask = (r > lo) & (r <= hi) if lo != -np.inf and hi != np.inf \
           else (r > lo) if hi == np.inf else (r <= hi)
    sub = df[mask]
    if len(sub) < 3:
        continue
    e3  = sub['excess_3d'].mean() * 100
    e5  = sub['excess_5d'].mean() * 100
    bch = sub['bench'].mean()     * 100
    print(f"  {label:^20}{len(sub):^5}{e3:^+15.3f}{e5:^+15.3f}{e5-e3:^+10.3f}{bch:^+10.3f}")

# ── Rolling 3-month sharpe ────────────────────────────────────────
print(f"\n{SEP}")
print("  Rolling Stability: 3-month excess return (non-overlapping)")
print(SEP)
df_sorted = df.sort_values('buy')
df_sorted['qtr'] = df_sorted['buy'].dt.to_period('Q')
print(f"{'quarter':^12}{'N':^5}{'3d-tgt_exc%':^15}{'5d-tgt_exc%':^15}{'bench%':^10}")
print('-' * 57)
for qtr in sorted(df_sorted['qtr'].unique()):
    sub = df_sorted[df_sorted['qtr'] == qtr]
    if len(sub) < 3: continue
    e3  = sub['excess_3d'].mean() * 100
    e5  = sub['excess_5d'].mean() * 100
    bch = sub['bench'].mean()     * 100
    print(f"  {str(qtr):^10}{len(sub):^5}{e3:^+15.3f}{e5:^+15.3f}{bch:^+10.3f}")
