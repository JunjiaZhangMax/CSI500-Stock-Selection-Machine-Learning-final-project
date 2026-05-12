"""Walk-forward comparison: score-prop vs vol-adjusted weights.

Vol-adjusted: after picking top_k by score, multiply each weight by 1/vol_20d
before cap enforcement. Naturally down-weights high-volatility stocks.

Same model in both: 5d-target, XGBoost, hl=60, top_k=30, cap=10%.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import spearmanr
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN

DATA_DIR = Path('data')
TOP_K, CAP, HL = 30, 0.10, 60

print("Loading data...")
prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel    = build_features(prices)
idx_close = index_df.set_index('date')['close']

print("Computing target_5d...")
close_piv = panel.pivot_table(index='date', columns='stock_code', values='close')
t5 = (close_piv.shift(-5) / close_piv - 1.0).stack().reset_index()
t5.columns = ['date', 'stock_code', 'target_5d']
panel = panel.merge(t5, on=['date', 'stock_code'], how='left')

all_trading = sorted(panel['date'].unique())


# ── Helpers ───────────────────────────────────────────────────────
def make_decay(df, hl=60, floor=0.5):
    ds  = np.sort(df['date'].unique())
    d2i = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)


def score_prop_w(scores_s, vol_s=None, top_k=TOP_K, cap=CAP, vol_adj=False):
    top = scores_s.nlargest(top_k)
    w   = top.values - top.values.min() + 1e-6
    w  /= w.sum()

    if vol_adj and vol_s is not None:
        vols = vol_s.reindex(top.index).values.astype(float)
        med  = float(np.nanmedian(vols[vols > 0]))
        vols = np.where(np.isnan(vols) | (vols <= 0), med, vols)
        w    = w / vols          # lower vol → higher weight
        w   /= w.sum()

    for _ in range(500):
        mask = w > cap
        if not mask.any():
            break
        excess   = (w[mask] - cap).sum()
        w[mask]  = cap
        w[~mask] += excess * (w[~mask] / w[~mask].sum())
    return pd.Series(w / w.sum(), index=top.index)


def eval_5d(weights, buy_date, sell_date):
    bp  = close_piv.loc[buy_date].reindex(weights.index)
    sp_ = close_piv.loc[sell_date].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    if valid.sum() == 0:
        return np.nan
    w_v = weights[valid] / weights[valid].sum()
    return float((w_v * (sp_[valid] / bp[valid] - 1)).sum())


# ── Walk-forward monthly retrain ──────────────────────────────────
start_eval   = pd.Timestamp('2025-10-01')
end_eval     = pd.Timestamp('2026-05-08')
eval_dates   = [pd.Timestamp(d) for d in all_trading
                if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training monthly models (5d-target)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 86:
        continue
    cutoff = pd.Timestamp(available[-6])
    tr = panel[panel['date'] <= cutoff].dropna(subset=FEATURE_COLUMNS + ['target_5d'])
    if len(tr) < 5000:
        continue
    sw = make_decay(tr, hl=HL)
    m  = make_xgb()
    m.fit(tr[FEATURE_COLUMNS].values, tr['target_5d'].values,
          sample_weight=sw, verbose=False)
    model_cache[ms] = (m, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr):,}")


def get_model(date):
    cands = [ms for ms in model_cache if ms <= date]
    return model_cache[max(cands)] if cands else None


# ── Daily evaluation ──────────────────────────────────────────────
records = []
for i, d in enumerate(all_trading):
    buy_date = pd.Timestamp(d)
    if not (start_eval <= buy_date <= end_eval):
        continue
    sell_idx = i + 5
    if sell_idx >= len(all_trading):
        continue
    sell_date = pd.Timestamp(all_trading[sell_idx])

    mdl = get_model(buy_date)
    if mdl is None:
        continue
    m, _ = mdl

    pred_df = panel[panel['date'] == buy_date].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        continue

    bench = float(idx_close.loc[sell_date] / idx_close.loc[buy_date] - 1)

    scores  = pd.Series(m.predict(pred_df[FEATURE_COLUMNS].values),
                        index=pred_df['stock_code'].values)
    vol_s   = pred_df.set_index('stock_code')['vol_20d']

    w_base  = score_prop_w(scores, vol_adj=False)
    w_vadj  = score_prop_w(scores, vol_s=vol_s, vol_adj=True)

    p_base  = eval_5d(w_base, buy_date, sell_date)
    p_vadj  = eval_5d(w_vadj, buy_date, sell_date)

    if np.isnan(p_base) or np.isnan(p_vadj):
        continue

    # avg vol of selected stocks
    avg_vol_base = float(vol_s.reindex(w_base.index).mean())
    avg_vol_vadj = float(vol_s.reindex(w_vadj.index).mean())

    records.append({
        'buy':        buy_date,
        'sell':       sell_date,
        'month':      buy_date.strftime('%Y-%m'),
        'bench':      bench,
        'port_base':  p_base, 'exc_base':  p_base  - bench,
        'port_vadj':  p_vadj, 'exc_vadj':  p_vadj  - bench,
        'avg_vol_base': avg_vol_base,
        'avg_vol_vadj': avg_vol_vadj,
    })

df = pd.DataFrame(records)
print(f"\nEvaluated {len(df)} 5-day windows\n")

# ── Report ────────────────────────────────────────────────────────
SEP = '=' * 72
print(SEP)
print("  5-Day Hold Walk-Forward  Oct 2025 – May 2026")
print("  score-prop  vs  vol-adjusted weights  (same model, top_k=30)")
print(SEP)
print(f"{'month':^10}{'N':^5}{'base_exc%':^13}{'vadj_exc%':^13}{'delta':^10}{'bench%':^10}")
print('-' * 51)
for mo in sorted(df['month'].unique()):
    sub = df[df['month'] == mo]
    eb  = sub['exc_base'].mean() * 100
    ev  = sub['exc_vadj'].mean() * 100
    bch = sub['bench'].mean()    * 100
    print(f"{mo:^10}{len(sub):^5}{eb:^+13.3f}{ev:^+13.3f}{ev-eb:^+10.3f}{bch:^+10.3f}")
print('-' * 51)

print(f"\n{'metric':^18}{'score-prop':^18}{'vol-adj':^18}{'winner':^10}")
print('-' * 64)
for label, c_b, c_v, higher_is_better in [
    ('mean_excess%',  df['exc_base'].mean()*100,        df['exc_vadj'].mean()*100,        True),
    ('std_excess%',   df['exc_base'].std()*100,         df['exc_vadj'].std()*100,         False),
    ('sharpe',        df['exc_base'].mean()/df['exc_base'].std(),
                      df['exc_vadj'].mean()/df['exc_vadj'].std(),                         True),
    ('win_rate',      (df['exc_base']>0).mean(),        (df['exc_vadj']>0).mean(),        True),
    ('port_mean%',    df['port_base'].mean()*100,       df['port_vadj'].mean()*100,       True),
    ('max_loss%',     df['exc_base'].min()*100,         df['exc_vadj'].min()*100,         False),
    ('avg_vol_base',  df['avg_vol_base'].mean()*100,    df['avg_vol_vadj'].mean()*100,    False),
]:
    better = (c_v > c_b) if higher_is_better else (c_v < c_b)
    winner = 'vadj ***' if better else 'base'
    print(f"  {label:^16}  {c_b:^+18.4f}{c_v:^+18.4f}{winner:^10}")

# ── Regime breakdown ──────────────────────────────────────────────
idx_log = np.log(idx_close)
def idx_ret20(d):
    if d not in idx_log.index: return np.nan
    loc = idx_log.index.get_loc(d)
    return float(idx_log.iloc[loc] - idx_log.iloc[max(0, loc-20)]) if loc >= 20 else np.nan

df['idx_ret_20d'] = df['buy'].map(idx_ret20).astype(float)

print(f"\n{SEP}")
print("  Performance by Market Regime")
print(SEP)
print(f"{'regime':^22}{'N':^5}{'base_exc%':^13}{'vadj_exc%':^13}{'delta':^10}{'bench%':^10}")
print('-' * 73)
for label, lo, hi in [
    ('strong bull (>+3%)',  0.03,  np.inf),
    ('bull (+1% to +3%)',   0.01,  0.03),
    ('neutral (±1%)',      -0.01,  0.01),
    ('bear (-1% to -3%)',  -0.03, -0.01),
    ('strong bear (<-3%)', -np.inf,-0.03),
]:
    r = df['idx_ret_20d'].astype(float)
    if lo == -np.inf:    mask = r <= hi
    elif hi == np.inf:   mask = r > lo
    else:                mask = (r > lo) & (r <= hi)
    sub = df[mask]
    if len(sub) < 3: continue
    eb  = sub['exc_base'].mean() * 100
    ev  = sub['exc_vadj'].mean() * 100
    bch = sub['bench'].mean()    * 100
    print(f"  {label:^20}{len(sub):^5}{eb:^+13.3f}{ev:^+13.3f}{ev-eb:^+10.3f}{bch:^+10.3f}")

# ── Vol environment split ─────────────────────────────────────────
idx_vol20 = idx_log.diff().rolling(20).std() * np.sqrt(252)
df['mkt_vol'] = df['buy'].map(lambda d: float(idx_vol20.get(d, np.nan)))

print(f"\n{SEP}")
print("  Performance by Market Volatility (CSI500 realized 20d vol)")
print(SEP)
med_vol = df['mkt_vol'].median()
print(f"  Median market vol: {med_vol*100:.1f}%\n")
print(f"{'vol regime':^22}{'N':^5}{'base_exc%':^13}{'vadj_exc%':^13}{'delta':^10}")
print('-' * 63)
for label, mask in [
    (f'low vol  (<{med_vol*100:.0f}%)',  df['mkt_vol'] <= med_vol),
    (f'high vol (≥{med_vol*100:.0f}%)',  df['mkt_vol'] >  med_vol),
]:
    sub = df[mask]
    if len(sub) < 3: continue
    eb = sub['exc_base'].mean() * 100
    ev = sub['exc_vadj'].mean() * 100
    print(f"  {label:^20}{len(sub):^5}{eb:^+13.3f}{ev:^+13.3f}{ev-eb:^+10.3f}")
