"""Market Regime-Aware Ensemble v2 -- finer regime + smoother sample weights.

Changes vs v1:
  - 3 regime specialists: strong / neutral / weak
  - Smoother training weights: affinity-based, range [0.5, 1.0]
      strong:  w = decay x (0.5 + 0.5 * r)
      neutral: w = decay x (0.5 + 0.5 * (1 - |2r-1|))   (peaks at r=0.5)
      weak:    w = decay x (0.5 + 0.5 * (1 - r))
  - All specialists always see ALL training data; affinity just re-weights them.
  - Soft blend uses piecewise-linear (triangular) weights that always sum to 1:
      r in [0.0, 0.5]: w_weak=1-2r,  w_neutral=2r,    w_strong=0
      r in [0.5, 1.0]: w_weak=0,     w_neutral=2(1-r), w_strong=2r-1
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from features import build_features, FEATURE_COLUMNS, TARGET_COLUMN, training_frame

DATA_DIR     = Path('data')
TRAIN_CUTOFF = '2026-04-08'
TOP_K        = 50
CAP          = 0.08

prices   = pd.read_parquet(DATA_DIR / 'prices.parquet')
index_df = pd.read_parquet(DATA_DIR / 'index.parquet').sort_values('date').reset_index(drop=True)
index_df['date'] = pd.to_datetime(index_df['date'])
panel       = build_features(prices)
panel_close = panel.pivot_table(index='date', columns='stock_code', values='close')
idx_close   = index_df.set_index('date')['close']
all_trading = sorted(panel['date'].unique())
date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(all_trading)}


# ── Regime detector (same formula as v1) ────────────────────────────
def compute_regime_series(idx_close, lookback=20):
    """r(t) in [0,1]: sigmoid(50*ret_20d - 80*vol_20d - 0.5)."""
    log_close = np.log(idx_close.values)
    n = len(log_close)
    regimes = np.full(n, 0.5)
    for i in range(lookback, n):
        ret_n = np.exp(log_close[i] - log_close[i - lookback]) - 1
        d_rets = np.diff(log_close[i - lookback:i + 1])
        vol_n  = float(np.std(d_rets))
        signal = 50.0 * ret_n - 80.0 * vol_n - 0.5
        regimes[i] = 1.0 / (1.0 + np.exp(-signal))
    return pd.Series(regimes, index=idx_close.index)


regime_series = compute_regime_series(idx_close)

# ── Helpers ─────────────────────────────────────────────────────────
def make_decay(df, hl=120, floor=0.5):
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


def make_xgb(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=seed)


def soft_weights(r):
    """Triangular blending weights for 3 specialists. Sum to 1."""
    r = float(r)
    if r <= 0.5:
        w_weak   = 1.0 - 2.0 * r
        w_neutral = 2.0 * r
        w_strong  = 0.0
    else:
        w_weak    = 0.0
        w_neutral = 2.0 * (1.0 - r)
        w_strong  = 2.0 * r - 1.0
    return w_strong, w_neutral, w_weak


def eval_window(scores, sd, bd):
    weights = score_prop_w(scores)
    bp  = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    w_v   = weights[valid] / weights[valid].sum()
    port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    return port, bench


def train_trio(tr):
    """Train (baseline, strong, neutral, weak) models on training frame tr."""
    tr = tr.copy()
    tr['regime'] = tr['date'].map(regime_series).fillna(0.5)
    sw_d = make_decay(tr)
    r    = tr['regime'].values

    # smoother affinity weights
    w_strong  = 0.5 + 0.5 * r
    w_neutral = 0.5 + 0.5 * (1.0 - np.abs(2.0 * r - 1.0))
    w_weak    = 0.5 + 0.5 * (1.0 - r)

    b_m = make_xgb()
    b_m.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
            sample_weight=sw_d, verbose=False)

    s_m = make_xgb()
    s_m.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
            sample_weight=sw_d * w_strong, verbose=False)

    n_m = make_xgb()
    n_m.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
            sample_weight=sw_d * w_neutral, verbose=False)

    w_m = make_xgb()
    w_m.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
            sample_weight=sw_d * w_weak, verbose=False)

    return b_m, s_m, n_m, w_m


def predict_soft(pred_df, r, s_m, n_m, w_m):
    ws, wn, ww = soft_weights(r)
    s_s = pd.Series(s_m.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    s_n = pd.Series(n_m.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    s_w = pd.Series(w_m.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    return ws * s_s + wn * s_n + ww * s_w


# ── training data ──────────────────────────────────────────────────
train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
print("=== Regime label distribution (training period) ===")
train_df = train_df.copy()
train_df['regime'] = train_df['date'].map(regime_series)
print(f"  train_df: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
print(f"  regime mean={train_df['regime'].mean():.3f}  std={train_df['regime'].std():.3f}")
print(f"  strong (r>0.6): {(train_df['regime']>0.6).mean()*100:.1f}%")
print(f"  neutral (0.3-0.6): {((train_df['regime']>=0.3)&(train_df['regime']<=0.6)).mean()*100:.1f}%")
print(f"  weak  (r<0.3): {(train_df['regime']<0.3).mean()*100:.1f}%\n")

print(f"Train rows: {len(train_df):,}")
print("Training baseline + 3 specialists (strong/neutral/weak)...")
base_m, strong_m, neutral_m, weak_m = train_trio(train_df)
print("Done.\n")


# ── April daily backtest ─────────────────────────────────────────
april_pairs = []
for d in all_trading:
    sd = pd.Timestamp(d)
    if sd.year != 2026 or sd.month != 4:
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    april_pairs.append((sd, pd.Timestamp(all_trading[si - 3])))

buy_frames = {bd: panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
              for _, bd in april_pairs}

print("=" * 100)
print("1)  April Daily Backtest -- exp_021 baseline vs 3-Specialist Soft Ensemble")
print("=" * 100)
print(f"{'sell':^12}{'buy':^12}{'r':^7}{'regime':^9}{'ws wn ww':^19}"
      f"{'baseline%':^11}{'soft3%':^11}{'bench%':^9}")
print('-' * 95)

rows_base, rows_soft = [], []
for sd, bd in april_pairs:
    pred_df = buy_frames[bd]
    if len(pred_df) < TOP_K:
        continue
    r  = float(regime_series.get(bd, 0.5))
    ws, wn, ww = soft_weights(r)
    regime_label = ('strong' if r > 0.6 else ('neutral' if r >= 0.3 else 'weak'))

    s_base = pd.Series(base_m.predict(pred_df[FEATURE_COLUMNS]),
                       index=pred_df['stock_code'].values)
    s_soft = predict_soft(pred_df, r, strong_m, neutral_m, weak_m)

    portB, bench = eval_window(s_base, sd, bd)
    portS, _     = eval_window(s_soft, sd, bd)

    rows_base.append({'sell': sd, 'r': r, 'port': portB, 'bench': bench, 'excess': portB - bench})
    rows_soft.append({'sell': sd, 'r': r, 'port': portS, 'bench': bench, 'excess': portS - bench})

    print(f"{str(sd.date()):^12}{str(bd.date()):^12}{r:^7.3f}{regime_label:^9}"
          f"{ws:.2f}/{wn:.2f}/{ww:.2f}".center(19)
          + f"{(portB-bench)*100:^+11.3f}{(portS-bench)*100:^+11.3f}{bench*100:^+9.2f}")

df_base = pd.DataFrame(rows_base)
df_soft = pd.DataFrame(rows_soft)

print('\n' + '=' * 55)
print(f"{'Metric':^18}{'baseline':^18}{'soft3':^18}")
print('-' * 55)
for label, fn in [
    ('mean_excess%',  lambda d: d['excess'].mean()*100),
    ('std_excess%',   lambda d: d['excess'].std()*100),
    ('sharpe',        lambda d: d['excess'].mean()/d['excess'].std()),
    ('win_rate',      lambda d: (d['excess']>0).mean()),
    ('max_loss%',     lambda d: d['excess'].min()*100),
]:
    print(f"  {label:^16}{fn(df_base):^+18.4f}{fn(df_soft):^+18.4f}")

# by regime group
print('\n' + '=' * 75)
print("Performance by Regime Group (April only):")
print(f"{'group':^22}{'N':^5}{'baseline%':^14}{'soft3%':^14}")
print('-' * 55)
for label, mask_fn in [
    ('strong (r>0.6)',  lambda d: d['r'] > 0.6),
    ('neutral (0.3-0.6)', lambda d: (d['r'] >= 0.3) & (d['r'] <= 0.6)),
    ('weak (r<0.3)',    lambda d: d['r'] < 0.3),
]:
    mb = mask_fn(df_base); ms = mask_fn(df_soft)
    n  = mb.sum()
    if n == 0:
        print(f"  {label:^20}{n:^5}  (no data)")
        continue
    eb = df_base.loc[mb.values, 'excess'].mean()*100
    es = df_soft.loc[ms.values, 'excess'].mean()*100
    print(f"  {label:^20}{n:^5}{eb:^+14.3f}{es:^+14.3f}")


# ── Walk-forward Oct 2025 -> Apr 2026 ────────────────────────────
print('\n' + '=' * 90)
print("2)  Walk-Forward Backtest  Oct 2025 - Apr 2026  (monthly retrain)")
print("=" * 90)

start_eval = pd.Timestamp('2025-10-01')
end_eval   = pd.Timestamp('2026-04-30')
eval_dates = [pd.Timestamp(d) for d in all_trading
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training monthly walk-forward models (base / strong / neutral / weak)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr) < 5000:
        continue
    b_m, s_m, n_m, w_m = train_trio(tr)
    model_cache[ms] = (b_m, s_m, n_m, w_m, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr):,}")


def get_models(sell_date):
    cands = [ms for ms in model_cache if ms <= sell_date]
    if not cands:
        return None
    return model_cache[max(cands)]


wf = {'base': [], 'soft3': []}
for d in all_trading:
    sd = pd.Timestamp(d)
    if not (start_eval <= sd <= end_eval):
        continue
    si = date_to_idx[sd]
    if si < 3:
        continue
    bd = pd.Timestamp(all_trading[si - 3])
    mdls = get_models(sd)
    if mdls is None:
        continue
    b_m, s_m, n_m, w_m, _ = mdls
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        continue
    r = float(regime_series.get(bd, 0.5))

    s_b    = pd.Series(b_m.predict(pred_df[FEATURE_COLUMNS]),
                       index=pred_df['stock_code'].values)
    s_soft = predict_soft(pred_df, r, s_m, n_m, w_m)

    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    for label, scores in [('base', s_b), ('soft3', s_soft)]:
        port, _ = eval_window(scores, sd, bd)
        wf[label].append({'sell': sd, 'month': sd.strftime('%Y-%m'),
                          'r': r, 'port': port, 'bench': bench,
                          'excess': port - bench})

wfs = {k: pd.DataFrame(v) for k, v in wf.items()}

print(f"\n{'month':^10}{'N':^5}{'base_excess%':^16}{'soft3_excess%':^16}{'bench%':^10}{'mean_r':^8}")
print('-' * 65)
for mo in sorted(wfs['base']['month'].unique()):
    sb = wfs['base'] [wfs['base'] ['month'] == mo]
    ss = wfs['soft3'][wfs['soft3']['month'] == mo]
    print(f"{mo:^10}{len(sb):^5}"
          f"{sb['excess'].mean()*100:^+16.3f}"
          f"{ss['excess'].mean()*100:^+16.3f}"
          f"{sb['bench'].mean()*100:^+10.3f}"
          f"{sb['r'].mean():^8.3f}")

print('-' * 65)
for label, fn in [('mean_excess%', lambda d: d['excess'].mean()*100),
                   ('std_excess%',  lambda d: d['excess'].std()*100),
                   ('sharpe',       lambda d: d['excess'].mean()/d['excess'].std()),
                   ('win_rate',     lambda d: (d['excess']>0).mean())]:
    print(f"  {label:^12}", end='')
    for c in ['base', 'soft3']:
        print(f"    {fn(wfs[c]):^+14.4f}", end='')
    print()

print('\nRegime-conditional performance (walk-forward full period):')
for label, mask_fn in [
    ('strong (r>0.6)',     lambda d: d['r'] > 0.6),
    ('neutral (0.3-0.6)',  lambda d: (d['r'] >= 0.3) & (d['r'] <= 0.6)),
    ('weak   (r<0.3)',     lambda d: d['r'] < 0.3),
]:
    print(f"  {label:^22}", end='')
    for c in ['base', 'soft3']:
        sub = wfs[c][mask_fn(wfs[c])]
        e   = sub['excess'].mean()*100 if len(sub) > 0 else 0
        print(f"  {c}={e:^+7.3f}% (N={len(sub)})", end='')
    print()
