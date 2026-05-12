"""Market Regime-Aware Ensemble (weak supervision).

Pipeline:
  1. Compute regime label r(t) in [0,1] from past-20d CSI500 momentum + vol.
     r=1 => strong uptrend; r=0 => sideways/down. (weak supervision: rule-based)
  2. Train two specialist XGBoost models on the same features/target:
     - Bull model:  sample_weight = decay × r(t)
     - Side model:  sample_weight = decay × (1 - r(t))
  3. At prediction time, combine outputs using current regime:
     - Soft mix  : pred = r * bull + (1-r) * side
     - Hard switch: pred = bull if r >= 0.5 else side
  4. Compare against exp_021 baseline on April daily backtest, plus
     walk-forward Oct 2025 -> Apr 2026 to see regime-conditional performance.
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


# ── Regime detector (weak supervision) ─────────────────────────────
def compute_regime_series(idx_close, lookback=20):
    """Return regime score r(t) in [0,1] for each trading day.

    r = sigmoid(a * ret_20d - b * vol_20d - c)
    Tuned so that:
      strong uptrend (ret_20d=+5%, vol_20d=1.0%) -> r ≈ 0.92
      flat (ret_20d=0%, vol_20d=1.5%) -> r ≈ 0.38
      decline (ret_20d=-3%, vol_20d=2.0%) -> r ≈ 0.08
    """
    log_close = np.log(idx_close.values)
    n = len(log_close)
    regimes = np.full(n, 0.5)  # default neutral
    for i in range(lookback, n):
        ret_n = (np.exp(log_close[i] - log_close[i - lookback]) - 1)  # 20d return
        rets  = np.diff(np.exp(log_close[i - lookback:i + 1]) /
                        np.exp(log_close[i - lookback]))
        # daily returns
        d_rets = np.diff(np.exp(log_close[i - lookback:i + 1]))
        d_rets = np.diff(log_close[i - lookback:i + 1])  # log returns
        vol_n  = float(np.std(d_rets))
        signal = 50.0 * ret_n - 80.0 * vol_n - 0.5
        regimes[i] = 1.0 / (1.0 + np.exp(-signal))
    return pd.Series(regimes, index=idx_close.index)


regime_series = compute_regime_series(idx_close)
print("=== Regime label distribution (training period) ===")
train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
train_df['regime'] = train_df['date'].map(regime_series)
print(f"  train_df dates: {train_df['date'].min().date()} ~ {train_df['date'].max().date()}")
print(f"  regime mean: {train_df['regime'].mean():.3f}")
print(f"  regime std : {train_df['regime'].std():.3f}")
print(f"  bull-heavy days (r>0.7): {(train_df['regime']>0.7).mean()*100:.1f}%")
print(f"  side-heavy days (r<0.3): {(train_df['regime']<0.3).mean()*100:.1f}%")
print(f"  mid days     (0.3-0.7): {((train_df['regime']>=0.3)&(train_df['regime']<=0.7)).mean()*100:.1f}%\n")


# ── decay weights ──────────────────────────────────────────────────
def make_decay(df, hl=120, floor=0.5):
    ds    = np.sort(df['date'].unique())
    d2i   = {pd.Timestamp(d): i for i, d in enumerate(ds)}
    delta = (len(ds) - 1) - df['date'].map(d2i).values
    return np.maximum(np.exp(-np.log(2) * delta / hl), floor)


# ── score_prop weighting ──────────────────────────────────────────
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


# ── train baseline + regime models on full pre-April data ────────
def make_xgb(seed=42):
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=seed)


sw_decay = make_decay(train_df)
print(f"Train rows: {len(train_df):,}")

print("Training baseline (single model, exp_021 reproduction)...")
base_m = make_xgb()
base_m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
           sample_weight=sw_decay, verbose=False)

print("Training BULL specialist (sample weight = decay × regime)...")
bull_m = make_xgb()
bull_m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
           sample_weight=sw_decay * train_df['regime'].values, verbose=False)

print("Training SIDE specialist (sample weight = decay × (1 - regime))...")
side_m = make_xgb()
side_m.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN],
           sample_weight=sw_decay * (1.0 - train_df['regime'].values), verbose=False)
print()


# ── April daily backtest helpers ──────────────────────────────────
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


def predict_combined(pred_df, regime, mode='soft'):
    s_bull = pd.Series(bull_m.predict(pred_df[FEATURE_COLUMNS]),
                       index=pred_df['stock_code'].values)
    s_side = pd.Series(side_m.predict(pred_df[FEATURE_COLUMNS]),
                       index=pred_df['stock_code'].values)
    if mode == 'soft':
        return regime * s_bull + (1.0 - regime) * s_side
    elif mode == 'hard':
        return s_bull if regime >= 0.5 else s_side
    elif mode == 'bull_only':
        return s_bull
    elif mode == 'side_only':
        return s_side


def eval_window(scores, sd, bd):
    weights = score_prop_w(scores)
    bp  = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    w_v   = weights[valid] / weights[valid].sum()
    port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    return port, bench


# ── 1. April direct comparison ────────────────────────────────────
print("=" * 90)
print("1)  April Daily Backtest — exp_021 vs Regime Ensemble")
print("=" * 90)
print(f"{'sell':^12}{'buy':^12}{'r(buy)':^9}{'baseline':^11}{'soft':^11}"
      f"{'hard':^11}{'bull_only':^11}{'side_only':^11}{'bench':^9}")
print('-' * 100)

cols = ['baseline', 'soft', 'hard', 'bull_only', 'side_only']
results = {c: [] for c in cols}

for sd, bd in april_pairs:
    pred_df = buy_frames[bd]
    if len(pred_df) < TOP_K:
        continue
    r = float(regime_series.get(bd, 0.5))

    s_base = pd.Series(base_m.predict(pred_df[FEATURE_COLUMNS]),
                       index=pred_df['stock_code'].values)
    portB, bench = eval_window(s_base, sd, bd)
    portS, _     = eval_window(predict_combined(pred_df, r, 'soft'), sd, bd)
    portH, _     = eval_window(predict_combined(pred_df, r, 'hard'), sd, bd)
    portU, _     = eval_window(predict_combined(pred_df, r, 'bull_only'), sd, bd)
    portW, _     = eval_window(predict_combined(pred_df, r, 'side_only'), sd, bd)

    results['baseline'].append({'sell': sd, 'r': r, 'port': portB, 'bench': bench, 'excess': portB - bench})
    results['soft'].append    ({'sell': sd, 'r': r, 'port': portS, 'bench': bench, 'excess': portS - bench})
    results['hard'].append    ({'sell': sd, 'r': r, 'port': portH, 'bench': bench, 'excess': portH - bench})
    results['bull_only'].append({'sell': sd, 'r': r, 'port': portU, 'bench': bench, 'excess': portU - bench})
    results['side_only'].append({'sell': sd, 'r': r, 'port': portW, 'bench': bench, 'excess': portW - bench})

    print(f"{str(sd.date()):^12}{str(bd.date()):^12}{r:^9.3f}"
          f"{(portB-bench)*100:^+11.3f}{(portS-bench)*100:^+11.3f}"
          f"{(portH-bench)*100:^+11.3f}{(portU-bench)*100:^+11.3f}"
          f"{(portW-bench)*100:^+11.3f}{bench*100:^+9.2f}")

dfs = {c: pd.DataFrame(results[c]) for c in cols}

print('\n' + '=' * 70)
print(f"{'指标':^14}", end='')
for c in cols:
    print(f"{c:^11}", end='')
print()
print('-' * 70)

for label, fn in [
    ('mean_excess%',  lambda d: d['excess'].mean()*100),
    ('std_excess%',   lambda d: d['excess'].std()*100),
    ('sharpe',        lambda d: d['excess'].mean()/d['excess'].std()),
    ('win_rate',      lambda d: (d['excess']>0).mean()),
    ('max_loss%',     lambda d: d['excess'].min()*100),
]:
    print(f"  {label:^12}", end='')
    for c in cols:
        print(f"{fn(dfs[c]):^+11.3f}", end='')
    print()


# ── 2. Performance grouped by regime state ────────────────────────
print('\n' + '=' * 90)
print("2)  Performance by Regime State (April only)")
print("=" * 90)

ref = dfs['baseline']
bull_mask = ref['r'] >= 0.5
side_mask = ref['r'] <  0.5

print(f"\n{'group':^12}{'N':^6}{'baseline%':^12}{'soft%':^12}{'hard%':^12}{'bull_only%':^12}{'side_only%':^12}")
print('-' * 80)

for label, mask in [('regime>=0.5', bull_mask), ('regime< 0.5', side_mask)]:
    n = mask.sum()
    if n == 0:
        continue
    row = f"{label:^12}{n:^6}"
    for c in cols:
        e = dfs[c].loc[mask.values, 'excess'].mean()*100
        row += f"{e:^+12.3f}"
    print(row)


# ── 3. Walk-forward Oct 2025 -> Apr 2026 ──────────────────────────
print('\n' + '=' * 90)
print("3)  Walk-Forward Backtest  Oct 2025 - Apr 2026  (monthly retrain)")
print("=" * 90)

start_eval = pd.Timestamp('2025-10-01')
end_eval   = pd.Timestamp('2026-04-30')
eval_dates = [pd.Timestamp(d) for d in all_trading
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

# train monthly: produce (base, bull, side) trio per month
model_cache = {}
print("Training monthly walk-forward models (base / bull / side)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr     = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr) < 5000:
        continue
    tr        = tr.copy()
    tr['regime'] = tr['date'].map(regime_series).fillna(0.5)
    sw_d      = make_decay(tr)

    b_mdl = make_xgb()
    b_mdl.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN], sample_weight=sw_d, verbose=False)

    u_mdl = make_xgb()
    u_mdl.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
              sample_weight=sw_d * tr['regime'].values, verbose=False)

    w_mdl = make_xgb()
    w_mdl.fit(tr[FEATURE_COLUMNS], tr[TARGET_COLUMN],
              sample_weight=sw_d * (1.0 - tr['regime'].values), verbose=False)

    model_cache[ms] = (b_mdl, u_mdl, w_mdl, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr):,}")


def get_models(sell_date):
    cands = [ms for ms in model_cache if ms <= sell_date]
    if not cands:
        return None
    return model_cache[max(cands)]


# walk-forward daily eval
wf = {'base': [], 'soft': [], 'hard': []}
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
    b_mdl, u_mdl, w_mdl, _ = mdls
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        continue
    r = float(regime_series.get(bd, 0.5))

    s_b = pd.Series(b_mdl.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    s_u = pd.Series(u_mdl.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    s_w = pd.Series(w_mdl.predict(pred_df[FEATURE_COLUMNS]),
                    index=pred_df['stock_code'].values)
    s_soft = r * s_u + (1.0 - r) * s_w
    s_hard = s_u if r >= 0.5 else s_w

    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    for label, scores in [('base', s_b), ('soft', s_soft), ('hard', s_hard)]:
        port, _ = eval_window(scores, sd, bd)
        wf[label].append({'sell': sd, 'month': sd.strftime('%Y-%m'),
                          'r': r, 'port': port, 'bench': bench,
                          'excess': port - bench})

wfs = {k: pd.DataFrame(v) for k, v in wf.items()}

print(f"\n{'month':^10}{'N':^5}", end='')
for c in ['base', 'soft', 'hard']:
    print(f"{c+'_excess%':^14}", end='')
print(f"{'bench%':^10}{'mean_r':^8}")
print('-' * 75)

for mo in sorted(wfs['base']['month'].unique()):
    sub_b = wfs['base']  [wfs['base']  ['month'] == mo]
    sub_s = wfs['soft']  [wfs['soft']  ['month'] == mo]
    sub_h = wfs['hard']  [wfs['hard']  ['month'] == mo]
    if len(sub_b) == 0:
        continue
    print(f"{mo:^10}{len(sub_b):^5}"
          f"{sub_b['excess'].mean()*100:^+14.3f}"
          f"{sub_s['excess'].mean()*100:^+14.3f}"
          f"{sub_h['excess'].mean()*100:^+14.3f}"
          f"{sub_b['bench'].mean()*100:^+10.3f}"
          f"{sub_b['r'].mean():^8.3f}")

# overall
print('-' * 75)
for label, fn_label in [('mean_excess%', lambda d: d['excess'].mean()*100),
                          ('std_excess%',  lambda d: d['excess'].std()*100),
                          ('sharpe',       lambda d: d['excess'].mean()/d['excess'].std()),
                          ('win_rate',     lambda d: (d['excess']>0).mean())]:
    print(f"  {label:^10}    ", end='')
    for c in ['base', 'soft', 'hard']:
        print(f"{fn_label(wfs[c]):^+14.4f}", end='')
    print()

# regime-conditional (full walk-forward)
print(f"\n  按 regime 状态分组 (walk-forward 全期):")
for label, mask_fn in [('r>=0.5 (uptrend)', lambda d: d['r'] >= 0.5),
                        ('r<0.5  (sideways)', lambda d: d['r'] <  0.5)]:
    print(f"  {label:^20}", end='')
    for c in ['base', 'soft', 'hard']:
        sub = wfs[c][mask_fn(wfs[c])]
        e   = sub['excess'].mean()*100 if len(sub) > 0 else 0
        print(f"  {c}={e:+.3f}% (N={len(sub)})", end='')
    print()
