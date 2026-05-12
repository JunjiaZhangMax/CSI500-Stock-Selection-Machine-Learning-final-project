"""Ridge Regression + XGBoost Ensemble.

Motivation: in trending markets, linear momentum signals (ret_5d, ret_20d,
close_over_ma20 ...) are often sufficient and less noisy than the full XGB
non-linear model.  A Ridge model is fast, interpretable, and complementary.

Pipeline:
  1. Train Ridge (with time-decay weights, StandardScaler) on same features.
  2. Train XGBoost (exp_021 config) on same data.
  3. Blend: score = a * xgb + (1-a) * ridge,  a in {0.3, 0.5, 0.7}
  4. Also test regime-adaptive blend: higher Ridge weight when r > 0.5 (bull).
  5. April daily backtest + walk-forward Oct-Apr comparison.
"""
import sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


# ── Regime series ───────────────────────────────────────────────────
def compute_regime_series(idx_close, lookback=20):
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

# ── Helpers ──────────────────────────────────────────────────────────
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


def eval_window(scores, sd, bd):
    weights = score_prop_w(scores)
    bp  = panel_close.loc[bd].reindex(weights.index)
    sp_ = panel_close.loc[sd].reindex(weights.index)
    valid = (~bp.isna()) & (~sp_.isna())
    w_v   = weights[valid] / weights[valid].sum()
    port  = float((w_v * (sp_[valid] / bp[valid] - 1)).sum())
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)
    return port, bench


# ── Model factories ──────────────────────────────────────────────────
def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
        reg_lambda=1.0, tree_method='hist', n_jobs=-1, random_state=42)


def make_ridge(alpha=1.0):
    return Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=alpha)),
    ])


def train_pair(tr, ridge_alpha=1.0):
    """Return (xgb_model, ridge_pipeline) trained on tr."""
    sw = make_decay(tr)
    X  = tr[FEATURE_COLUMNS].values
    y  = tr[TARGET_COLUMN].values

    xm = make_xgb()
    xm.fit(X, y, sample_weight=sw, verbose=False)

    rm = make_ridge(ridge_alpha)
    rm.fit(X, y, ridge__sample_weight=sw)

    return xm, rm


def blend(xm, rm, pred_df, alpha):
    """alpha = weight on XGB, (1-alpha) = weight on Ridge."""
    X = pred_df[FEATURE_COLUMNS].values
    s_xgb   = xm.predict(X)
    s_ridge = rm.predict(X)
    # Standardize each to zero-mean / unit-std so scales are comparable
    def _norm(s):
        s = s - s.mean()
        std = s.std()
        return s / std if std > 0 else s
    combined = alpha * _norm(s_xgb) + (1 - alpha) * _norm(s_ridge)
    return pd.Series(combined, index=pred_df['stock_code'].values)


# ── Train on full pre-April data ─────────────────────────────────────
train_df = training_frame(panel, max_date=TRAIN_CUTOFF, target=TARGET_COLUMN)
print(f"Train rows: {len(train_df):,}  ({train_df['date'].min().date()} ~ {train_df['date'].max().date()})")
print("Training XGBoost + Ridge on training data...")
xm_full, rm_full = train_pair(train_df)
print("Done.\n")

# Ridge coefficients (standardised) — tells us which features matter linearly
ridge_coef = rm_full.named_steps['ridge'].coef_
coef_df = pd.Series(ridge_coef, index=FEATURE_COLUMNS).sort_values(key=abs, ascending=False)
print("Ridge top coefficients (|coef| sorted):")
for feat, coef in coef_df.items():
    bar = '+' * int(abs(coef) * 30) if coef > 0 else '-' * int(abs(coef) * 30)
    print(f"  {feat:28s}: {coef:+.4f}  {bar}")
print()


# ── April daily backtest ─────────────────────────────────────────────
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

ALPHAS = [1.0, 0.7, 0.5, 0.3]   # XGB weight in blend

print("=" * 110)
print("1)  April Daily Backtest -- XGB vs Ridge vs Blends")
print("=" * 110)
header = f"{'sell':^12}{'buy':^12}{'r':^7}"
for a in ALPHAS:
    label = 'xgb_only' if a == 1.0 else ('ridge_only' if a == 0.0 else f'xgb{int(a*10)}r{int((1-a)*10)}')
    header += f"{label+'_ex%':^14}"
header += f"{'adaptive_ex%':^14}{'bench%':^9}"
print(header)
print('-' * len(header))

cols = [f'a{int(a*10)}' for a in ALPHAS] + ['adaptive']
results = {c: [] for c in cols}

for sd, bd in april_pairs:
    pred_df = buy_frames[bd]
    if len(pred_df) < TOP_K:
        continue
    r = float(regime_series.get(bd, 0.5))

    row_str = f"{str(sd.date()):^12}{str(bd.date()):^12}{r:^7.3f}"
    bench_val = None

    for a, col in zip(ALPHAS, cols[:-1]):
        sc = blend(xm_full, rm_full, pred_df, a)
        port, bench = eval_window(sc, sd, bd)
        if bench_val is None:
            bench_val = bench
        results[col].append({'sell': sd, 'r': r, 'port': port, 'bench': bench,
                              'excess': port - bench})
        row_str += f"{(port-bench)*100:^+14.3f}"

    # Regime-adaptive: r>0.5 -> give Ridge more weight (linear momentum dominates in bull)
    #   alpha_xgb = 0.3 + 0.4*(1-r)  => at r=1: 0.3 xgb / 0.7 ridge
    #                                  => at r=0: 0.7 xgb / 0.3 ridge
    alpha_adaptive = 0.3 + 0.4 * (1.0 - r)
    sc_adap = blend(xm_full, rm_full, pred_df, alpha_adaptive)
    port_a, _ = eval_window(sc_adap, sd, bd)
    results['adaptive'].append({'sell': sd, 'r': r, 'port': port_a, 'bench': bench_val,
                                'excess': port_a - bench_val})
    row_str += f"{(port_a-bench_val)*100:^+14.3f}{bench_val*100:^+9.2f}"
    print(row_str)

dfs = {c: pd.DataFrame(results[c]) for c in cols}

print('\n' + '=' * 80)
print(f"{'Metric':^18}", end='')
for a, col in zip(ALPHAS, cols[:-1]):
    label = 'xgb' if a == 1.0 else f'xgb{int(a*10)}r{int((1-a)*10)}'
    print(f"{label:^14}", end='')
print(f"{'adaptive':^14}")
print('-' * 80)

all_cols = cols[:-1] + ['adaptive']
for label, fn in [
    ('mean_excess%',  lambda d: d['excess'].mean()*100),
    ('std_excess%',   lambda d: d['excess'].std()*100),
    ('sharpe',        lambda d: d['excess'].mean()/d['excess'].std()),
    ('win_rate',      lambda d: (d['excess']>0).mean()),
    ('max_loss%',     lambda d: d['excess'].min()*100),
]:
    print(f"  {label:^16}", end='')
    for col in all_cols:
        print(f"{fn(dfs[col]):^+14.4f}", end='')
    print()


# ── Walk-forward Oct 2025 -> Apr 2026 ────────────────────────────────
print('\n' + '=' * 90)
print("2)  Walk-Forward  Oct 2025 - Apr 2026  (monthly retrain)")
print("=" * 90)

start_eval = pd.Timestamp('2025-10-01')
end_eval   = pd.Timestamp('2026-05-08')
eval_dates = [pd.Timestamp(d) for d in all_trading
              if start_eval <= pd.Timestamp(d) <= end_eval]
month_starts = sorted(set(d.replace(day=1) for d in eval_dates))

model_cache = {}
print("Training monthly models (xgb + ridge)...")
for ms in month_starts:
    available = [d for d in all_trading if pd.Timestamp(d) < ms]
    if len(available) < 80:
        continue
    cutoff = pd.Timestamp(available[-4])
    tr = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
    if len(tr) < 5000:
        continue
    xm, rm = train_pair(tr)
    model_cache[ms] = (xm, rm, cutoff)
    print(f"  {ms.strftime('%Y-%m')}: cutoff={cutoff.date()}  rows={len(tr):,}")


def get_models(sell_date):
    cands = [ms for ms in model_cache if ms <= sell_date]
    return model_cache[max(cands)] if cands else None


wf = {c: [] for c in all_cols}
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
    xm, rm, _ = mdls
    pred_df = panel[panel['date'] == bd].dropna(subset=FEATURE_COLUMNS)
    if len(pred_df) < TOP_K:
        continue
    r = float(regime_series.get(bd, 0.5))
    bench = float(idx_close.loc[sd] / idx_close.loc[bd] - 1)

    for a, col in zip(ALPHAS, cols[:-1]):
        sc   = blend(xm, rm, pred_df, a)
        port, _ = eval_window(sc, sd, bd)
        wf[col].append({'sell': sd, 'month': sd.strftime('%Y-%m'),
                        'r': r, 'port': port, 'bench': bench,
                        'excess': port - bench})

    alpha_a = 0.3 + 0.4 * (1.0 - r)
    sc_a = blend(xm, rm, pred_df, alpha_a)
    port_a, _ = eval_window(sc_a, sd, bd)
    wf['adaptive'].append({'sell': sd, 'month': sd.strftime('%Y-%m'),
                           'r': r, 'port': port_a, 'bench': bench,
                           'excess': port_a - bench})

wfs = {c: pd.DataFrame(wf[c]) for c in all_cols}

print(f"\n{'month':^10}{'N':^5}", end='')
for col in all_cols:
    lbl = 'xgb' if col == 'a10' else (col if col == 'adaptive' else col)
    print(f"{lbl+'_ex%':^14}", end='')
print(f"{'bench%':^10}{'r':^7}")
print('-' * (10 + 5 + 14 * len(all_cols) + 17))

for mo in sorted(wfs['a10']['month'].unique()):
    sub0 = wfs['a10'][wfs['a10']['month'] == mo]
    print(f"{mo:^10}{len(sub0):^5}", end='')
    for col in all_cols:
        sub = wfs[col][wfs[col]['month'] == mo]
        print(f"{sub['excess'].mean()*100:^+14.3f}", end='')
    print(f"{sub0['bench'].mean()*100:^+10.3f}{sub0['r'].mean():^7.3f}")

print('-' * (10 + 5 + 14 * len(all_cols) + 17))
for label, fn in [('mean_excess%', lambda d: d['excess'].mean()*100),
                   ('std_excess%',  lambda d: d['excess'].std()*100),
                   ('sharpe',       lambda d: d['excess'].mean()/d['excess'].std()),
                   ('win_rate',     lambda d: (d['excess']>0).mean())]:
    print(f"  {label:^12}", end='')
    for col in all_cols:
        print(f"{fn(wfs[col]):^+14.4f}", end='')
    print()

# ── Generate Window 2 submission with best blend ─────────────────────
print('\n' + '=' * 60)
print("3)  Window 2 Submission (prediction date = latest data)")
print("=" * 60)
pred_date = panel['date'].max()
pred_df   = panel[panel['date'] == pred_date].dropna(subset=FEATURE_COLUMNS)
print(f"Prediction date: {pred_date.date()}  ({len(pred_df)} stocks)")

current_r = float(regime_series.get(pred_date, 0.5))
print(f"Current regime score r = {current_r:.3f}  "
      f"({'strong bull' if current_r > 0.6 else ('neutral' if current_r > 0.3 else 'weak')})")

# Retrain on all data up to pred_date - 3 (forward horizon embargo)
all_dates = sorted(panel['date'].unique())
d_idx     = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}
cutoff_i  = d_idx[pred_date] - 3
cutoff    = pd.Timestamp(all_dates[cutoff_i])
tr_final  = training_frame(panel, max_date=cutoff, target=TARGET_COLUMN)
print(f"Final model train cutoff: {cutoff.date()}  rows: {len(tr_final):,}")
xm_final, rm_final = train_pair(tr_final)

alpha_w2 = 0.3 + 0.4 * (1.0 - current_r)   # regime-adaptive
print(f"Regime-adaptive alpha_xgb = {alpha_w2:.2f}  (Ridge weight = {1-alpha_w2:.2f})")

for a_val, label in [(1.0, 'xgb_only'), (0.5, 'xgb5r5'), (alpha_w2, 'adaptive')]:
    sc = blend(xm_final, rm_final, pred_df, a_val)
    w  = score_prop_w(sc)
    out = Path(f'outputs/submissions/exp_ridge_{label}_window2.csv')
    pd.DataFrame({'stock_code': w.index, 'weight': w.values}).to_csv(out, index=False)
    print(f"  Saved: {out.name}  (top stock: {w.idxmax()} {w.max()*100:.1f}%)")
