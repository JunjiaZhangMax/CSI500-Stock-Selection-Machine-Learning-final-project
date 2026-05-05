# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CSI500 stock-selection competition. The goal is to build a long-only portfolio of CSI500 constituents that maximizes excess return over the CSI500 index benchmark, evaluated over two 5-trading-day windows.

**Competition rules:**
- Portfolio must hold ≥ 30 stocks with positive weight
- No single weight may exceed 10%
- Weights must sum to 1.0 (tolerance 1e-4)
- All stocks must be current CSI500 constituents (6-digit zero-padded codes)

## Common Commands

All commands are run from the **project root**.

```bash
# Install dependencies
pip install -r requirements.txt

# Download data (initial, ~10-30 min, ~500 API calls via akshare)
python scripts/download_data.py --start 20250101 --end 20260430

# Incremental data update
python scripts/download_data.py --update --end 20260430

# Generate a submission (latest date)
python src/baseline_xgboost.py --top-k 50 --out outputs/submissions/week1.csv

# Generate a submission as of a specific date (no future leakage)
python src/baseline_xgboost.py --as-of 20260503 --top-k 50 --out outputs/submissions/week1.csv

# Validate submission format
python src/validate_submission.py outputs/submissions/week1.csv

# Score submission against realized returns
python src/score_submission.py outputs/submissions/week1.csv --start 2026-04-21 --end 2026-04-25

# Rolling backtest across multiple windows (calls baseline_xgboost + score_submission via subprocess)
python src/rolling_backtest.py
```

## Architecture

Data flows in one direction: download → features → model → portfolio → evaluation.

**`scripts/download_data.py`** — Fetches CSI500 constituent list and daily OHLCV (forward-adjusted) from akshare's sina backend. Writes three files to `data/`:
- `prices.parquet` — per-stock daily OHLCV panel (columns: date, stock_code, open, close, high, low, volume, amount, turnover)
- `index.parquet` — CSI500 index OHLCV (benchmark)
- `constituents.csv` — current index members

**`src/features.py`** — Stateless feature engineering module. `build_features(prices)` returns a (date × stock_code) panel with `FEATURE_COLUMNS` and `TARGET_COLUMN` (`target_5d` = 5-day forward return). Features are: momentum returns (1/5/10/20/60d), 20d vol, volume z-score, turnover MA, price-over-MA ratios, RSI-14, plus cross-sectional rank of ret_5d/ret_20d/vol_20d. Rows with NaN features (first ~60 days per stock) are kept; callers filter via `training_frame()` or `prediction_frame()`.

**`src/baseline_xgboost.py`** — Main pipeline: loads parquet → calls `build_features` → time-splits with a 5-day embargo between train and validation → trains XGBoost → reports rank IC → outputs a weighted portfolio CSV. Also contains `rolling_backtest()` (used internally after training) and `build_portfolio()` which uses iterative rank-weighted capping to enforce the 10% constraint.

**`src/score_submission.py`** — Evaluates a submission CSV against actual prices. Entry price is `close(day_before_start)`, fallback to `open(start)` if no prior close. Exit is `close(end)`, fallback to last available close. Compares portfolio return to CSI500 index return.

**`src/validate_submission.py`** — Checks format compliance (columns, weight constraints, universe membership) before submission.

**`src/rolling_backtest.py`** — Orchestrates a sliding-window backtest over `data/prices.parquet` by calling `baseline_xgboost.py` and `score_submission.py` as subprocesses with `--as-of` flags. Summarizes excess return across windows.

## Key Constants

| Symbol | Value | Where |
|--------|-------|-------|
| `FORWARD_HORIZON` | 5 trading days | features.py, baseline_xgboost.py |
| `EMBARGO_DAYS` | 5 days | baseline_xgboost.py |
| `VAL_DAYS` | 10 days | baseline_xgboost.py |
| `MIN_STOCKS` | 30 | baseline_xgboost.py, validate_submission.py |
| `MAX_WEIGHT` | 0.10 | baseline_xgboost.py, validate_submission.py |

## Extending the Baseline

The natural extension points are:
- **`features.py`**: add new columns to `FEATURE_COLUMNS` (fundamentals, industry dummies, alternative data, better normalization)
- **`baseline_xgboost.py` `train_model()`**: swap or ensemble models; the interface expects `(train_df, val_df) → model` with `model.predict(X[FEATURE_COLUMNS])`
- **`build_portfolio()`**: alternative weighting or optimization schemes; must satisfy the `MIN_STOCKS`/`MAX_WEIGHT`/sum-to-1 constraints enforced by `assert`
