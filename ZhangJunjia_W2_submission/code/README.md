# CSI500 Stock Selection Submission Package

Author: Junjia (Max) Zhang  ·  jz7842@nyu.edu

This package contains the final submissions for both prediction windows
plus the code to reproduce them byte-for-byte from the included data.

## Contents

```
submission_pkg/
├── README.md                                    this file
├── requirements.txt                             Python dependencies
├── data/
│   ├── prices.parquet                           CSI500 daily OHLCV (Oct 2024 - May 2026)
│   ├── index.parquet                            CSI500 index OHLCV (benchmark)
│   └── constituents.csv                         current CSI500 member list
├── src/
│   ├── features.py                              feature engineering module
│   └── validate_submission.py                   submission format validator
├── scripts/
│   ├── reproduce_w1.py                          regenerates w1_021 submission
│   └── reproduce_w2.py                          regenerates w2_023 submission
└── submissions/
    ├── w1_021_score_prop_cap8.csv               Window 1 final submission
    └── w2_023_pow2_softvol.csv                  Window 2 final submission
```

## Quick start

```bash
pip install -r requirements.txt

# Reproduce both submissions
python scripts/reproduce_w1.py
python scripts/reproduce_w2.py

# Validate
python src/validate_submission.py submissions/w1_021_score_prop_cap8.csv
python src/validate_submission.py submissions/w2_023_pow2_softvol.csv
```

Both reproducer scripts produce **byte-identical** output to the included
submission CSVs (verified with `diff`).

## Window 1 — `w1_021_score_prop_cap8.csv`

**Hold period:** May 6 - May 8 2026 (3 trading days)
**As-of date:** April 30 2026 (last trading day before May Day holiday)
**Method:** XGBoost on 3-day forward return + score-proportional weighting

| Component | Value |
|-----------|-------|
| Model | Single XGBoost (n_estimators=400, max_depth=5, lr=0.05) |
| Target | `target_3d` = close(t+3) / close(t) - 1 |
| Features | 16 features from `src/features.py` (momentum, vol, RSI, MA ratios, ranks, amplitude) |
| Training cutoff | 2026-04-08 (hardcoded to give ~3 weeks embargo before May 6) |
| Sample weights | Exponential time-decay, half-life=120 days, floor=0.5 |
| Selection | Top 50 stocks by predicted score |
| Weighting | Score-proportional, iteratively capped at 8% |

## Window 2 — `w2_023_pow2_softvol.csv`

**Hold period:** May 11 - May 15 2026 (5 trading days)
**As-of date:** May 8 2026
**Method:** 2-target ensemble + concentrated soft vol-adj weighting

| Component | Value |
|-----------|-------|
| Models | 2x XGBoost (target_3d + target_5d, same hyperparameters as W1) |
| Training cutoff | Last trading day before May 1st minus 5-day embargo (= 2026-04-23) |
| Sample weights | Exponential time-decay, half-life=60 days, floor=0.5 |
| Ensemble | Rank-percentile average: 0.5 × rank(score_3d) + 0.5 × rank(score_5d) |
| Blacklist | Excludes 002261 (regulatory warning issued 2026-05-07/08) |
| Selection | Top 30 stocks by ensemble score (after blacklist) |
| Weighting | `(score_norm)^2 / vol_20d^0.5`, iteratively capped at 10% |
| Min weight | 0.1% per stock (positive weight constraint) |

The `score^2` term concentrates weight on top-ranked stocks; the `vol^0.5`
denominator (instead of `1/vol`) softens the vol penalty so high-prediction
high-vol stocks (e.g. 300857) retain meaningful weight.

## Reproducibility notes

- Tested with: Python 3.13, xgboost 3.x, pandas 2.x, numpy 2.x
- All XGBoost training uses `random_state=42` and `tree_method='hist'`
- `pd.pivot_table` ordering is stable in tested versions
- Output CSV files match originals byte-for-byte under these conditions

## Constraint compliance (verified)

Both submissions satisfy:
- ≥ 30 stocks with positive weight
- Each weight ≤ 10% (W1 ≤ 8%)
- Weights sum to 1.0 (within 1e-4 tolerance)
- All stocks are valid 6-digit CSI500 constituent codes
