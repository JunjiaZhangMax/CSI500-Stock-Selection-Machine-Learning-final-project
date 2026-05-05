# Experiment Ideas

Strategies and methods to try for improving excess return over CSI500.

---

## Feature Engineering

- [ ] Industry/sector dummy variables (Shenwan Level-1)
- [ ] Fundamental factors: PE, PB, ROE from financial reports
- [ ] Earnings revision momentum (analyst estimate changes)
- [ ] Intraday VWAP deviation as a microstructure signal
- [ ] Overnight return vs. intraday return split
- [ ] 52-week high proximity (anchoring effect)
- [ ] Short-term reversal (1d or 2d return mean-reversion)

## Model

- [x] LightGBM (faster, often comparable to XGBoost on tabular data)
- [x] XGBoost baseline
- [x] Cross-sectional rank target: replace raw return with daily percentile rank to remove market beta
- [ ] Two-stage: classify top/bottom quintile → rank within top quintile
- [ ] Ridge regression on rank-normalized features (simple baseline)
- [ ] Ensemble: average XGBoost + LightGBM + linear model predictions
- [ ] Walk-forward expanding window instead of rolling window

## Portfolio Construction

- [ ] Mean-variance optimization with IC-weighted alpha signals
- [ ] Black-Litterman with CSI500 market cap weights as prior
- [ ] Neutralize industry exposure (weight per industry ≈ index weight)
- [ ] Turnover penalty: penalize large deviations from previous portfolio
- [ ] Equal-weight top-K instead of rank-weighted (simpler, sometimes better)

## Risk / Evaluation

- [x] top-20% IC and top-50% IC (predicted-top subset IC)
- [x] Hit Rate @top_k vs actual top-2K (lenient selection quality metric)
- [ ] ICIR (IC / std_IC) stability tracking
- [ ] Sector-neutral IC to isolate stock selection from sector bets
- [ ] Drawdown analysis per backtest window

---

## Completed Experiments

Metrics explanation:
- `val_IC`: mean Spearman IC across all stocks (all 10 folds)
- `top20_IC`: IC restricted to predicted top-20% stocks per day
- `hit_rate`: fraction of predicted top-K found in actual top-2K (lenient; exp_001~005 used strict top-K)
- `bt_sharpe`: rolling backtest Sharpe over the bt window

| # | exp_name | Model | target | top_k | folds | bt window | val_IC ± std | top20_IC | hit_rate | bt_sharpe | notes |
|---|----------|-------|--------|-------|-------|-----------|--------------|----------|----------|-----------|-------|
| 001 | exp_001_xgboost_baseline | XGBoost | raw 5d | 50 | 5 | 2026-03~04 | +0.0021 ± 0.117 | — | — | 0.071 | reference baseline |
| 002 | exp_002_lightgbm | LightGBM | raw 5d | 50 | 5 | 2026-03~04 | -0.0020 ± 0.107 | — | — | 0.269 | early stopping → 93 trees |
| 003 | exp_003_lgbm_rank_target | LightGBM | rank 5d | 50 | 5 | 2026-03~04 | +0.0315 ± 0.078 | — | — | 0.075 | rank target removes market beta |
| 004 | exp_004_lgbm_rank_target_3d_top30 | LightGBM | rank 3d | 30 | 5 | 2026-03~04 | +0.0438 ± 0.059 | — | — | 0.271 | 3d horizon cleaner signal |
| 005 | exp_005_lgbm_rank_target_3d_top30_10folds | LightGBM | rank 3d | 30 | 10 | 2025-10~2026-04 | +0.0314 ± 0.069 | +0.0439 | 0.054* | 0.541 | 10 folds more honest; fold5 Dec/Jan IC=-0.12 |
| 006 | exp_006_xgboost_3d_top50 | XGBoost | raw 3d | 50 | 10 | 2025-10~2026-04 | -0.0119 ± 0.081 | +0.0177 | **0.255** | 0.535 | negative full IC but positive top20 IC; high hit_rate |
| 007 | exp_007_lgbm_rank_target_3d_top50 | LightGBM | rank 3d | 50 | 10 | 2025-10~2026-04 | +0.0314 ± 0.069 | +0.0439 | 0.173 | 0.477 | top_k=50 vs 005: hit_rate 3x better, sharpe slightly lower |
| 008 | exp_008_ensemble_xgb_lgbm_rank_3d_top50 | XGB+LGBM ensemble | rank 3d | 50 | 10 | 2025-10~2026-04 | +0.0276 ± 0.074 | **+0.0546** | 0.173 | **0.556** | top20_IC最高；bt_sharpe超过单模型；val_IC因XGB拖累略降 |

*exp_005 hit_rate used strict actual_k=30; exp_006/007 use lenient actual_k=100 (top-2K).
