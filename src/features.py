"""
Feature engineering for the CSI500 stock-selection baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "ret_1d", "ret_5d", "ret_10d", "ret_20d", "ret_60d",
    "vol_20d", "volume_z_20d", "turnover_ma_20d",
    "close_over_ma20", "close_over_ma60", "rsi_14",
    "ret_5d_rank", "ret_20d_rank", "vol_20d_rank",
    "amplitude_ma_20d", "amplitude_ma_20d_rank",
]

# Extended feature set — requires industry_map to be passed to build_features().
INDUSTRY_FEATURE_COLUMNS = FEATURE_COLUMNS + [
    "ret_5d_ind_z", "ret_20d_ind_z", "vol_20d_ind_z",
    "ret_5d_ind_rank", "ret_20d_ind_rank",
]

TARGET_COLUMN      = "target_3d"
TARGET_RANK_COLUMN = "target_3d_rank"
FORWARD_HORIZON    = 3


def _per_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features that only depend on a single stock's time series."""
    df = df.sort_values("date").copy()
    close = df["close"]

    # momentum
    df["ret_1d"]   = close.pct_change(1)
    df["ret_5d"]   = close.pct_change(5)
    df["ret_10d"]  = close.pct_change(10)
    df["ret_20d"]  = close.pct_change(20)
    df["ret_60d"]  = close.pct_change(60)
    df["ret_120d"] = close.pct_change(120)

    # volatility
    df["vol_5d"]  = df["ret_1d"].rolling(5).std()
    df["vol_20d"] = df["ret_1d"].rolling(20).std()
    df["vol_ratio"] = df["vol_5d"] / df["vol_20d"].replace(0, np.nan)

    # volume z-score
    vol = df["volume"].astype(float)
    vol_mean = vol.rolling(20).mean()
    vol_std  = vol.rolling(20).std().replace(0, np.nan)
    df["volume_z_20d"] = (vol - vol_mean) / vol_std

    # turnover
    if "turnover" in df.columns:
        df["turnover_ma_20d"] = df["turnover"].astype(float).rolling(20).mean()
    else:
        df["turnover_ma_20d"] = np.nan

    # Amihud illiquidity: mean(|ret_1d| / amount) over 20d, scaled
    if "amount" in df.columns:
        amount = df["amount"].astype(float).replace(0, np.nan)
        df["amihud_20d"] = (df["ret_1d"].abs() / amount).rolling(20).mean() * 1e6
    else:
        df["amihud_20d"] = np.nan

    # price-over-MA
    df["close_over_ma20"] = close / close.rolling(20).mean() - 1.0
    df["close_over_ma60"] = close / close.rolling(60).mean() - 1.0

    # 120-day high proximity (anchoring effect; always ≤ 0)
    df["close_over_high52w"] = close / close.rolling(120).max() - 1.0

    # RSI-14
    delta = close.diff()
    up   = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean().replace(0, np.nan)
    df["rsi_14"] = 100 - 100 / (1 + up / down)

    # intraday range (OHLC liquidity proxy)
    if "high" in df.columns and "low" in df.columns:
        df["intraday_range_20d"] = (
            ((df["high"] - df["low"]) / close).rolling(20).mean()
        )
        # amplitude = (high - low) / prev_close; 20d MA smooths noise
        prev_close = close.shift(1).replace(0, np.nan)
        df["amplitude_1d"]    = (df["high"] - df["low"]) / prev_close
        df["amplitude_ma_20d"] = df["amplitude_1d"].rolling(20).mean()
    else:
        df["intraday_range_20d"] = np.nan
        df["amplitude_1d"]       = np.nan
        df["amplitude_ma_20d"]   = np.nan

    df[TARGET_COLUMN] = close.shift(-FORWARD_HORIZON) / close - 1.0
    return df


def _industry_relative_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Within-industry z-score and rank for key features, computed per date."""
    for col in ["ret_5d", "ret_20d", "vol_20d"]:
        grp  = panel.groupby(["date", "industry"])[col]
        mean_ = grp.transform("mean")
        std_  = grp.transform("std").replace(0, np.nan)
        panel[f"{col}_ind_z"] = (panel[col] - mean_) / std_
    for col in ["ret_5d", "ret_20d"]:
        panel[f"{col}_ind_rank"] = (
            panel.groupby(["date", "industry"])[col].rank(method="average", pct=True)
        )
    return panel


def _cross_sectional_ranks(panel: pd.DataFrame) -> pd.DataFrame:
    """Daily cross-sectional percentile rank of selected features and target."""
    for col in ["ret_1d", "ret_5d", "ret_20d", "ret_60d", "vol_20d", "amihud_20d", "amplitude_ma_20d"]:
        panel[f"{col}_rank"] = (
            panel.groupby("date")[col].rank(method="average", pct=True)
        )
    panel[TARGET_RANK_COLUMN] = (
        panel.groupby("date")[TARGET_COLUMN].rank(method="average", pct=True)
    )
    return panel


def build_features(
    prices: pd.DataFrame,
    industry_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a (date, stock_code) panel with FEATURE_COLUMNS and TARGET_COLUMN.

    Pass industry_map (columns: stock_code, industry) to also compute
    INDUSTRY_FEATURE_COLUMNS (within-industry z-scores and ranks).
    """
    required = {"date", "stock_code", "close", "volume"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"prices is missing required columns: {missing}")

    prices = prices.copy()
    prices["date"] = pd.to_datetime(prices["date"])
    panel = (
        prices.groupby("stock_code", group_keys=False)
        .apply(_per_stock_features)
        .reset_index(drop=True)
    )
    panel = _cross_sectional_ranks(panel)

    if industry_map is not None:
        panel = panel.merge(
            industry_map[["stock_code", "industry"]],
            on="stock_code", how="left",
        )
        panel = _industry_relative_features(panel)

    return panel


def training_frame(
    panel: pd.DataFrame,
    min_date=None,
    max_date=None,
    target: str = TARGET_COLUMN,
) -> pd.DataFrame:
    """Rows usable for supervised training: all features present AND target present."""
    df = panel.dropna(subset=FEATURE_COLUMNS + [target]).copy()
    if min_date is not None:
        df = df[df["date"] >= pd.Timestamp(min_date)]
    if max_date is not None:
        df = df[df["date"] <= pd.Timestamp(max_date)]
    return df


def prediction_frame(panel: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Rows for a single prediction date (defaults to the latest date)."""
    if as_of is None:
        as_of = panel["date"].max()
    as_of = pd.Timestamp(as_of)
    return panel[panel["date"] == as_of].dropna(subset=FEATURE_COLUMNS).copy()
