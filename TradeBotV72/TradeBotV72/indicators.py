"""
indicators.py — Technical indicators

FIXES v6.1:
  [BUG] rsi_divergence: slice comparison was comparing current bar against
        a window that included the current bar — always returned False.
        Fixed: compare close[-1] against close[-lookback:-1] (exclude current).
"""

import pandas as pd
import numpy as np


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta    = series.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    ema_fast   = ema(series, fast)
    ema_slow   = ema(series, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


def bollinger_bands(series: pd.Series, period=20, std=2.0):
    mid     = series.rolling(period).mean()
    std_val = series.rolling(period).std()
    upper   = mid + std * std_val
    lower   = mid - std * std_val
    return upper, mid, lower


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high  = df["high"]
    low   = df["low"]
    close = df["close"]
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()


def is_bullish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] < prev["open"] and
            curr["close"] > curr["open"] and
            curr["open"]  < prev["close"] and
            curr["close"] > prev["open"])


def is_bearish_engulfing(df: pd.DataFrame) -> bool:
    if len(df) < 2:
        return False
    prev = df.iloc[-2]
    curr = df.iloc[-1]
    return (prev["close"] > prev["open"] and
            curr["close"] < curr["open"] and
            curr["open"]  > prev["close"] and
            curr["close"] < prev["open"])


def is_hammer(df: pd.DataFrame) -> bool:
    candle     = df.iloc[-1]
    body       = abs(candle["close"] - candle["open"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    return (lower_wick >= 2 * body and upper_wick <= body * 0.5 and
            candle["close"] > candle["open"])


def is_shooting_star(df: pd.DataFrame) -> bool:
    candle     = df.iloc[-1]
    body       = abs(candle["close"] - candle["open"])
    upper_wick = candle["high"] - max(candle["open"], candle["close"])
    lower_wick = min(candle["open"], candle["close"]) - candle["low"]
    return (upper_wick >= 2 * body and lower_wick <= body * 0.5 and
            candle["close"] < candle["open"])


def get_trend(df: pd.DataFrame, fast: int = 21, slow: int = 55) -> str:
    close = df["close"]
    if len(close) < slow + 5:
        return "NEUTRAL"
    ema_f = ema(close, fast).iloc[-1]
    ema_s = ema(close, slow).iloc[-1]
    if ema_f > ema_s * 1.0003:
        return "UP"
    elif ema_f < ema_s * 0.9997:
        return "DOWN"
    return "NEUTRAL"


def rsi_divergence(df: pd.DataFrame, period: int = 14, lookback: int = 20) -> str:
    """
    Detect RSI divergence.
    FIX v6.1: Was comparing close[-1] against a slice that included close[-1].
    Now correctly compares current bar against the prior lookback window only.

    Returns: "BULLISH_DIV", "BEARISH_DIV", or "NONE"
    """
    if len(df) < lookback + period:
        return "NONE"

    # Current bar values
    current_close = df["close"].iloc[-1]
    rsi_series    = rsi(df["close"], period)
    current_rsi   = rsi_series.iloc[-1]

    # Prior window (exclude current bar)
    prior_closes = df["close"].iloc[-lookback:-1].values
    prior_rsi    = rsi_series.iloc[-lookback:-1].values

    if len(prior_closes) < 5:
        return "NONE"

    # Bullish divergence: price makes lower low, RSI makes higher low
    price_lower_low = current_close < prior_closes.min()
    rsi_higher_low  = current_rsi   > prior_rsi.min()
    if price_lower_low and rsi_higher_low:
        return "BULLISH_DIV"

    # Bearish divergence: price makes higher high, RSI makes lower high
    price_higher_high = current_close > prior_closes.max()
    rsi_lower_high    = current_rsi   < prior_rsi.max()
    if price_higher_high and rsi_lower_high:
        return "BEARISH_DIV"

    return "NONE"
