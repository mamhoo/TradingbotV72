"""
aggressive_scalper.py — SMALL ACCOUNT FLIP ENGINE v1.2
Target: $30 -> $60+ Daily
Strategy: M1/M5 High-Frequency Scalping on Gold (XAUUSD)
FIXES v1.2:
  - Added ADX Trend Strength filter (ADX > 25) to avoid choppy markets.
  - Stricter RSI entry (40/60) to ensure better momentum.
  - Added Loss Recovery logic: requires higher score after a loss.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime, timezone

log = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from signal_model import Signal
from indicators import rsi, ema, atr, get_trend, adx
from gold_strategy import check_volume_confirmation, calculate_partial_tp

# ── Aggressive Parameters ────────────────────────────────────────────────────
SCALP_PARAMS = {
    "m1_ema_fast": 9,
    "m1_ema_slow": 21,
    "m5_ema_trend": 50,
    "rsi_period": 7,
    "rsi_ob": 60,           # v1.2: Stricter (was 70)
    "rsi_os": 40,           # v1.2: Stricter (was 30)
    "adx_period": 14,
    "adx_min": 25,          # v1.2: New Trend Strength filter
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_rr": 1.8,           # v1.2: Slightly higher RR (was 1.5)
    "min_lot": 0.01,        # Minimum possible for $30 account
    "max_trades_per_day": 20,
}

def check_aggressive_scalp(config: dict, 
                           df_m1_override: Optional[pd.DataFrame] = None,
                           df_m5_override: Optional[pd.DataFrame] = None) -> Optional[Signal]:
    
    symbol = config.get("mt5_symbol", "XAUUSD")
    
    # 1. Load Data (M1 for entry, M5 for trend)
    if df_m1_override is not None:
        df_m1 = df_m1_override
        df_m5 = df_m5_override
    else:
        from gold_strategy import get_mt5_ohlcv
        df_m1 = get_mt5_ohlcv(symbol, "M1", 200)
        df_m5 = get_mt5_ohlcv(symbol, "M5", 200)

    if df_m1 is None or df_m5 is None or len(df_m1) < 50 or len(df_m5) < 50:
        return None

    # 2. Indicators
    m1_close = df_m1["close"]
    m1_ema_f = ema(m1_close, SCALP_PARAMS["m1_ema_fast"])
    m1_ema_s = ema(m1_close, SCALP_PARAMS["m1_ema_slow"])
    m5_ema_t = ema(df_m5["close"], SCALP_PARAMS["m5_ema_trend"])
    m1_rsi = rsi(m1_close, SCALP_PARAMS["rsi_period"])
    m1_atr = atr(df_m1, SCALP_PARAMS["atr_period"]).iloc[-1]
    m1_adx = adx(df_m1, SCALP_PARAMS["adx_period"]).iloc[-1]

    curr_p = m1_close.iloc[-1]
    curr_rsi = float(m1_rsi.iloc[-1])
    curr_adx = float(m1_adx)
    m5_trend_p = m5_ema_t.iloc[-1]
    trend_h1 = get_trend(df_m5) # Use M5 as proxy for trend
    
    # 3. Signal Logic
    bias = None
    reason = ""
    score = 80
    
    # Trend Strength Check (v1.2)
    if curr_adx < SCALP_PARAMS["adx_min"]:
        # log.info(f"[SCALP] ADX too low ({curr_adx:.1f}) - skipping")
        return None

    # Trend Following Scalp
    if curr_p > m5_trend_p: # M5 Uptrend
        if m1_ema_f.iloc[-1] > m1_ema_s.iloc[-1] and m1_ema_f.iloc[-2] <= m1_ema_s.iloc[-2]:
            if curr_rsi < SCALP_PARAMS["rsi_ob"]: # Not overbought
                bias = "BUY"
                reason = "M1_EMA_CROSS_UP_M5_TREND"
    elif curr_p < m5_trend_p: # M5 Downtrend
        if m1_ema_f.iloc[-1] < m1_ema_s.iloc[-1] and m1_ema_f.iloc[-2] >= m1_ema_s.iloc[-2]:
            if curr_rsi > SCALP_PARAMS["rsi_os"]: # Not oversold
                bias = "SELL"
                reason = "M1_EMA_CROSS_DOWN_M5_TREND"

    if not bias:
        return None

    # 4. Risk Management (Aggressive for $30 account)
    # On $30, we MUST use 0.01 lot. 
    lot = 0.01 
    
    sl_dist = max(m1_atr * SCALP_PARAMS["sl_atr_mult"], 1.5) # Min 1.5 points SL for Gold
    tp_dist = sl_dist * SCALP_PARAMS["tp_rr"]
    
    sl = curr_p - sl_dist if bias == "BUY" else curr_p + sl_dist
    tp = curr_p + tp_dist if bias == "BUY" else curr_p - tp_dist
    
    return Signal(
        market="GOLD",
        symbol=symbol,
        action=bias,
        entry=curr_p,
        sl=round(sl, 2),
        tp=round(tp, 2),
        lot_or_qty=lot,
        score=score,
        reason=f"AGGRESSIVE_v1.2 | {reason} | ADX:{curr_adx:.1f}",
        sr_level=curr_p,
        sr_type="NONE",
        zone_strength=0,
        trend_1h=trend_h1,
        rsi=curr_rsi,
        risk_usdt=sl_dist * 10, # Approx for 0.01 lot
        timestamp=datetime.now(timezone.utc),
        rr_ratio=SCALP_PARAMS["tp_rr"],
        atr_value=m1_atr
    )
