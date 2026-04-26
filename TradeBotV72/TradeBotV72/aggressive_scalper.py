"""
aggressive_scalper.py — SMALL ACCOUNT FLIP ENGINE v1.0
Target: $30 -> $60+ Daily
Strategy: M1/M5 High-Frequency Scalping on Gold (XAUUSD)
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
from indicators import rsi, ema, atr, bollinger_bands
from gold_strategy import check_volume_confirmation, calculate_partial_tp

# ── Aggressive Parameters ────────────────────────────────────────────────────
SCALP_PARAMS = {
    "m1_ema_fast": 9,
    "m1_ema_slow": 21,
    "m5_ema_trend": 50,
    "rsi_period": 7,
    "rsi_ob": 70,
    "rsi_os": 30,
    "atr_period": 14,
    "sl_atr_mult": 1.5,
    "tp_rr": 1.5,           # Quick wins
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

    if df_m1 is None or df_m5 is None or len(df_m1) < 50:
        return None

    # 2. Indicators
    m1_close = df_m1["close"]
    m1_ema_f = ema(m1_close, SCALP_PARAMS["m1_ema_fast"])
    m1_ema_s = ema(m1_close, SCALP_PARAMS["m1_ema_slow"])
    m5_ema_t = ema(df_m5["close"], SCALP_PARAMS["m5_ema_trend"])
    m1_rsi = rsi(m1_close, SCALP_PARAMS["rsi_period"])
    m1_atr = atr(df_m1, SCALP_PARAMS["atr_period"]).iloc[-1]

    curr_p = m1_close.iloc[-1]
    curr_rsi = m1_rsi.iloc[-1]
    m5_trend_p = m5_ema_t.iloc[-1]
    
    # 3. Signal Logic
    bias = None
    reason = ""
    
    # Trend Following Scalp
    if curr_p > m5_trend_p: # M5 Uptrend
        if m1_ema_f.iloc[-1] > m1_ema_s.iloc[-1] and m1_ema_f.iloc[-2] <= m1_ema_s.iloc[-2]:
            if curr_rsi < 65: # Not overbought yet
                bias = "BUY"
                reason = "M1_EMA_CROSS_UP_M5_TREND"
    elif curr_p < m5_trend_p: # M5 Downtrend
        if m1_ema_f.iloc[-1] < m1_ema_s.iloc[-1] and m1_ema_f.iloc[-2] >= m1_ema_s.iloc[-2]:
            if curr_rsi > 35: # Not oversold yet
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
        market="GOLD", symbol=symbol, action=bias,
        entry=curr_p, sl=round(sl, 2), tp=round(tp, 2),
        lot_or_qty=lot, score=80,
        reason=f"AGGRESSIVE | {reason}",
        risk_usdt=sl_dist * 10, # Approx for 0.01 lot
        rr_ratio=SCALP_PARAMS["tp_rr"],
        timestamp=datetime.now(timezone.utc)
    )
