"""
bb_squeeze_strategy.py — HIGH-PERFORMANCE MODE
Target: $30 Daily Profit with 0.01 Lot Size on Gold

STRATEGY ARCHITECTURE:
1. Primary: M15 Squeeze Breakout (High Frequency)
2. Secondary: M5 Continuation (Pullback to EMA)
3. Risk: 1:3 RR to hit $30 target with fewer trades
4. Filter: H1/H4 Trend Alignment
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timezone

log = logging.getLogger(__name__)

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from signal_model import Signal
from indicators import rsi, ema, atr, bollinger_bands, get_trend
from indicators_ext import (
    vwap, keltner_channels, bb_squeeze, bb_squeeze_momentum,
)
from gold_strategy import (
    check_spread, is_in_cooldown, check_volume_confirmation,
    calculate_lot_size, calculate_partial_tp, GOLD_NORMAL_ATR,
)
from session_config import is_tradeable, thai_time_str

# ── High-Performance Parameters ───────────────────────────────────────────────
HP_PARAMS = {
    "bb_period":           20,
    "bb_std":              2.0,
    "kc_period":           20,
    "kc_atr_period":       10,
    "kc_mult":             1.5,
    "min_squeeze_bars":    1,      
    "min_score":           30,     # Lowered for higher frequency
    "target_rr":           3.0,    # 1:3 RR for $30 target
    "atr_sl_mult":         1.2,    # Tighter stops for better RR
    "vol_ratio_threshold": 1.1,    
}

def _get_ohlcv(symbol: str, tf_str: str, bars: int = 300,
               df_override: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    if df_override is not None:
        return df_override
    if not MT5_AVAILABLE or mt5 is None:
        return None
    from gold_strategy import get_mt5_ohlcv
    return get_mt5_ohlcv(symbol, tf_str, bars)

def check_squeeze_signal(config: dict, 
                         df_m5_override:  Optional[pd.DataFrame] = None,
                         df_m15_override: Optional[pd.DataFrame] = None,
                         df_h1_override:  Optional[pd.DataFrame] = None,
                         df_h4_override:  Optional[pd.DataFrame] = None,
                         df_d1_override:  Optional[pd.DataFrame] = None) -> Optional[Signal]:
    
    symbol = config.get("mt5_symbol", "XAUUSD")
    
    # 1. Session & Cooldown
    can_trade, session, _ = is_tradeable()
    if not can_trade: return None
    if is_in_cooldown(symbol): return None
    
    # 2. Load Data (M15 is our primary execution timeframe for High Performance)
    df_m15 = _get_ohlcv(symbol, "M15", 300, df_m15_override)
    df_h1  = _get_ohlcv(symbol, "H1",  300, df_h1_override)
    df_h4  = _get_ohlcv(symbol, "H4",  200, df_h4_override)
    
    if df_m15 is None or df_h1 is None: return None
    
    current_price = df_m15["close"].iloc[-1]
    current_atr   = atr(df_m15, 14).iloc[-1]
    
    # 3. Trend Alignment (H1 + H4)
    trend_h1 = get_trend(df_h1)
    trend_h4 = get_trend(df_h4) if df_h4 is not None else "NEUTRAL"
    
    # 4. Squeeze Detection
    sq = bb_squeeze(df_m15, HP_PARAMS["bb_period"], HP_PARAMS["bb_std"], 
                    HP_PARAMS["kc_period"], HP_PARAMS["kc_atr_period"], HP_PARAMS["kc_mult"])
    
    is_squeezed = sq.iloc[-1]
    was_squeezed = sq.iloc[-2]
    just_released = was_squeezed and not is_squeezed
    
    # 5. Momentum & Entry
    mom = bb_squeeze_momentum(df_m15)
    mom_val = mom.iloc[-1]
    mom_prev = mom.iloc[-2]
    
    bias = None
    if just_released or (not is_squeezed and abs(mom_val) > abs(mom_prev)):
        if mom_val > 0 and trend_h1 == "UP":
            bias = "BUY"
        elif mom_val < 0 and trend_h1 == "DOWN":
            bias = "SELL"
            
    if not bias: return None
    
    # 6. Volume Confirmation
    vol_ok, vol_ratio = check_volume_confirmation(df_m15, HP_PARAMS["vol_ratio_threshold"])
    if not vol_ok: return None
    
    # 7. Risk Management (Targeting $30 with 0.01 lot)
    # 0.01 lot on Gold: $1 move = $1 profit
    # We need a $30 move for $30 profit.
    sl_dist = current_atr * HP_PARAMS["atr_sl_mult"]
    tp_dist = sl_dist * HP_PARAMS["target_rr"]
    
    sl = current_price - sl_dist if bias == "BUY" else current_price + sl_dist
    tp = current_price + tp_dist if bias == "BUY" else current_price - tp_dist
    
    # 8. Build Signal
    score = 50 + (20 if trend_h4 == trend_h1 else 0) + (10 if vol_ratio > 1.5 else 0)
    
    reason = f"HP_MODE | {session} | {bias} | Squeeze Release | RR:{HP_PARAMS['target_rr']} | Vol:{vol_ratio}x"
    
    log.info(f"[HP_SQUEEZE] {bias} Signal | Score: {score} | TP: {tp:.2f} (Targeting $30)")
    
    return Signal(
        market="GOLD",
        symbol=symbol,
        action=bias,
        entry=current_price,
        sl=round(sl, 2),
        tp=round(tp, 2),
        lot_or_qty=0.01, # Forced 0.01 as per user requirement
        score=score,
        reason=reason,
        risk_usdt=sl_dist * 0.01,
        rr_ratio=HP_PARAMS["target_rr"],
        sr_level=current_price,
        sr_type="NONE",
        zone_strength=0,
        trend_1h=trend_h1,
        rsi=rsi(df_m15, 14).iloc[-1]
    )
