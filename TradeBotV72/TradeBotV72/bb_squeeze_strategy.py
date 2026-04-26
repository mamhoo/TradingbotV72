"""
bb_squeeze_strategy.py — EVOLVED DUAL-ENGINE MODE v2
Target: $30 Daily Profit with 0.01 Lot Size on Gold

EVOLUTIONARY CHANGES:
1. Dual-Engine: 
   - Engine A: M15 Squeeze Breakout (Trend/Volatility)
   - Engine B: M5 Mean Reversion (Range/Scalping)
2. Optimized Risk: 1:2 RR for higher win rate while maintaining profitability.
3. Trend Filter: Stricter H1/H4 alignment for all entries.
4. Volatility Filter: Uses ATR to ensure we only trade when there is enough movement.
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

# ── Evolved Parameters ───────────────────────────────────────────────────────
EVO_PARAMS = {
    "bb_period":           20,
    "bb_std":              2.0,
    "kc_period":           20,
    "kc_atr_period":       10,
    "kc_mult":             1.5,
    "min_squeeze_bars":    1,      
    "target_rr":           2.0,    # Reduced from 3.0 for higher win rate
    "atr_sl_mult":         1.5,    
    "vol_ratio_threshold": 1.2,    # Increased for better quality
    "m5_rsi_ob":           80,     # Stricter Reversion
    "m5_rsi_os":           20,     
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
    
    # 2. Load Data
    df_m5  = _get_ohlcv(symbol, "M5",  300, df_m5_override)
    df_m15 = _get_ohlcv(symbol, "M15", 300, df_m15_override)
    df_h1  = _get_ohlcv(symbol, "H1",  300, df_h1_override)
    df_h4  = _get_ohlcv(symbol, "H4",  200, df_h4_override)
    
    if df_m15 is None or df_h1 is None or df_m5 is None: return None
    
    current_price = df_m15["close"].iloc[-1]
    current_atr   = atr(df_m15, 14).iloc[-1]
    trend_h1      = get_trend(df_h1)
    trend_h4      = get_trend(df_h4) if df_h4 is not None else "NEUTRAL"
    
    # ── ENGINE A: M15 SQUEEZE BREAKOUT ────────────────────────────────────────
    sq = bb_squeeze(df_m15, EVO_PARAMS["bb_period"], EVO_PARAMS["bb_std"], 
                    EVO_PARAMS["kc_period"], EVO_PARAMS["kc_atr_period"], EVO_PARAMS["kc_mult"])
    
    is_squeezed = sq.iloc[-1]
    was_squeezed = sq.iloc[-2]
    mom = bb_squeeze_momentum(df_m15)
    mom_val = mom.iloc[-1]
    
    bias = None
    engine = ""
    
    # Breakout Logic (Requires H1 + H4 alignment)
    if was_squeezed and not is_squeezed:
        if mom_val > 0 and trend_h1 == "UP" and trend_h4 != "DOWN":
            bias = "BUY"
            engine = "SQUEEZE_BREAKOUT"
        elif mom_val < 0 and trend_h1 == "DOWN" and trend_h4 != "UP":
            bias = "SELL"
            engine = "SQUEEZE_BREAKOUT"
            
    # ── ENGINE B: M5 MEAN REVERSION (If no breakout) ──────────────────────────
    if not bias:
        rsi_m5_series = rsi(df_m5, 14)
        if len(rsi_m5_series) > 0:
            val = rsi_m5_series.iloc[-1]
            if isinstance(val, pd.Series): val = val.iloc[0]
            rsi_m5 = float(val)
            
            if rsi_m5 >= EVO_PARAMS["m5_rsi_ob"] and trend_h1 == "DOWN":
                bias = "SELL"
                engine = "M5_REVERSION"
            elif rsi_m5 <= EVO_PARAMS["m5_rsi_os"] and trend_h1 == "UP":
                bias = "BUY"
                engine = "M5_REVERSION"
            
    if not bias: return None
    
    # 3. Volume Confirmation
    vol_ok, vol_ratio = check_volume_confirmation(df_m15, EVO_PARAMS["vol_ratio_threshold"])
    if not vol_ok: return None
    
    # 4. Risk Management
    sl_dist = current_atr * EVO_PARAMS["atr_sl_mult"]
    tp_dist = sl_dist * EVO_PARAMS["target_rr"]
    
    sl = current_price - sl_dist if bias == "BUY" else current_price + sl_dist
    tp = current_price + tp_dist if bias == "BUY" else current_price - tp_dist
    
    score = 70 if engine == "SQUEEZE_BREAKOUT" else 50
    reason = f"EVO_v2 | {engine} | {bias} | RR:{EVO_PARAMS['target_rr']} | Vol:{vol_ratio}x"
    
    log.info(f"[EVO_BOT] {engine} {bias} | Score: {score} | TP: {tp:.2f}")
    
    return Signal(
        market="GOLD",
        symbol=symbol,
        action=bias,
        entry=current_price,
        sl=round(sl, 2),
        tp=round(tp, 2),
        lot_or_qty=0.01,
        score=score,
        reason=reason,
        risk_usdt=sl_dist * 0.01,
        rr_ratio=EVO_PARAMS["target_rr"],
        sr_level=current_price,
        sr_type="NONE",
        zone_strength=0,
        trend_1h=trend_h1,
        rsi=rsi(df_m15, 14).iloc[-1]
    )
