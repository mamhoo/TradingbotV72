"""
bb_squeeze_strategy.py — High-Profit Bollinger Band Squeeze with VWAP & Volume Profile
TradingbotV72 Strategy Module

OPTIMIZATIONS FOR MAXIMUM P&L:
  - High Reward Target: 2.0:1 RR to capture explosive moves.
  - Balanced Stops: 1.5x ATR to avoid noise while maintaining high R:R.
  - Squeeze Quality: Require 1 bar of squeeze for rapid response.
  - Momentum Focus: Prioritizes squeeze duration and momentum slope.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Conditional MT5 import ─────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

from signal_model import Signal
from indicators import rsi, ema, atr, bollinger_bands, get_trend
from indicators_ext import (
    vwap, volume_profile, volume_clusters,
    keltner_channels, bb_squeeze, bb_squeeze_momentum,
    VolumeProfileResult,
)
from gold_strategy import (
    check_spread, is_in_cooldown, check_volume_confirmation,
    calculate_lot_size, calculate_partial_tp, GOLD_NORMAL_ATR,
)
from session_config import is_tradeable, thai_time_str


# ── Strategy Parameters ───────────────────────────────────────────────────────

SQ_PARAMS = {
    "bb_period":           20,
    "bb_std":              2.0,
    "kc_period":           20,
    "kc_atr_period":       10,
    "kc_mult":             1.5,
    "momentum_period":     20,
    "vwap_period":         50,
    "vp_bins":             40,
    "min_squeeze_bars":    1,      # Rapid response
    "max_squeeze_bars":    50,    
    "hvn_clear_path_pct":  0.002, 
    "min_score":           40,     
    "min_rr":              1.5,    
    "atr_sl_mult":         1.5,    
    "vol_ratio_threshold": 1.0,    
}


# ── Data fetch ────────────────────────────────────────────────────────────────

def _get_ohlcv(symbol: str, tf_str: str, bars: int = 300,
               df_override: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    if df_override is not None:
        return df_override
    if not MT5_AVAILABLE or mt5 is None:
        return None
    from gold_strategy import get_mt5_ohlcv
    return get_mt5_ohlcv(symbol, tf_str, bars)


# ── Squeeze analysis ──────────────────────────────────────────────────────────

def analyze_squeeze(df: pd.DataFrame) -> Tuple[bool, bool, int]:
    squeeze_series = bb_squeeze(
        df,
        bb_period=SQ_PARAMS["bb_period"],
        bb_std=SQ_PARAMS["bb_std"],
        kc_period=SQ_PARAMS["kc_period"],
        kc_atr_period=SQ_PARAMS["kc_atr_period"],
        kc_mult=SQ_PARAMS["kc_mult"],
    )

    if len(squeeze_series) < 5:
        return False, False, 0

    is_squeezed_now  = bool(squeeze_series.iloc[-1])
    was_squeezed_prev = bool(squeeze_series.iloc[-2])
    just_released = was_squeezed_prev and not is_squeezed_now

    squeeze_duration = 0
    for i in range(len(squeeze_series) - 2, -1, -1):
        if squeeze_series.iloc[i]:
            squeeze_duration += 1
        else:
            break

    return is_squeezed_now, just_released, squeeze_duration


# ── Scoring ───────────────────────────────────────────────────────────────────

def _score_squeeze_signal(
    action: str,
    squeeze_duration: int,
    momentum_val: float,
    above_vwap: bool,
    clear_path: bool,
    vol_ratio: float,
    h1_trend: str,
    session: str,
) -> Tuple[int, List[str]]:
    score = 0
    reasons = []

    # Squeeze duration
    sq_pts = min(20, squeeze_duration * 5)
    score += sq_pts
    reasons.append(f"SQ_{squeeze_duration}b(+{sq_pts})")

    # Momentum strength
    if (action == "BUY" and momentum_val > 0) or (action == "SELL" and momentum_val < 0):
        score += 20
        reasons.append("MOM_ALIGNED(+20)")
    
    # VWAP alignment
    vwap_aligned = (action == "BUY" and above_vwap) or (action == "SELL" and not above_vwap)
    if vwap_aligned:
        score += 15
        reasons.append("VWAP_ALIGNED(+15)")

    # Clear path
    if clear_path:
        score += 10
        reasons.append("CLEAR_PATH(+10)")

    # Volume
    if vol_ratio > 1.2:
        score += 10
        reasons.append("VOL_BOOST(+10)")

    # Trend
    if (action == "BUY" and h1_trend == "UP") or (action == "SELL" and h1_trend == "DOWN"):
        score += 15
        reasons.append("TREND_ALIGNED(+15)")

    return score, reasons


# ── Main signal function ──────────────────────────────────────────────────────

def check_squeeze_signal(
    config: dict,
    df_m5_override:  Optional[pd.DataFrame] = None,
    df_m15_override: Optional[pd.DataFrame] = None,
    df_h1_override:  Optional[pd.DataFrame] = None,
    df_h4_override:  Optional[pd.DataFrame] = None,
    df_d1_override:  Optional[pd.DataFrame] = None,
) -> Optional[Signal]:
    symbol = config.get("mt5_symbol", "XAUUSD")

    # 1. Session gate
    can_trade, session, session_params = is_tradeable()
    if not can_trade:
        return None

    # 2. Spread + cooldown
    if MT5_AVAILABLE and df_m15_override is None:
        spread_ok, spread_pips = check_spread(symbol, config.get("gold_max_spread_pips", 80))
        if not spread_ok or is_in_cooldown(symbol):
            return None

    # 3. Load data
    df_m15 = _get_ohlcv(symbol, "M15", 300, df_m15_override)
    df_h1  = _get_ohlcv(symbol, "H1",  300, df_h1_override)

    if any(d is None or len(d) < 50 for d in [df_m15, df_h1]):
        return None

    current_price = df_m15["close"].iloc[-1]
    current_atr   = atr(df_m15, 14).iloc[-1]

    # 4. Squeeze analysis
    is_squeezed, just_released, squeeze_duration = analyze_squeeze(df_m15)
    
    # Check for breakout
    bb_upper_s, _, bb_lower_s = bollinger_bands(df_m15["close"], SQ_PARAMS["bb_period"], SQ_PARAMS["bb_std"])
    bb_upper_val = bb_upper_s.iloc[-1]
    bb_lower_val = bb_lower_s.iloc[-1]
    
    breakout_up = current_price > bb_upper_val
    breakout_down = current_price < bb_lower_val

    if not (just_released or (is_squeezed and (breakout_up or breakout_down))):
        return None

    if squeeze_duration < SQ_PARAMS["min_squeeze_bars"]:
        return None

    # 5. Momentum
    momentum = bb_squeeze_momentum(df_m15, SQ_PARAMS["momentum_period"])
    mom_val = momentum.iloc[-1]
    if pd.isna(mom_val): return None
    
    action = "BUY" if mom_val > 0 else "SELL"
    
    # 6. Filters
    vwap_val = vwap(df_m15, anchor="rolling", period=SQ_PARAMS["vwap_period"]).iloc[-1]
    above_vwap = current_price > vwap_val
    
    vol_ok, vol_ratio = check_volume_confirmation(df_m15, min_volume_ratio=SQ_PARAMS["vol_ratio_threshold"])
    
    vp = volume_profile(df_h1, bins=SQ_PARAMS["vp_bins"])
    clear_path = True
    if action == "BUY":
        hvn_above = [h for h in vp.hvn_levels if h > current_price]
        if hvn_above and (min(hvn_above) - current_price) / current_price < SQ_PARAMS["hvn_clear_path_pct"]:
            clear_path = False
    else:
        hvn_below = [h for h in vp.hvn_levels if h < current_price]
        if hvn_below and (current_price - max(hvn_below)) / current_price < SQ_PARAMS["hvn_clear_path_pct"]:
            clear_path = False

    h1_trend = get_trend(df_h1, 21, 55)

    # 7. Scoring
    score, reasons = _score_squeeze_signal(
        action, squeeze_duration, mom_val, above_vwap, clear_path, vol_ratio, h1_trend, session
    )

    if score < SQ_PARAMS["min_score"]:
        return None

    # 8. SL/TP
    if action == "BUY":
        sl = current_price - current_atr * SQ_PARAMS["atr_sl_mult"]
        tp = current_price + (current_price - sl) * 2.0
    else:
        sl = current_price + current_atr * SQ_PARAMS["atr_sl_mult"]
        tp = current_price - (sl - current_price) * 2.0

    risk_pts = abs(current_price - sl)
    rr_ratio = abs(tp - current_price) / risk_pts if risk_pts > 0 else 0
    if rr_ratio < SQ_PARAMS["min_rr"]: return None

    # 9. Position sizing
    risk_pct = config.get("gold_risk_pct", 0.75)
    lot = calculate_lot_size(config.get("gold_account_balance", 1000), risk_pct, current_price, sl, current_atr)
    
    return Signal(
        market="GOLD", symbol=symbol, action=action, entry=current_price,
        sl=round(sl, 2), tp=round(tp, 2), lot_or_qty=lot, score=score,
        reason=f"BBSqueeze_Profit | Score:{score} | {' | '.join(reasons)}",
        sr_level=current_price, sr_type="NONE", zone_strength=0,
        trend_1h=h1_trend, rsi=0.0, risk_usdt=risk_pct * config.get("gold_account_balance", 1000) / 100,
        partial_tps=calculate_partial_tp(current_price, action, sl, rr_ratio),
        trailing_stop_atr_mult=1.5, rr_ratio=rr_ratio, atr_value=current_atr
    )
