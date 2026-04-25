"""
bb_squeeze_strategy.py — Bollinger Band Squeeze with VWAP & Volume Profile
TradingbotV72 Strategy Module

STRATEGY CONCEPT:
  The Bollinger Band Squeeze identifies periods of low volatility (consolidation)
  where the Bollinger Bands contract inside the Keltner Channels. When the squeeze
  releases (BB expands beyond KC), it signals a high-probability breakout.
  VWAP confirms the direction (price above VWAP = bullish bias) and the Volume
  Profile ensures the breakout is not immediately blocked by a High Volume Node.

ENTRY LOGIC:
  SQUEEZE CONDITION:
    - Bollinger Bands (20, 2.0) are entirely inside Keltner Channels (20, 1.5 ATR)
    - Squeeze must have been active for at least 3 bars (confirmed consolidation)

  BREAKOUT CONDITION:
    - Squeeze has just released (previous bar was squeezed, current bar is not)
    - OR price has broken decisively outside the BB (> 0.5 ATR from the band)

  BUY  : Breakout is upward (close > BB upper), price > VWAP, momentum is positive,
         and the nearest HVN above current price is at least 0.5% away (clear path).

  SELL : Breakout is downward (close < BB lower), price < VWAP, momentum is negative,
         and the nearest HVN below current price is at least 0.5% away (clear path).

FILTERS:
  - Session gate
  - Spread check
  - Cooldown after SL hit
  - Squeeze duration: minimum 3 bars of squeeze before entry
  - Volume must expand on breakout (confirming genuine move, not fake-out)
  - D1 trend: soft gate (score penalty if counter-trend)

SL/TP:
  - SL: Opposite side of the Keltner Channel midline (EMA) + ATR buffer
        (if squeeze fails, price returns to the midline)
  - TP: Nearest High Volume Node in the direction of the breakout
        (HVN acts as a magnet and then resistance/support)
  - Minimum RR: 1.5

SCORING (0-100):
  - Squeeze duration (bars): +5 per bar, max +25
  - Momentum direction match : +20
  - VWAP alignment           : +15
  - Clear path (no HVN block): +15
  - Volume expansion         : +15
  - D1 alignment             : +10 / 0 / -10
  - Session bonus            : +5
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
    "min_squeeze_bars":    3,     # Minimum bars of squeeze before entry
    "max_squeeze_bars":    30,    # Ignore if squeeze lasted too long (stale)
    "hvn_clear_path_pct":  0.005, # HVN must be at least 0.5% away for clear path
    "min_score":           55,
    "min_rr":              1.5,
    "atr_sl_mult":         1.2,
    "vol_ratio_threshold": 1.3,
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
    """
    Analyze the squeeze state of the most recent bars.

    Returns
    -------
    (is_currently_squeezed, just_released, squeeze_duration_bars)
    """
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

    # Just released: previous bar was in squeeze, current bar is not
    just_released = was_squeezed_prev and not is_squeezed_now

    # Count consecutive squeeze bars ending at the previous bar
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
    momentum_positive: bool,
    above_vwap: bool,
    clear_path: bool,
    vol_expanding: bool,
    d1_trend: str,
    session: str,
) -> Tuple[int, List[str]]:
    """Compute signal score and reason list."""
    score = 0
    reasons = []

    # Squeeze duration (more bars = stronger compression = bigger breakout)
    sq_pts = min(25, squeeze_duration * 5)
    score += sq_pts
    reasons.append(f"SQ_{squeeze_duration}bars(+{sq_pts})")

    # Momentum direction
    momentum_match = (action == "BUY" and momentum_positive) or \
                     (action == "SELL" and not momentum_positive)
    if momentum_match:
        score += 20
        reasons.append("MOMENTUM_MATCH(+20)")
    else:
        reasons.append("MOMENTUM_MISMATCH")

    # VWAP alignment
    vwap_aligned = (action == "BUY" and above_vwap) or \
                   (action == "SELL" and not above_vwap)
    if vwap_aligned:
        score += 15
        reasons.append("VWAP_ALIGNED(+15)")
    else:
        score -= 5
        reasons.append("VWAP_AGAINST(-5)")

    # Clear path (no HVN blocking)
    if clear_path:
        score += 15
        reasons.append("CLEAR_PATH(+15)")
    else:
        score -= 10
        reasons.append("HVN_BLOCK(-10)")

    # Volume expansion
    if vol_expanding:
        score += 15
        reasons.append("VOL_EXPAND(+15)")
    else:
        reasons.append("VOL_WEAK")

    # D1 alignment
    if (action == "BUY" and d1_trend == "UP") or (action == "SELL" and d1_trend == "DOWN"):
        score += 10
        reasons.append(f"D1_ALIGNED(+10)")
    elif (action == "BUY" and d1_trend == "DOWN") or (action == "SELL" and d1_trend == "UP"):
        score -= 10
        reasons.append(f"D1_COUNTER(-10)")
    else:
        reasons.append(f"D1_{d1_trend}")

    # Session bonus
    if session == "LONDON_NY_OVERLAP":
        score += 5
        reasons.append("OVERLAP(+5)")
    elif session in ("LONDON", "NEW_YORK"):
        score += 3
        reasons.append(f"{session}(+3)")

    return score, reasons


# ── Main signal function ──────────────────────────────────────────────────────

def check_squeeze_signal(
    config: dict,
    # Backtest overrides
    df_m15_override: Optional[pd.DataFrame] = None,
    df_h1_override:  Optional[pd.DataFrame] = None,
    df_h4_override:  Optional[pd.DataFrame] = None,
    df_d1_override:  Optional[pd.DataFrame] = None,
) -> Optional[Signal]:
    """
    Bollinger Band Squeeze with VWAP & Volume Profile signal.

    Returns a Signal if all conditions are met, else None.
    Accepts optional DataFrame overrides for backtesting without MT5.
    """
    symbol = config.get("mt5_symbol", "XAUUSD")

    # ── 1. Session gate ───────────────────────────────────────────────────────
    can_trade, session, session_params = is_tradeable()
    if not can_trade:
        return None

    log.info("[SQ] %s | Session: %s", thai_time_str(), session)

    # ── 2. Spread + cooldown ──────────────────────────────────────────────────
    if MT5_AVAILABLE and df_m15_override is None:
        spread_ok, spread_pips = check_spread(symbol, config.get("gold_max_spread_pips", 80))
        if not spread_ok:
            log.info("[SQ] Spread %.1f too high — skip", spread_pips)
            return None
        if is_in_cooldown(symbol):
            return None

    # ── 3. Load data ──────────────────────────────────────────────────────────
    df_m15 = _get_ohlcv(symbol, "M15", 300, df_m15_override)
    df_h1  = _get_ohlcv(symbol, "H1",  300, df_h1_override)
    df_h4  = _get_ohlcv(symbol, "H4",  200, df_h4_override)
    df_d1  = _get_ohlcv(symbol, "D1",  100, df_d1_override)

    if any(d is None or (isinstance(d, pd.DataFrame) and len(d) < 50)
           for d in [df_m15, df_h1, df_h4]):
        log.warning("[SQ] Insufficient data — skip")
        return None

    current_price = df_m15["close"].iloc[-1]
    current_atr   = atr(df_m15, 14).iloc[-1]

    # ── 4. Squeeze analysis ───────────────────────────────────────────────────
    is_squeezed, just_released, squeeze_duration = analyze_squeeze(df_m15)

    # We need a squeeze release (breakout) to enter
    if not just_released:
        log.info("[SQ] No squeeze release — is_squeezed=%s, duration=%d",
                 is_squeezed, squeeze_duration)
        return None

    # Squeeze must have been active long enough
    if squeeze_duration < SQ_PARAMS["min_squeeze_bars"]:
        log.info("[SQ] Squeeze too short (%d bars < %d) — skip",
                 squeeze_duration, SQ_PARAMS["min_squeeze_bars"])
        return None

    if squeeze_duration > SQ_PARAMS["max_squeeze_bars"]:
        log.info("[SQ] Squeeze too long (%d bars > %d) — stale, skip",
                 squeeze_duration, SQ_PARAMS["max_squeeze_bars"])
        return None

    log.info("[SQ] Squeeze released after %d bars", squeeze_duration)

    # ── 5. Bollinger Bands & Keltner Channels ─────────────────────────────────
    bb_upper_s, bb_mid_s, bb_lower_s = bollinger_bands(
        df_m15["close"], SQ_PARAMS["bb_period"], SQ_PARAMS["bb_std"]
    )
    kc_upper_s, kc_mid_s, kc_lower_s = keltner_channels(
        df_m15, SQ_PARAMS["kc_period"], SQ_PARAMS["kc_atr_period"], SQ_PARAMS["kc_mult"]
    )

    bb_upper_val = bb_upper_s.iloc[-1]
    bb_lower_val = bb_lower_s.iloc[-1]
    kc_mid_val   = kc_mid_s.iloc[-1]

    # ── 6. Determine breakout direction ───────────────────────────────────────
    # Breakout direction from squeeze
    prev_close = df_m15["close"].iloc[-2]
    curr_close = df_m15["close"].iloc[-1]

    if curr_close > bb_upper_val:
        action = "BUY"
    elif curr_close < bb_lower_val:
        action = "SELL"
    else:
        # Price is still inside the bands — check momentum for direction
        momentum = bb_squeeze_momentum(df_m15, SQ_PARAMS["momentum_period"])
        mom_val = momentum.iloc[-1]
        if pd.isna(mom_val):
            log.info("[SQ] No clear breakout direction — skip")
            return None
        action = "BUY" if mom_val > 0 else "SELL"
        log.info("[SQ] Direction from momentum: %s (%.4f)", action, mom_val)

    # ── 7. Momentum confirmation ──────────────────────────────────────────────
    momentum = bb_squeeze_momentum(df_m15, SQ_PARAMS["momentum_period"])
    mom_val = momentum.iloc[-1]
    momentum_positive = (not pd.isna(mom_val)) and (mom_val > 0)

    # ── 8. VWAP ───────────────────────────────────────────────────────────────
    vwap_val = vwap(df_m15, anchor="rolling", period=SQ_PARAMS["vwap_period"]).iloc[-1]
    if pd.isna(vwap_val):
        vwap_val = kc_mid_val
    above_vwap = current_price > vwap_val

    # ── 9. Volume confirmation ────────────────────────────────────────────────
    vol_ok, vol_ratio = check_volume_confirmation(
        df_m15, min_volume_ratio=SQ_PARAMS["vol_ratio_threshold"]
    )
    vol_expanding = vol_ok

    # ── 10. Volume Profile — check for clear path ─────────────────────────────
    vp = volume_profile(df_h1, bins=SQ_PARAMS["vp_bins"])
    poc = vp.poc

    # Check if there's an HVN blocking the breakout direction
    clear_path = True
    if action == "BUY":
        # Check if nearest HVN above current price is far enough
        hvn_above = [h for h in vp.hvn_levels if h > current_price]
        if hvn_above:
            nearest_hvn_above = min(hvn_above)
            dist_to_hvn = (nearest_hvn_above - current_price) / current_price
            if dist_to_hvn < SQ_PARAMS["hvn_clear_path_pct"]:
                clear_path = False
                log.info("[SQ] HVN blocking BUY at %.2f (%.2f%% away)",
                         nearest_hvn_above, dist_to_hvn * 100)
    else:
        hvn_below = [h for h in vp.hvn_levels if h < current_price]
        if hvn_below:
            nearest_hvn_below = max(hvn_below)
            dist_to_hvn = (current_price - nearest_hvn_below) / current_price
            if dist_to_hvn < SQ_PARAMS["hvn_clear_path_pct"]:
                clear_path = False
                log.info("[SQ] HVN blocking SELL at %.2f (%.2f%% away)",
                         nearest_hvn_below, dist_to_hvn * 100)

    # ── 11. D1 trend ─────────────────────────────────────────────────────────
    d1_trend = "UNKNOWN"
    if df_d1 is not None and len(df_d1) >= 60:
        d1_trend = get_trend(df_d1, fast=21, slow=55)

    # ── 12. Scoring ───────────────────────────────────────────────────────────
    score, reasons = _score_squeeze_signal(
        action, squeeze_duration, momentum_positive,
        above_vwap, clear_path, vol_expanding, d1_trend, session,
    )

    log.info("[SQ] Score: %d | %s", score, " | ".join(reasons))

    if score < SQ_PARAMS["min_score"]:
        log.info("[SQ] Score %d < %d — skip", score, SQ_PARAMS["min_score"])
        return None

    # ── 13. SL/TP calculation ─────────────────────────────────────────────────
    # SL: at the KC midline (EMA) with ATR buffer — if squeeze fails, price returns to mean
    if action == "BUY":
        sl = kc_mid_val - current_atr * SQ_PARAMS["atr_sl_mult"]

        # TP: nearest HVN above current price (first magnet)
        hvn_above = [h for h in vp.hvn_levels if h > current_price * 1.001]
        if hvn_above:
            tp = min(hvn_above)
        else:
            # Fallback: 2x risk
            risk_pts = current_price - sl
            tp = current_price + risk_pts * 2.0

    else:  # SELL
        sl = kc_mid_val + current_atr * SQ_PARAMS["atr_sl_mult"]

        hvn_below = [h for h in vp.hvn_levels if h < current_price * 0.999]
        if hvn_below:
            tp = max(hvn_below)
        else:
            risk_pts = sl - current_price
            tp = current_price - risk_pts * 2.0

    # Validate SL/TP with MT5 constraints
    if MT5_AVAILABLE and df_m15_override is None:
        sym_info = mt5.symbol_info(symbol)
        if sym_info:
            min_stop = sym_info.trade_stops_level * sym_info.point
            tick = mt5.symbol_info_tick(symbol)
            spread_buf = (tick.ask - tick.bid) * 2 if tick else 0
            if action == "BUY":
                sl = min(sl, current_price - min_stop - spread_buf)
            else:
                sl = max(sl, current_price + min_stop + spread_buf)

    risk_pts   = abs(current_price - sl)
    reward_pts = abs(tp - current_price)
    rr_ratio   = reward_pts / risk_pts if risk_pts > 0 else 0

    if rr_ratio < SQ_PARAMS["min_rr"]:
        log.info("[SQ] RR %.2f too low — skip", rr_ratio)
        return None

    # ── 14. Position sizing ───────────────────────────────────────────────────
    risk_pct = config.get("gold_risk_pct", 0.5)
    lot = calculate_lot_size(
        config.get("gold_account_balance", 1000),
        risk_pct,
        current_price, sl, current_atr,
        config.get("gold_lot_base", 0.01),
        config.get("gold_max_lot", 5.0),
    )
    risk_usdt   = config.get("gold_account_balance", 1000) * risk_pct / 100
    partial_tps = calculate_partial_tp(current_price, action, sl, rr_ratio)

    # ── 15. Build reason string ───────────────────────────────────────────────
    reason = (
        f"{session} | BBSqueeze | Score:{score} | "
        f"SQ_Duration:{squeeze_duration}bars | "
        f"BB=[{bb_lower_val:.1f},{bb_upper_val:.1f}] | "
        f"VWAP:{vwap_val:.1f} | POC:{poc:.1f} | "
        f"Momentum:{'POS' if momentum_positive else 'NEG'} | "
        f"ClearPath:{clear_path} | "
        f"RR:{rr_ratio:.1f} | VOL:{vol_ratio:.1f}x | "
        f"{' | '.join(reasons)}"
    )

    log.info(
        "[SQ] SIGNAL: %s | Score:%d | Lot:%.3f | SL:%.2f | TP:%.2f | RR:%.2f",
        action, score, lot, sl, tp, rr_ratio,
    )

    return Signal(
        market="GOLD",
        symbol=symbol,
        action=action,
        entry=current_price,
        sl=round(sl, 2),
        tp=round(tp, 2),
        lot_or_qty=lot,
        score=score,
        reason=reason,
        sr_level=poc,
        sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
        zone_strength=squeeze_duration,
        trend_1h=d1_trend,
        rsi=rsi(df_m15["close"], 14).iloc[-1],
        risk_usdt=risk_usdt,
        partial_tps=partial_tps,
        trailing_stop_atr_mult=1.5,
        rr_ratio=rr_ratio,
        atr_value=current_atr,
        timestamp=datetime.now(timezone.utc),
    )
