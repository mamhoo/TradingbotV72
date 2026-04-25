"""
mean_reversion_strategy.py — Mean Reversion with Volume Clusters
TradingbotV72 Strategy Module

STRATEGY CONCEPT:
  Mean Reversion trades the idea that price tends to return to its statistical mean
  after an overextension. Volume Clusters (High Volume Nodes) act as magnetic price
  levels where institutional activity has concentrated — these become support/resistance
  zones where reversion is most likely to occur.

ENTRY LOGIC:
  BUY  : Price is below the lower Bollinger Band (oversold), RSI < 40,
         price is near a High Volume Node (support cluster), and
         Z-score of price is below -1.5 (statistically cheap).
         Volume must be expanding (confirming capitulation/absorption).

  SELL : Price is above the upper Bollinger Band (overbought), RSI > 60,
         price is near a High Volume Node (resistance cluster), and
         Z-score of price is above +1.5 (statistically expensive).
         Volume must be expanding.

FILTERS:
  - Session gate (same as classic strategy)
  - Spread check
  - Cooldown after SL hit
  - D1 trend: soft gate (penalty if counter-trend, not hard block)
  - Anti-chasing: price must not be too far from VWAP

SL/TP:
  - SL: Beyond the nearest Low Volume Node (LVN) + ATR buffer
        (LVN = thin area where price can accelerate if reversion fails)
  - TP: At the Point of Control (POC) or VWAP (mean target)
  - Minimum RR: 1.3

SCORING (0-100):
  - Z-score extreme (< -2.0 or > +2.0)  : +20
  - Z-score moderate (< -1.5 or > +1.5) : +10
  - RSI extreme (< 25 or > 75)          : +20
  - RSI moderate (< 35 or > 65)         : +10
  - At HVN cluster                       : +25
  - Near HVN cluster (within 0.5%)       : +15
  - BB band touch                        : +15
  - Volume expanding                     : +10
  - D1 alignment                         : +10 / 0 / -10
  - Session bonus (London/NY overlap)    : +5
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
from datetime import datetime, timezone

log = logging.getLogger(__name__)

# ── Conditional MT5 import (graceful fallback for backtesting) ─────────────────
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
    zscore, VolumeCluster, VolumeProfileResult,
)
from gold_strategy import (
    check_spread, is_in_cooldown, check_volume_confirmation,
    calculate_lot_size, calculate_partial_tp, GOLD_NORMAL_ATR,
)
from session_config import is_tradeable, thai_time_str


# ── Strategy Parameters ───────────────────────────────────────────────────────

MR_PARAMS = {
    "bb_period":        20,
    "bb_std":           2.0,
    "rsi_period":       14,
    "zscore_period":    20,
    "vwap_period":      50,
    "vp_bins":          40,
    "vp_top_clusters":  6,
    "zscore_extreme":   2.0,
    "zscore_moderate":  1.5,
    "rsi_extreme_buy":  25,
    "rsi_moderate_buy": 35,
    "rsi_extreme_sell": 75,
    "rsi_moderate_sell":65,
    "hvn_tolerance_pct":0.005,   # 0.5% tolerance to be "at" an HVN
    "min_score":        45,
    "min_rr":           1.3,
    "atr_sl_mult":      1.5,     # SL = LVN ± atr_sl_mult * ATR
    "vwap_max_dist_pct":0.025,   # Anti-chase: max distance from VWAP (relaxed for reversion)
}


# ── Data fetch (with MT5 fallback) ────────────────────────────────────────────

def _get_ohlcv(symbol: str, tf_str: str, bars: int = 300,
               df_override: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from MT5, or return df_override for backtesting."""
    if df_override is not None:
        return df_override
    if not MT5_AVAILABLE or mt5 is None:
        return None
    from gold_strategy import get_mt5_ohlcv
    return get_mt5_ohlcv(symbol, tf_str, bars)


# ── Core signal logic ─────────────────────────────────────────────────────────

def _score_mr_signal(
    action: str,
    zs: float,
    rsi_val: float,
    at_hvn: bool,
    near_hvn: bool,
    at_band: bool,
    vol_expanding: bool,
    d1_trend: str,
    session: str,
) -> Tuple[int, List[str]]:
    """Compute signal score and reason list."""
    score = 0
    reasons = []

    # Z-score
    if action == "BUY":
        if zs <= -MR_PARAMS["zscore_extreme"]:
            score += 20
            reasons.append(f"ZSCORE_EXTREME({zs:.1f})")
        elif zs <= -MR_PARAMS["zscore_moderate"]:
            score += 10
            reasons.append(f"ZSCORE_MOD({zs:.1f})")
    else:
        if zs >= MR_PARAMS["zscore_extreme"]:
            score += 20
            reasons.append(f"ZSCORE_EXTREME({zs:.1f})")
        elif zs >= MR_PARAMS["zscore_moderate"]:
            score += 10
            reasons.append(f"ZSCORE_MOD({zs:.1f})")

    # RSI
    if action == "BUY":
        if rsi_val <= MR_PARAMS["rsi_extreme_buy"]:
            score += 20
            reasons.append(f"RSI_EXTREME({rsi_val:.0f})")
        elif rsi_val <= MR_PARAMS["rsi_moderate_buy"]:
            score += 10
            reasons.append(f"RSI_MOD({rsi_val:.0f})")
        else:
            reasons.append(f"RSI_OK({rsi_val:.0f})")
    else:
        if rsi_val >= MR_PARAMS["rsi_extreme_sell"]:
            score += 20
            reasons.append(f"RSI_EXTREME({rsi_val:.0f})")
        elif rsi_val >= MR_PARAMS["rsi_moderate_sell"]:
            score += 10
            reasons.append(f"RSI_MOD({rsi_val:.0f})")
        else:
            reasons.append(f"RSI_OK({rsi_val:.0f})")

    # Volume cluster
    if at_hvn:
        score += 25
        reasons.append("AT_HVN")
    elif near_hvn:
        score += 15
        reasons.append("NEAR_HVN")
    else:
        reasons.append("NO_HVN")

    # Bollinger Band touch
    if at_band:
        score += 15
        reasons.append("BB_TOUCH")

    # Volume expansion
    if vol_expanding:
        score += 10
        reasons.append("VOL_EXPAND")

    # D1 alignment
    if (action == "BUY" and d1_trend == "DOWN") or (action == "SELL" and d1_trend == "UP"):
        score -= 10
        reasons.append(f"D1_COUNTER(-10)")
    elif (action == "BUY" and d1_trend == "UP") or (action == "SELL" and d1_trend == "DOWN"):
        score += 10
        reasons.append(f"D1_ALIGNED(+10)")
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


def check_mr_signal(
    config: dict,
    # Backtest overrides (pass DataFrames directly to bypass MT5)
    df_m15_override: Optional[pd.DataFrame] = None,
    df_h1_override:  Optional[pd.DataFrame] = None,
    df_h4_override:  Optional[pd.DataFrame] = None,
    df_d1_override:  Optional[pd.DataFrame] = None,
) -> Optional[Signal]:
    """
    Mean Reversion with Volume Clusters signal.

    Returns a Signal if all conditions are met, else None.
    Accepts optional DataFrame overrides for backtesting without MT5.
    """
    symbol = config.get("mt5_symbol", "XAUUSD")

    # ── 1. Session gate ───────────────────────────────────────────────────────
    can_trade, session, session_params = is_tradeable()
    if not can_trade:
        return None

    log.info("[MR] %s | Session: %s", thai_time_str(), session)

    # ── 2. Spread + cooldown ──────────────────────────────────────────────────
    if MT5_AVAILABLE and df_m15_override is None:
        spread_ok, spread_pips = check_spread(symbol, config.get("gold_max_spread_pips", 80))
        if not spread_ok:
            log.info("[MR] Spread %.1f too high — skip", spread_pips)
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
        log.warning("[MR] Insufficient data — skip")
        return None

    current_price = df_m15["close"].iloc[-1]
    current_atr   = atr(df_m15, 14).iloc[-1]

    # ── 4. Volume confirmation ────────────────────────────────────────────────
    vol_ok, vol_ratio = check_volume_confirmation(
        df_m15, min_volume_ratio=config.get("gold_min_volume_ratio", 1.2)
    )
    # For mean reversion we don't hard-block on volume, but score it
    vol_expanding = vol_ok

    # ── 5. Bollinger Bands ────────────────────────────────────────────────────
    bb_upper, bb_mid, bb_lower = bollinger_bands(
        df_m15["close"],
        period=MR_PARAMS["bb_period"],
        std=MR_PARAMS["bb_std"],
    )
    bb_upper_val = bb_upper.iloc[-1]
    bb_lower_val = bb_lower.iloc[-1]
    bb_mid_val   = bb_mid.iloc[-1]

    # ── 6. RSI ────────────────────────────────────────────────────────────────
    rsi_val = rsi(df_m15["close"], MR_PARAMS["rsi_period"]).iloc[-1]

    # ── 7. Z-score ────────────────────────────────────────────────────────────
    zs = zscore(df_m15["close"], MR_PARAMS["zscore_period"]).iloc[-1]
    if pd.isna(zs):
        log.info("[MR] Z-score NaN — skip")
        return None

    # ── 8. VWAP ───────────────────────────────────────────────────────────────
    vwap_val = vwap(df_m15, anchor="rolling", period=MR_PARAMS["vwap_period"]).iloc[-1]
    if pd.isna(vwap_val):
        vwap_val = bb_mid_val

    vwap_dist_pct = abs(current_price - vwap_val) / vwap_val
    if vwap_dist_pct > MR_PARAMS["vwap_max_dist_pct"]:
        log.info("[MR] Price %.2f too far from VWAP %.2f (%.2f%%) — skip",
                 current_price, vwap_val, vwap_dist_pct * 100)
        return None

    # ── 9. Volume Profile & Clusters ─────────────────────────────────────────
    # Use H1 data for a broader volume profile
    vp = volume_profile(df_h1, bins=MR_PARAMS["vp_bins"])
    clusters = volume_clusters(
        df_h1,
        bins=MR_PARAMS["vp_bins"],
        top_n=MR_PARAMS["vp_top_clusters"],
        current_price=current_price,
    )

    # Check proximity to HVN
    at_hvn   = vp.is_at_hvn(current_price, tolerance_pct=MR_PARAMS["hvn_tolerance_pct"])
    near_hvn = vp.is_at_hvn(current_price, tolerance_pct=MR_PARAMS["hvn_tolerance_pct"] * 2)

    # ── 10. Determine action ──────────────────────────────────────────────────
    # BUY: price below lower BB, oversold RSI, negative Z-score
    buy_signal  = (current_price <= bb_lower_val and
                   rsi_val <= 45 and
                   zs <= -MR_PARAMS["zscore_moderate"])

    # SELL: price above upper BB, overbought RSI, positive Z-score
    sell_signal = (current_price >= bb_upper_val and
                   rsi_val >= 55 and
                   zs >= MR_PARAMS["zscore_moderate"])

    if not buy_signal and not sell_signal:
        log.info("[MR] No reversion setup: price=%.2f BB=[%.2f,%.2f] RSI=%.1f Z=%.2f",
                 current_price, bb_lower_val, bb_upper_val, rsi_val, zs)
        return None

    action = "BUY" if buy_signal else "SELL"
    at_band = True  # We already confirmed price is at/beyond the band

    # ── 11. D1 trend ─────────────────────────────────────────────────────────
    d1_trend = "UNKNOWN"
    if df_d1 is not None and len(df_d1) >= 60:
        d1_trend = get_trend(df_d1, fast=21, slow=55)

    # ── 12. Scoring ───────────────────────────────────────────────────────────
    score, reasons = _score_mr_signal(
        action, zs, rsi_val, at_hvn, near_hvn, at_band,
        vol_expanding, d1_trend, session,
    )

    log.info("[MR] Score: %d | %s", score, " | ".join(reasons))

    if score < MR_PARAMS["min_score"]:
        log.info("[MR] Score %d < %d — skip", score, MR_PARAMS["min_score"])
        return None

    # ── 13. SL/TP calculation ─────────────────────────────────────────────────
    # SL: beyond the nearest LVN (thin area where price can accelerate)
    # TP: at the POC (Point of Control) or VWAP — the mean target
    lvn_nearest = vp.nearest_lvn(current_price)
    poc = vp.poc

    if action == "BUY":
        # SL below the nearest LVN or below current price - ATR buffer
        if lvn_nearest and lvn_nearest < current_price:
            sl = lvn_nearest - current_atr * MR_PARAMS["atr_sl_mult"]
        else:
            sl = current_price - current_atr * MR_PARAMS["atr_sl_mult"] * 1.5

        # TP at POC or VWAP (whichever is closer above current price)
        tp_candidates = [p for p in [poc, vwap_val, bb_mid_val] if p > current_price]
        tp = min(tp_candidates) if tp_candidates else current_price + (current_price - sl) * 1.5

    else:  # SELL
        if lvn_nearest and lvn_nearest > current_price:
            sl = lvn_nearest + current_atr * MR_PARAMS["atr_sl_mult"]
        else:
            sl = current_price + current_atr * MR_PARAMS["atr_sl_mult"] * 1.5

        tp_candidates = [p for p in [poc, vwap_val, bb_mid_val] if p < current_price]
        tp = max(tp_candidates) if tp_candidates else current_price - (sl - current_price) * 1.5

    # Validate SL/TP
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

    if rr_ratio < MR_PARAMS["min_rr"]:
        log.info("[MR] RR %.2f too low — skip", rr_ratio)
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
    cluster_info = f"HVN@{vp.nearest_hvn(current_price):.1f}" if vp.hvn_levels else "NO_HVN"
    reason = (
        f"{session} | MeanReversion | Score:{score} | "
        f"BB=[{bb_lower_val:.1f},{bb_upper_val:.1f}] | "
        f"Z:{zs:.2f} | RSI:{rsi_val:.0f} | "
        f"VWAP:{vwap_val:.1f} | POC:{poc:.1f} | "
        f"{cluster_info} | RR:{rr_ratio:.1f} | "
        f"VOL:{vol_ratio:.1f}x | {' | '.join(reasons)}"
    )

    log.info(
        "[MR] SIGNAL: %s | Score:%d | Lot:%.3f | SL:%.2f | TP:%.2f | RR:%.2f",
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
        sr_level=vp.nearest_hvn(current_price) or current_price,
        sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
        zone_strength=len(vp.hvn_levels),
        trend_1h=d1_trend,
        rsi=rsi_val,
        risk_usdt=risk_usdt,
        partial_tps=partial_tps,
        trailing_stop_atr_mult=1.5,
        rr_ratio=rr_ratio,
        atr_value=current_atr,
        timestamp=datetime.now(timezone.utc),
    )
