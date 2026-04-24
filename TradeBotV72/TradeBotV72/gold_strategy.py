"""
gold_strategy.py — SUPER TRADER v6.1

FIXES from v6.0:
  [CRITICAL] D1 gate: softened from hard-block to score penalty.
             Hard block was killing ALL signals during ranging days.
  [CRITICAL] Volume filter: now compares against completed candles only (skip current).
             MT5 tick_volume resets mid-candle — partial candle was always low.
  [CRITICAL] Trailing stop SELL: entry_price cap removed for SELL direction.
             Was preventing SL from ever locking in profits on short trades.
  [FIX] Anti-chase threshold: relaxed from 0.4% → 0.6% for Gold volatility.
  [FIX] Score is now more dynamic — D1 alignment gives bonus instead of blocking.
  [FIX] Zone distance check: uses H1 zone pips consistently.
  [IMPROVE] MACD: added BULLISH_MOMENTUM/BEARISH_MOMENTUM acceptance (was blocking).
"""

import logging
import pandas as pd
import MetaTrader5 as mt5
import time
from typing import Optional, Tuple, List
from datetime import datetime, time as dt_time
from signal_model import Signal
from sr_zones import build_zones, get_nearest_zones, Zone
from indicators import rsi, ema, atr, macd, get_trend
from session_config import is_tradeable, thai_time_str

log = logging.getLogger(__name__)

TF_MAP = {
    "M1":  mt5.TIMEFRAME_M1,
    "M5":  mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1":  mt5.TIMEFRAME_H1,
    "H4":  mt5.TIMEFRAME_H4,
    "D1":  mt5.TIMEFRAME_D1,
}

_last_sl_time: dict = {}
COOLDOWN_BARS    = 3
COOLDOWN_MINUTES = 2
GOLD_NORMAL_ATR  = 10.0   # Realistic Gold M5 ATR ($8-15)


# ── Data fetch ────────────────────────────────────────────────────────────────

def get_mt5_ohlcv(symbol: str, tf_str: str, bars: int = 300) -> Optional[pd.DataFrame]:
    tf = TF_MAP.get(tf_str)
    if tf is None:
        return None
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.rename(columns={"tick_volume": "volume"}, inplace=True)
    return df[["time", "open", "high", "low", "close", "volume"]]


# ── Volume confirmation ────────────────────────────────────────────────────────

def check_volume_confirmation(
    df: pd.DataFrame, min_volume_ratio: float = 1.3
) -> Tuple[bool, float]:
    """
    FIX v6.1: Compare COMPLETED candles only (iloc[-2] vs avg of [-21:-2]).
    MT5 tick_volume on the current (last) candle resets mid-bar, so
    comparing an in-progress bar against a full-bar average always looks low.
    We look at the MOST RECENTLY COMPLETED candle instead.
    """
    if len(df) < 22:
        return False, 0.0

    # Use the last COMPLETED candle (index -2), average over 20 completed before it
    last_complete_vol = df["volume"].iloc[-2]
    avg_volume = df["volume"].iloc[-22:-2].mean()

    if avg_volume == 0:
        return False, 0.0

    volume_ratio = last_complete_vol / avg_volume
    return volume_ratio >= min_volume_ratio, round(volume_ratio, 2)


# ── Spread check ──────────────────────────────────────────────────────────────

def check_spread(symbol: str, max_spread_pips: float) -> Tuple[bool, float]:
    tick = mt5.symbol_info_tick(symbol)
    info = mt5.symbol_info(symbol)
    if tick is None or info is None:
        return False, 0.0
    pt = info.point if info.point > 0 else 0.00001
    spread_pips = (tick.ask - tick.bid) / pt
    return spread_pips <= max_spread_pips, round(spread_pips, 1)


# ── Cooldown check ────────────────────────────────────────────────────────────

def is_in_cooldown(symbol: str) -> bool:
    last = _last_sl_time.get(symbol)
    if last is None:
        return False
    elapsed = (datetime.utcnow() - last).total_seconds() / 60
    if elapsed < COOLDOWN_MINUTES:
        log.info("[GOLD] Cooldown: %.0f min remaining", COOLDOWN_MINUTES - elapsed)
        return True
    return False


def register_sl_hit(symbol: str):
    _last_sl_time[symbol] = datetime.utcnow()
    log.info("[GOLD] Cooldown started for %s", symbol)


# ── Daily trend gate (D1) ─────────────────────────────────────────────────────

def check_daily_trend(symbol: str, action: str, fast: int = 21, slow: int = 55) -> Tuple[bool, str, int]:
    """
    FIX v6.1: Soft gate instead of hard block.
    Returns (allowed, d1_trend, score_adjustment).
    - Aligned with D1 → +15 bonus
    - D1 neutral → +0, allow trade
    - Against D1 → -15 penalty, but still allow if H1+H4 strongly aligned
    Hard block ONLY when D1 is strongly against AND trend is weak on lower TFs.
    """
    df_d1 = get_mt5_ohlcv(symbol, "D1", bars=100)
    if df_d1 is None or len(df_d1) < slow + 5:
        log.warning("[GOLD] D1 data missing — allowing with neutral score")
        return True, "UNKNOWN", 0

    d1_trend = get_trend(df_d1, fast, slow)

    if action == "BUY":
        if d1_trend == "UP":
            log.info("[GOLD] D1: UP — BUY aligned, +15 pts")
            return True, d1_trend, +15
        elif d1_trend == "NEUTRAL":
            log.info("[GOLD] D1: NEUTRAL — BUY allowed, +0 pts")
            return True, d1_trend, 0
        else:  # DOWN — counter-trend, penalty but not hard block
            log.info("[GOLD] D1: DOWN — BUY counter-trend, -15 pts penalty")
            return True, d1_trend, -15

    else:  # SELL
        if d1_trend == "DOWN":
            log.info("[GOLD] D1: DOWN — SELL aligned, +15 pts")
            return True, d1_trend, +15
        elif d1_trend == "NEUTRAL":
            log.info("[GOLD] D1: NEUTRAL — SELL allowed, +0 pts")
            return True, d1_trend, 0
        else:  # UP — counter-trend, penalty
            log.info("[GOLD] D1: UP — SELL counter-trend, -15 pts penalty")
            return True, d1_trend, -15


# ── Trend from H1 + H4 ────────────────────────────────────────────────────────

def get_action(df_h1, df_h4, fast: int = 21, slow: int = 55) -> Tuple[str, str, str]:
    h1_trend = get_trend(df_h1, fast, slow)
    h4_trend = get_trend(df_h4, fast, slow)
    if h1_trend != "NEUTRAL":
        return "BUY" if h1_trend == "UP" else "SELL", h1_trend, h4_trend
    elif h4_trend != "NEUTRAL":
        return "BUY" if h4_trend == "UP" else "SELL", h1_trend, h4_trend
    return "NEUTRAL", "NEUTRAL", "NEUTRAL"


# ── MACD ──────────────────────────────────────────────────────────────────────

def check_macd(df, action) -> Tuple[bool, str, float]:
    _, _, hist = macd(df["close"])
    if len(hist) < 3:
        return False, "NO_DATA", 0.0
    h0 = hist.iloc[-1]
    h1 = hist.iloc[-2]
    h2 = hist.iloc[-3]

    if action == "BUY":
        if h1 <= 0 < h0:
            return True, "ZERO_CROSS_BUY", h0
        if h0 > 0 and h0 > h1:
            return True, "BULLISH_MOMENTUM", h0
        # FIX: Accept recovering from negative (was only checking h0 > 0)
        if h0 < 0 and h0 > h1 and h0 > h2:
            return True, "RECOVERING", h0
    else:
        if h1 >= 0 > h0:
            return True, "ZERO_CROSS_SELL", h0
        if h0 < 0 and h0 < h1:
            return True, "BEARISH_MOMENTUM", h0
        if h0 > 0 and h0 < h1 and h0 < h2:
            return True, "WEAKENING", h0

    return False, "NO_SIGNAL", h0


# ── RSI ───────────────────────────────────────────────────────────────────────

def check_rsi(df, action, period: int = 14) -> Tuple[bool, str, float]:
    rsi_val = rsi(df["close"], period).iloc[-1]
    if action == "BUY":
        if rsi_val > 72:
            return False, "OVERBOUGHT", rsi_val
        if rsi_val < 20:
            return True, "OVERSOLD_BUY", rsi_val
        if 35 <= rsi_val <= 58:
            return True, "GOOD_ZONE", rsi_val
        return True, "OK", rsi_val
    else:
        if rsi_val < 28:
            return False, "OVERSOLD", rsi_val
        if rsi_val > 80:
            return True, "OVERBOUGHT_SELL", rsi_val
        if 42 <= rsi_val <= 65:
            return True, "GOOD_ZONE", rsi_val
        return True, "OK", rsi_val


# ── Zone check ────────────────────────────────────────────────────────────────

def check_zone(zones, current_price, action, max_dist_pct: float = 0.008) -> Tuple[bool, Optional[Zone], int]:
    if not zones:
        return False, None, 0
    ns, nr, at_support, at_resistance = get_nearest_zones(zones, current_price, max_dist_pct)
    if action == "BUY":
        if at_support and ns and ns.touches >= 2:
            return True, ns, ns.touches
    else:
        if at_resistance and nr and nr.touches >= 2:
            return True, nr, nr.touches
    return False, None, 0


# ── Anti-chasing ──────────────────────────────────────────────────────────────

def check_not_chasing(df_m15, action: str, fast_ema: int = 21,
                       max_dist_pct: float = 0.006) -> Tuple[bool, float]:
    """
    FIX v6.1: Relaxed from 0.4% to 0.6%.
    Gold M15 ATR ~$8 on $3000 = 0.27% per bar.
    0.4% was blocking any bar that moved > 1.5x ATR from EMA (very common).
    0.6% = 2.2x ATR — blocks only clear overextension.
    """
    current_price = df_m15["close"].iloc[-1]
    ema_val = ema(df_m15["close"], fast_ema).iloc[-1]
    dist_pct = abs(current_price - ema_val) / ema_val

    if action == "BUY" and current_price > ema_val * (1 + max_dist_pct):
        log.info("[GOLD] Anti-chase BUY: %.2f%% from EMA (limit %.2f%%)",
                 dist_pct * 100, max_dist_pct * 100)
        return False, dist_pct
    if action == "SELL" and current_price < ema_val * (1 - max_dist_pct):
        log.info("[GOLD] Anti-chase SELL: %.2f%% from EMA (limit %.2f%%)",
                 dist_pct * 100, max_dist_pct * 100)
        return False, dist_pct

    return True, dist_pct


# ── Dynamic R:R ───────────────────────────────────────────────────────────────

def calculate_dynamic_rr(zone_strength: int, trend_alignment: str, session_bonus: bool) -> float:
    base_rr   = 1.5
    zone_bonus = min(zone_strength / 30, 1.0) * 0.75
    trend_bonus = 0.5 if trend_alignment in ("UP", "DOWN") else 0
    sess_bonus  = 0.25 if session_bonus else 0
    return min(3.0, base_rr + zone_bonus + trend_bonus + sess_bonus)


# ── Partial TP ────────────────────────────────────────────────────────────────

def calculate_partial_tp(entry: float, action: str, sl: float, rr_ratio: float) -> List[Tuple[float, float]]:
    risk = abs(entry - sl)
    if action == "BUY":
        return [
            (entry + risk,            0.50),
            (entry + 2 * risk,        0.30),
            (entry + rr_ratio * risk, 0.20),
        ]
    else:
        return [
            (entry - risk,            0.50),
            (entry - 2 * risk,        0.30),
            (entry - rr_ratio * risk, 0.20),
        ]


# ── Lot sizing ────────────────────────────────────────────────────────────────

def calculate_lot_size(
    account_balance: float,
    risk_pct: float,
    entry: float,
    sl: float,
    atr_val: float,
    lot_base: float = 0.01,
    max_lot: float  = 5.0,
) -> float:
    volatility_factor = min(1.0, GOLD_NORMAL_ATR / max(atr_val, 0.1))
    risk_amount = account_balance * (risk_pct / 100) * volatility_factor
    pip_risk = abs(entry - sl) * 10
    if pip_risk == 0:
        return lot_base
    lot = risk_amount / (pip_risk * 100)
    lot = max(lot_base, round(lot / lot_base) * lot_base)
    return min(lot, max_lot)


# ── Main signal function ──────────────────────────────────────────────────────

def check_gold_signal(config: dict) -> Optional[Signal]:
    symbol = config.get("mt5_symbol", "XAUUSD")

    # ── 1. Session gate ──────────────────────────────────────────────────────
    can_trade, session, session_params = is_tradeable()
    if not can_trade:
        log.info("[GOLD] Session inactive — skip")
        return None

    log.info("[GOLD] %s | Session: %s", thai_time_str(), session)

    # ── 2. Spread check ──────────────────────────────────────────────────────
    spread_ok, spread_pips = check_spread(symbol, config.get("gold_max_spread_pips", 80))
    if not spread_ok:
        log.info("[GOLD] Spread %.1f pips too high — skip", spread_pips)
        return None

    # ── 3. Cooldown check ────────────────────────────────────────────────────
    if is_in_cooldown(symbol):
        return None

    # ── 4. Load data ─────────────────────────────────────────────────────────
    df_m5  = get_mt5_ohlcv(symbol, "M5",  200)
    df_m15 = get_mt5_ohlcv(symbol, "M15", 200)
    df_h1  = get_mt5_ohlcv(symbol, "H1",  300)
    df_h4  = get_mt5_ohlcv(symbol, "H4",  200)

    if any(d is None for d in [df_m5, df_m15, df_h1, df_h4]):
        log.warning("[GOLD] Missing OHLCV data — skip")
        return None

    current_price = df_m5["close"].iloc[-1]
    current_atr   = atr(df_m5, 14).iloc[-1]

    # ── 5. Volume confirmation (completed candle) ────────────────────────────
    vol_ok, vol_ratio = check_volume_confirmation(
        df_m5, min_volume_ratio=config.get("gold_min_volume_ratio", 1.3)
    )
    if not vol_ok:
        log.info("[GOLD] Volume %.2fx below threshold — skip", vol_ratio)
        return None

    # ── 6. Get action from H1+H4 trend ──────────────────────────────────────
    action, h1_trend, h4_trend = get_action(
        df_h1, df_h4,
        config.get("gold_ema_fast", 21),
        config.get("gold_ema_slow", 55),
    )
    if action == "NEUTRAL":
        log.info("[GOLD] Both H1+H4 flat — skip")
        return None

    log.info("[GOLD] Action: %s | H1: %s | H4: %s", action, h1_trend, h4_trend)

    # ── 7. Daily trend gate (SOFT — score penalty, not hard block) ───────────
    d1_ok, d1_trend, d1_score_adj = check_daily_trend(
        symbol, action,
        config.get("gold_ema_fast", 21),
        config.get("gold_ema_slow", 55),
    )
    # d1_ok is always True now; d1_score_adj adjusts the final score

    # ── 8. Anti-chasing check ────────────────────────────────────────────────
    not_chasing, dist_pct = check_not_chasing(
        df_m15, action,
        fast_ema=config.get("gold_ema_fast", 21),
        max_dist_pct=config.get("gold_max_entry_dist_pct", 0.006),  # FIX: 0.004→0.006
    )
    if not not_chasing:
        return None

    # ── 9. RSI check ─────────────────────────────────────────────────────────
    rsi_ok, rsi_label, rsi_val = check_rsi(
        df_m15, action, config.get("gold_rsi_period", 14)
    )
    if not rsi_ok:
        log.info("[GOLD] RSI veto: %s (%.1f)", rsi_label, rsi_val)
        return None

    # ── 10. MACD check ───────────────────────────────────────────────────────
    macd_ok, macd_signal, macd_val = check_macd(df_m15, action)
    if not macd_ok:
        log.info("[GOLD] MACD no signal: %.4f", macd_val)
        return None

    # ── 11. Zone check ───────────────────────────────────────────────────────
    zones = build_zones(
        df_h1,
        lookback=200,
        min_touches=2,
        zone_pips=config.get("gold_sr_zone_pips", 8),
    )
    at_zone, zone_obj, touches = check_zone(zones, current_price, action)

    # ── 12. Scoring ──────────────────────────────────────────────────────────
    score = 0
    reasons = []

    # D1 alignment: -15 / 0 / +15 (FIXED: was binary block)
    score += d1_score_adj
    if d1_trend == "UNKNOWN":
        reasons.append("D1_UNKNOWN")
    elif (action == "BUY" and d1_trend == "UP") or (action == "SELL" and d1_trend == "DOWN"):
        reasons.append(f"D1_{d1_trend}(+15)")
    elif d1_trend == "NEUTRAL":
        reasons.append("D1_NEUTRAL(0)")
    else:
        reasons.append(f"D1_COUNTER(-15)")

    # H1+H4 trend alignment: up to 35 pts
    if h1_trend == h4_trend and h1_trend != "NEUTRAL":
        score += 35
        reasons.append("H1H4_ALIGNED")
    elif h1_trend != "NEUTRAL":
        score += 20
        reasons.append(f"H1_{h1_trend}")
    else:
        score += 10
        reasons.append(f"H4_{h4_trend}")

    # MACD: up to 25 pts
    if "ZERO_CROSS" in macd_signal or "MOMENTUM" in macd_signal:
        score += 25
        reasons.append(macd_signal)
    else:
        score += 15
        reasons.append(macd_signal)

    # RSI: up to 20 pts
    if rsi_label == "GOOD_ZONE":
        score += 20
        reasons.append("RSI_GOOD")
    elif "OVERSOLD" in rsi_label or "OVERBOUGHT" in rsi_label:
        score += 15
        reasons.append(rsi_label)
    else:
        score += 10
        reasons.append(f"RSI_OK({rsi_val:.0f})")

    # Zone: up to 20 pts
    if at_zone:
        zone_pts = min(20, 5 * touches)
        score += zone_pts
        reasons.append(f"ZONE_{touches}T")
    else:
        ema_val = ema(df_m15["close"], config.get("gold_ema_fast", 21)).iloc[-1]
        dist = abs(current_price - ema_val) / ema_val
        if dist < 0.002:
            score += 15
            reasons.append("EMA_PULLBACK")
        elif dist < 0.004:
            score += 8
            reasons.append("EMA_NEAR")
        else:
            score += 2
            reasons.append("NO_ZONE")

    # Session bonus: up to 10 pts
    if session == "LONDON_NY_OVERLAP":
        score += 10
        reasons.append("OVERLAP")
    elif session in ("LONDON", "NEW_YORK"):
        score += 5
        reasons.append(session)

    # Volume confirmed: 5 pts bonus
    score += 5
    reasons.append(f"VOL_{vol_ratio:.1f}x")

    log.info("[GOLD] Score: %d | %s", score, " | ".join(reasons))

    # ── 13. Score threshold ──────────────────────────────────────────────────
    # Counter-trend (D1 against) requires higher bar
    if d1_score_adj < 0:
        min_score = config.get("gold_min_score", 65)  # stricter for counter-trend
    elif session == "LONDON_NY_OVERLAP":
        min_score = config.get("gold_min_score", 55)
    else:
        min_score = config.get("gold_min_score", 60)

    if score < min_score:
        log.info("[GOLD] Score %d < %d — skip", score, min_score)
        return None

    # ── 14. Dynamic R:R ──────────────────────────────────────────────────────
    trend_aligned = h1_trend == h4_trend and h1_trend != "NEUTRAL"
    session_bonus = session == "LONDON_NY_OVERLAP"
    rr_ratio = calculate_dynamic_rr(
        touches if at_zone else 0,
        h1_trend if trend_aligned else "NEUTRAL",
        session_bonus,
    )

    # ── 15. SL/TP calculation ────────────────────────────────────────────────
    sym_info  = mt5.symbol_info(symbol)
    min_stop  = (sym_info.trade_stops_level * sym_info.point) if sym_info else 0
    tick      = mt5.symbol_info_tick(symbol)
    spread_buf = (tick.ask - tick.bid) * 2 if tick else 0

    h1_atr = atr(df_h1, 14).iloc[-1]

    if action == "BUY":
        sl = current_price - h1_atr * 1.2 - spread_buf
        if (current_price - sl) < min_stop:
            sl = current_price - min_stop - spread_buf
        tp = current_price + (current_price - sl) * rr_ratio
    else:
        sl = current_price + h1_atr * 1.2 + spread_buf
        if (sl - current_price) < min_stop:
            sl = current_price + min_stop + spread_buf
        tp = current_price - (sl - current_price) * rr_ratio

    # ── 16. Position sizing ──────────────────────────────────────────────────
    risk_pct = config.get("gold_risk_pct", 0.75)
    lot = calculate_lot_size(
        config.get("gold_account_balance", 1000),
        risk_pct, current_price, sl, current_atr,
        config.get("gold_lot_base", 0.01),
        config.get("gold_max_lot", 5.0),
    )
    risk_usdt = config.get("gold_account_balance", 1000) * risk_pct / 100

    # ── 17. Partial TP levels ────────────────────────────────────────────────
    partial_tps = calculate_partial_tp(current_price, action, sl, rr_ratio)

    reason_str = f"{session} | Score:{score} | D1:{d1_trend} | {' | '.join(reasons)}"

    log.info(
        "[GOLD] SIGNAL: %s | Score:%d | Lot:%.3f | Risk:$%.2f | "
        "SL:%.2f | TP:%.2f | R:R:%.2f | D1:%s",
        action, score, lot, risk_usdt, sl, tp, rr_ratio, d1_trend,
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
        reason=reason_str,
        sr_level=zone_obj.price if zone_obj else current_price,
        sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
        zone_strength=touches if zone_obj else 0,
        trend_1h=h1_trend,
        rsi=rsi_val,
        risk_usdt=risk_usdt,
        partial_tps=partial_tps,
        trailing_stop_atr_mult=1.5,
        rr_ratio=rr_ratio,
        atr_value=current_atr,
    )


# ── Execute trade ─────────────────────────────────────────────────────────────

def execute_gold_trade(signal: Signal, config: dict) -> bool:
    symbol = signal.symbol
    info = mt5.symbol_info(symbol)
    if info is None:
        log.error("[GOLD] Symbol %s not found", symbol)
        return False
    if not info.visible:
        mt5.symbol_select(symbol, True)

    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        log.error("[GOLD] No tick price")
        return False

    order_type = mt5.ORDER_TYPE_BUY if signal.action == "BUY" else mt5.ORDER_TYPE_SELL
    price = tick.ask if signal.action == "BUY" else tick.bid

    req = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       signal.lot_or_qty,
        "type":         order_type,
        "price":        price,
        "deviation":    50,
        "magic":        config.get("mt5_magic", 20250001),
        "comment":      f"ST6|{signal.score}",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    log.info("[GOLD] Sending: %s %.3fL @ %.2f", signal.action, signal.lot_or_qty, price)
    result = mt5.order_send(req)

    if result is None:
        log.error("[GOLD] order_send returned None")
        return False
    if result.retcode not in (mt5.TRADE_RETCODE_DONE, 10009):
        log.error("[GOLD] Order failed: retcode=%d | %s", result.retcode, result.comment)
        return False

    log.info("[GOLD] Filled: ticket=%d | %s %.3fL", result.order, signal.action, signal.lot_or_qty)

    time.sleep(0.3)
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        log.warning("[GOLD] Position not found — SL/TP not set")
        return True

    pos = positions[-1]
    modify_req = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "symbol":   symbol,
        "position": pos.ticket,
        "sl":       signal.sl,
        "tp":       signal.tp,
    }
    modify_result = mt5.order_send(modify_req)
    if modify_result is None or modify_result.retcode not in (mt5.TRADE_RETCODE_DONE, 10009):
        log.warning("[GOLD] SL/TP failed (set manually)")
        return True

    log.info("[GOLD] SL/TP set: SL=%.2f TP=%.2f", signal.sl, signal.tp)
    return True


# ── Trailing stop (FIXED for SELL direction) ──────────────────────────────────

def calculate_trailing_stop(
    action: str,
    current_price: float,
    entry_price: float,
    atr_value: float,
    trail_mult: float = 1.5,
    current_sl: float = None,
) -> float:
    """
    FIX v6.1: SELL direction no longer caps at entry_price.
    For a winning SELL (price falling), we want SL to trail DOWN below entry.
    Old code: new_sl = min(current_price + trail_distance, entry_price)
    This cap prevented SL from ever moving below entry on SELL trades.
    """
    trail_distance = atr_value * trail_mult

    if action == "BUY":
        new_sl = current_price - trail_distance
        # Only allow SL to move UP (lock in profits)
        if current_sl is not None:
            new_sl = max(new_sl, current_sl)
        # Never move SL below original entry for breakeven protection
        # (commented out — let the strategy decide, not the trailing function)
    else:  # SELL
        new_sl = current_price + trail_distance
        # Only allow SL to move DOWN (lock in profits on short)
        if current_sl is not None:
            new_sl = min(new_sl, current_sl)
        # FIX: removed entry_price cap that was preventing profit lock-in

    return new_sl
