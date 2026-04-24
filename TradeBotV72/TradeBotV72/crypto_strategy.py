"""
crypto_strategy.py — REWRITTEN v2.0 with Balanced BUY/SELL Logic

CRITICAL FIXES:
  1. Symmetric BUY/SELL logic — identical criteria, mirrored
  2. 4H trend filter — trade WITH the trend, not against it
  3. Simplified scoring — no false precision
  4. Stricter entry — require zone + momentum alignment
  5. Reduced risk — 0.5% base risk, max 1.5% for A+ setups

STRATEGY PHILOSOPHY:
  - Trade pullbacks in established trends
  - Demand/Supply zones as entry triggers
  - RSI + MACD for momentum confirmation
  - Skip choppy/ranging markets
"""

import logging
import ccxt
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from signal_model import Signal
from sr_zones import build_zones, get_nearest_zones
from indicators import rsi, ema, macd, atr, get_trend

log = logging.getLogger(__name__)


def get_exchange(config: dict):
    """Initialize ccxt exchange."""
    exchange_id = config["exchange"]
    exchange_class = getattr(ccxt, exchange_id)
    return exchange_class({
        "apiKey":          config["api_key"],
        "secret":          config["api_secret"],
        "enableRateLimit": True,
        "options":         {"defaultType": "spot"},
    })


def get_ohlcv(exchange, symbol: str, timeframe: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """Fetch OHLCV from exchange."""
    try:
        raw = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["time"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df[["time", "open", "high", "low", "close", "volume"]]
    except Exception as e:
        log.error(f"[CRYPTO] OHLCV fetch error {symbol} {timeframe}: {e}")
        return None


def get_trend_direction(df_1h, df_4h, fast=21, slow=55) -> Tuple[str, str, bool]:
    """
    Returns (h1_trend, h4_trend, is_aligned)
    Aligned = both timeframes agree = high probability
    """
    h1_trend = get_trend(df_1h, fast, slow)
    h4_trend = get_trend(df_4h, fast, slow)
    aligned = (h1_trend == h4_trend) and h1_trend != "NEUTRAL"
    return h1_trend, h4_trend, aligned


def check_macd(df, action) -> Tuple[bool, str, float]:
    """
    Simplified MACD — only high-probability signals.
    Symmetric for BUY and SELL.
    """
    _, _, hist = macd(df["close"])
    if len(hist) < 3:
        return False, "NO_DATA", 0.0

    h0 = hist.iloc[-1]
    h1 = hist.iloc[-2]

    if action == "BUY":
        # Zero cross: negative -> positive
        if h1 <= 0 < h0:
            return True, "ZERO_CROSS_BUY", h0
        # Rising from deeply negative
        if h1 < -0.5 and h0 > h1 and h0 < 0:
            return True, "RECOVERY_BUY", h0

    else:  # SELL
        # Zero cross: positive -> negative
        if h1 >= 0 > h0:
            return True, "ZERO_CROSS_SELL", h0
        # Falling from deeply positive
        if h1 > 0.5 and h0 < h1 and h0 > 0:
            return True, "REJECTION_SELL", h0

    return False, "NO_SIGNAL", h0


def check_rsi(df, action, period=14) -> Tuple[bool, str, float]:
    """
    RSI check — symmetric for BUY/SELL.
    Buy pullbacks in uptrend, sell rallies in downtrend.
    """
    rsi_val = rsi(df["close"], period).iloc[-1]

    if action == "BUY":
        if rsi_val > 70:
            return False, "OVERBOUGHT", rsi_val
        if rsi_val < 25:
            return False, "OVERSOLD_TRAP", rsi_val
        if 35 <= rsi_val <= 55:
            return True, "PULLBACK_BUY", rsi_val
        return True, "OK", rsi_val

    else:  # SELL
        if rsi_val < 30:
            return False, "OVERSOLD", rsi_val
        if rsi_val > 75:
            return False, "OVERBOUGHT_TRAP", rsi_val
        if 45 <= rsi_val <= 65:
            return True, "PULLBACK_SELL", rsi_val
        return True, "OK", rsi_val


def check_zone(zones, current_price, action, max_dist_pct=0.008) -> Tuple[bool, Optional[object], int]:
    """Check if price is at a valid zone."""
    if not zones:
        return False, None, 0

    ns, nr, at_support, at_resistance = get_nearest_zones(zones, current_price, max_dist_pct)

    if action == "BUY":
        if at_support and ns and ns.is_fresh and ns.touches >= 2:
            return True, ns, ns.touches
    else:
        if at_resistance and nr and nr.is_fresh and nr.touches >= 2:
            return True, nr, nr.touches

    return False, None, 0


def score_signal(action: str, trend_aligned: bool, h1_trend: str,
                 rsi_ok: bool, rsi_label: str, rsi_val: float,
                 macd_ok: bool, macd_signal: str, macd_val: float,
                 at_zone: bool, touches: int) -> Tuple[int, str]:
    """Score signal 0-100. Symmetric for BUY/SELL."""
    score = 0
    reasons = []

    # Trend alignment: 35 pts
    if trend_aligned:
        score += 35
        reasons.append(f"ALIGNED({h1_trend})")
    elif h1_trend != "NEUTRAL":
        score += 20
        reasons.append(f"H1_ONLY({h1_trend})")
    else:
        score += 10
        reasons.append("TREND_WEAK")

    # MACD: 25 pts
    if macd_ok:
        if "ZERO_CROSS" in macd_signal:
            score += 25
            reasons.append("MACD_ZERO_CROSS")
        else:
            score += 15
            reasons.append(macd_signal)
    else:
        reasons.append(f"MACD_NO({macd_val:.2f})")

    # RSI: 20 pts
    if rsi_ok:
        if "PULLBACK" in rsi_label:
            score += 20
            reasons.append(f"RSI_{rsi_label}")
        else:
            score += 10
            reasons.append(f"RSI_{rsi_label}({rsi_val:.0f})")
    else:
        reasons.append(f"RSI_{rsi_label}({rsi_val:.0f})")

    # Zone: 20 pts
    if at_zone:
        zone_pts = min(20, 5 * touches)
        score += zone_pts
        reasons.append(f"ZONE_{touches}T")
    else:
        reasons.append("NO_ZONE")

    return min(100, score), " | ".join(reasons)


def calculate_position_size(account_balance: float, risk_pct: float,
                            entry: float, sl: float,
                            leverage: int = 1) -> float:
    """Calculate position size in quote currency (USDT)."""
    risk_amount = account_balance * (risk_pct / 100)
    sl_distance_pct = abs(entry - sl) / entry
    if sl_distance_pct == 0:
        return 0.0
    position_usdt = (risk_amount / sl_distance_pct) * leverage
    return min(position_usdt, account_balance * 0.25)  # max 25% of balance


def get_risk_multiplier(score: int, trend_aligned: bool) -> float:
    """Scale risk based on signal score and trend alignment."""
    if score < 50:
        return 0.0
    elif score < 65:
        return 0.5
    elif score < 80:
        return 1.0
    else:
        # High score only gets max risk if trends aligned
        return 1.5 if trend_aligned else 1.0


def check_crypto_signal(symbol: str, config: dict, exchange=None) -> Optional[Signal]:
    """Main crypto signal function. Returns Signal if valid, else None."""
    if exchange is None:
        exchange = get_exchange(config)

    df_15m = get_ohlcv(exchange, symbol, "15m", 200)
    df_1h  = get_ohlcv(exchange, symbol, "1h",  300)
    df_4h  = get_ohlcv(exchange, symbol, "4h",  200)

    if df_1h is None or df_4h is None:
        log.warning(f"[CRYPTO] {symbol} — missing data")
        return None

    current_price = df_1h["close"].iloc[-1]

    # Build zones on 1H
    zones = build_zones(df_1h, lookback=200, min_touches=2, zone_pips=current_price * 0.003)
    if not zones:
        return None

    # Trend filter — H4 + H1 alignment
    h1_trend, h4_trend, trend_aligned = get_trend_direction(
        df_1h, df_4h, config["gold_ema_fast"], config["gold_ema_slow"]
    )

    log.info(f"[CRYPTO] {symbol} | H1:{h1_trend} H4:{h4_trend} aligned:{trend_aligned}")

    # Skip if both flat
    if h1_trend == "NEUTRAL" and h4_trend == "NEUTRAL":
        log.info(f"[CRYPTO] {symbol} — both timeframes flat, skip")
        return None

    # Determine action
    if h1_trend == "NEUTRAL":
        if h4_trend != "NEUTRAL":
            action = "BUY" if h4_trend == "UP" else "SELL"
        else:
            return None
    else:
        action = "BUY" if h1_trend == "UP" else "SELL"

    # Check zone
    at_zone, zone_obj, touches = check_zone(zones, current_price, action)

    # RSI check
    rsi_ok, rsi_label, rsi_val = check_rsi(df_1h, action, config["gold_rsi_period"])
    if not rsi_ok:
        log.info(f"[CRYPTO] {symbol} — RSI veto: {rsi_label} ({rsi_val:.1f})")
        return None

    # MACD check
    macd_ok, macd_signal, macd_val = check_macd(df_1h, action)
    if not macd_ok:
        log.info(f"[CRYPTO] {symbol} — MACD no signal")
        return None

    # Score signal
    score, reason = score_signal(
        action, trend_aligned, h1_trend,
        rsi_ok, rsi_label, rsi_val,
        macd_ok, macd_signal, macd_val,
        at_zone, touches
    )

    log.info(f"[CRYPTO] {symbol} Score: {score}/100 | {reason}")

    # Minimum score threshold
    min_score = 55 if trend_aligned else 65
    if score < min_score:
        log.info(f"[CRYPTO] {symbol} score {score} < {min_score} — skip")
        return None

    # Risk multiplier
    risk_mult = get_risk_multiplier(score, trend_aligned)
    if risk_mult == 0:
        return None

    # Calculate SL/TP
    atr_val = atr(df_1h, 14).iloc[-1]

    if action == "BUY":
        sl = zone_obj.low - atr_val * 0.5 if at_zone else current_price - atr_val * 1.5
        tp = current_price + (current_price - sl) * config["crypto_rr_ratio"]
    else:
        sl = zone_obj.high + atr_val * 0.5 if at_zone else current_price + atr_val * 1.5
        tp = current_price - (sl - current_price) * config["crypto_rr_ratio"]

    # Position sizing
    base_risk_pct = config["crypto_risk_pct"]
    scaled_risk_pct = min(base_risk_pct * risk_mult, config["crypto_max_risk_pct"])

    position_usdt = calculate_position_size(
        config["crypto_account_balance"],
        scaled_risk_pct,
        current_price, sl,
        config["crypto_leverage"]
    )

    qty = position_usdt / current_price if current_price > 0 else 0
    risk_usdt = config["crypto_account_balance"] * scaled_risk_pct / 100

    log.info(f"[CRYPTO] {symbol} {action} | Score:{score} | Mult:{risk_mult}x | Risk:${risk_usdt:.2f}")

    return Signal(
        market="CRYPTO",
        symbol=symbol,
        action=action,
        entry=current_price,
        sl=sl,
        tp=tp,
        lot_or_qty=round(qty, 6),
        score=score,
        reason=f"Score:{score} | RiskMult:{risk_mult}x | Risk:{scaled_risk_pct:.1f}% | {reason}",
        sr_level=zone_obj.price if zone_obj else current_price,
        sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
        zone_strength=touches if zone_obj else 0,
        trend_1h=h1_trend,
        rsi=rsi_val,
        risk_usdt=risk_usdt,
    )


def execute_crypto_trade(signal: Signal, config: dict, exchange=None) -> bool:
    """Place order via ccxt."""
    if exchange is None:
        exchange = get_exchange(config)

    try:
        side = signal.action.lower()
        order = exchange.create_order(
            symbol=signal.symbol,
            type="market",
            side=side,
            amount=signal.lot_or_qty,
        )
        log.info(f"[CRYPTO] Order placed: {order['id']} | {signal.symbol} {side} {signal.lot_or_qty}")

        # Set stop loss
        try:
            sl_side = "sell" if side == "buy" else "buy"
            exchange.create_order(
                symbol=signal.symbol,
                type="stop_market" if config["crypto_leverage"] > 1 else "stop_loss",
                side=sl_side,
                amount=signal.lot_or_qty,
                params={"stopPrice": signal.sl}
            )
            log.info(f"[CRYPTO] SL order placed at {signal.sl}")
        except Exception as e:
            log.warning(f"[CRYPTO] SL order failed (set manually): {e}")

        return True

    except Exception as e:
        log.error(f"[CRYPTO] Order error: {e}")
        return False
