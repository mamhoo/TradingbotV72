"""
smc_gold_strategy.py — Gold Strategy using Smart Money Concepts (SMC) v7.0

HOW IT FITS INTO THE BOT:
  In main.py you can swap check_gold_signal() for check_gold_signal_smc()
  or run them in parallel and take whichever scores higher.

SMC ENTRY LOGIC (top-down):
  HTF (H4/D1) : Market structure → are we bullish or bearish?
  MTF (H1)    : Order Block + FVG identification
  LTF (M15)   : Entry confirmation — BOS or displacement candle on M15

ENTRY CONDITIONS (BUY example):
  1. H4 structure is BULLISH (or CHoCH_UP just happened)
  2. H1 price is in DISCOUNT zone (below 50% of H1 range)
  3. H1 price is AT or NEAR a fresh Demand Order Block
  4. H1 has an unfilled Bull FVG below (confluence)
  5. Sell-side liquidity was recently swept (smart money grabbed stops)
  6. M15 shows BOS_UP or displacement candle → LTF confirmation
  7. Combined SMC score >= 65

SL placement: Below the Demand OB low (with ATR buffer)
TP placement: At the nearest BSL (buy-side liquidity) or next swing high
"""

import logging
import pandas as pd
import MetaTrader5 as mt5
from typing import Optional, Tuple
from datetime import datetime, timezone

from signal_model import Signal
from smc_concepts import (
    build_smc_context, score_smc_signal,
    get_premium_discount, check_entry_confirmation,
    analyze_market_structure,
)
from gold_strategy import (
    get_mt5_ohlcv, check_spread, is_in_cooldown,
    check_volume_confirmation, calculate_lot_size,
    calculate_partial_tp, register_sl_hit,
    GOLD_NORMAL_ATR,
)
from indicators import rsi, atr, ema
from session_config import is_tradeable, thai_time_str

log = logging.getLogger(__name__)


def check_gold_signal_smc(config: dict) -> Optional[Signal]:
    """
    SMC-based Gold signal.
    Returns a Signal if all conditions met, else None.
    """
    symbol = config.get("mt5_symbol", "XAUUSD")

    # ── 1. Session gate ───────────────────────────────────────────────────
    can_trade, session, session_params = is_tradeable()
    if not can_trade:
        return None

    log.info("[SMC] %s | Session: %s", thai_time_str(), session)

    # ── 2. Spread + cooldown ──────────────────────────────────────────────
    spread_ok, spread_pips = check_spread(symbol, config.get("gold_max_spread_pips", 80))
    if not spread_ok:
        log.info("[SMC] Spread %.1f too high — skip", spread_pips)
        return None

    if is_in_cooldown(symbol):
        return None

    # ── 3. Load data ──────────────────────────────────────────────────────
    df_m15 = get_mt5_ohlcv(symbol, "M15", 100)
    df_h1  = get_mt5_ohlcv(symbol, "H1",  200)
    df_h4  = get_mt5_ohlcv(symbol, "H4",  150)
    df_d1  = get_mt5_ohlcv(symbol, "D1",  60)

    # [FIX] Robust check for None or empty DataFrames
    if any(d is None or (isinstance(d, pd.DataFrame) and d.empty) for d in [df_m15, df_h1, df_h4]):
        log.warning("[SMC] Missing or empty data — skip")
        return None

    current_price = df_h1["close"].iloc[-1]
    current_atr   = atr(df_h1, 14).iloc[-1]

    # ── 4. Volume confirmation ────────────────────────────────────────────
    # [FIX] Avoid ambiguous DataFrame truth value check
    df_vol = get_mt5_ohlcv(symbol, "M5", 50)
    # Use .empty for robust DataFrame check
    if df_vol is None or df_vol.empty:
        df_vol = df_m15

    vol_ok, vol_ratio = check_volume_confirmation(
        df_vol,
        min_volume_ratio=config.get("gold_min_volume_ratio", 1.3)
    )
    if not vol_ok:
        log.info("[SMC] Volume %.2fx below threshold — skip", vol_ratio)
        return None

    # ── 5. HTF structure (H4 / D1) — determine bias ───────────────────────
    ms_h4 = analyze_market_structure(df_h4.tail(100), window=5)
    ms_d1 = analyze_market_structure(df_d1.tail(50), window=3) if (df_d1 is not None and not df_d1.empty) else None

    log.info("[SMC] H4 structure: %s | last_event: %s", ms_h4.trend, ms_h4.last_event)
    if ms_d1 is not None:
        log.info("[SMC] D1 structure: %s", ms_d1.trend)

    # Determine directional bias from H4
    if ms_h4.just_broke_up() or ms_h4.trend == "BULLISH":
        bias = "BUY"
    elif ms_h4.just_broke_down() or ms_h4.trend == "BEARISH":
        bias = "SELL"
    else:
        log.info("[SMC] H4 ranging — no clear bias, skip")
        return None

    # D1 gate: don't trade against daily
    # [FIX] ms_d1 is a MarketStructure object, but let's be safe
    if ms_d1 is not None:
        if bias == "BUY"  and ms_d1.trend == "BEARISH":
            log.info("[SMC] D1 BEARISH blocks BUY — skip")
            return None
        if bias == "SELL" and ms_d1.trend == "BULLISH":
            log.info("[SMC] D1 BULLISH blocks SELL — skip")
            return None

    log.info("[SMC] Bias: %s", bias)

    # ── 6. Build H1 SMC context ───────────────────────────────────────────
    ctx = build_smc_context(
        df_h1,
        lookback_structure=100,
        lookback_ob=60,
        lookback_fvg=60,
        lookback_liq=100,
    )

    # ── 7. Premium / discount filter ──────────────────────────────────────
    pd_zone, equilibrium = get_premium_discount(df_h1, lookback=50)
    log.info("[SMC] Price zone: %s | EQ: %.2f", pd_zone, equilibrium)

    # Buy in discount, sell in premium
    if bias == "BUY"  and pd_zone == "PREMIUM":
        log.info("[SMC] BUY in PREMIUM zone — waiting for discount entry")
        return None
    if bias == "SELL" and pd_zone == "DISCOUNT":
        log.info("[SMC] SELL in DISCOUNT zone — waiting for premium entry")
        return None

    # ── 8. SMC score ──────────────────────────────────────────────────────
    smc_score, smc_reason = score_smc_signal(ctx, bias)
    log.info("[SMC] Score: %d | %s", smc_score, smc_reason)

    min_score = config.get("gold_min_score", 70)
    if smc_score < min_score:
        log.info("[SMC] Score %d < %d — skip", smc_score, min_score)
        return None

    # ── 9. M15 entry confirmation ─────────────────────────────────────────
    m15_confirmed, m15_reason = check_entry_confirmation(df_m15, bias)
    if not m15_confirmed:
        log.info("[SMC] No M15 confirmation (%s) — skip", m15_reason)
        return None

    log.info("[SMC] M15 confirmed: %s", m15_reason)

    # ── 10. RSI filter (basic sanity) ────────────────────────────────────
    from indicators import rsi as calc_rsi
    rsi_val = calc_rsi(df_h1["close"], 14).iloc[-1]
    if bias == "BUY"  and rsi_val > 72:
        log.info("[SMC] RSI overbought %.1f — skip BUY", rsi_val)
        return None
    if bias == "SELL" and rsi_val < 28:
        log.info("[SMC] RSI oversold %.1f — skip SELL", rsi_val)
        return None

    # ── 11. SL / TP calculation ───────────────────────────────────────────
    sym_info   = mt5.symbol_info(symbol)
    min_stop   = (sym_info.trade_stops_level * sym_info.point) if sym_info else 0
    tick       = mt5.symbol_info_tick(symbol)
    spread_buf = (tick.ask - tick.bid) * 2 if tick else 0

    if bias == "BUY":
        # [IMPROVEMENT] Wider SL for SMC (0.3 -> 0.8 ATR buffer)
        ob = ctx.nearest_demand_ob()
        if ob and ob.is_fresh:
            sl = ob.low - current_atr * 0.8 - spread_buf
            log.info("[SMC] SL below Demand OB low: %.2f", sl)
        else:
            sl = current_price - current_atr * 2.0 - spread_buf

        sl = min(sl, current_price - min_stop - spread_buf)

        # TP at nearest BSL (liquidity above) or next swing high
        bsl = ctx.bsl_levels[0] if ctx.bsl_levels else None
        if bsl and bsl.price > current_price:
            # [IMPROVEMENT] Ensure TP is at least 1.5R
            tp = max(bsl.price, current_price + (current_price - sl) * 1.5)
            log.info("[SMC] TP at BSL: %.2f", tp)
        else:
            risk  = current_price - sl
            tp    = current_price + risk * 2.0   # default 2R

    else:  # SELL
        # [IMPROVEMENT] Wider SL for SMC (0.3 -> 0.8 ATR buffer)
        ob = ctx.nearest_supply_ob()
        if ob and ob.is_fresh:
            sl = ob.high + current_atr * 0.8 + spread_buf
            log.info("[SMC] SL above Supply OB high: %.2f", sl)
        else:
            sl = current_price + current_atr * 2.0 + spread_buf

        sl = max(sl, current_price + min_stop + spread_buf)

        ssl = ctx.ssl_levels[0] if ctx.ssl_levels else None
        if ssl and ssl.price < current_price:
            # [IMPROVEMENT] Ensure TP is at least 1.5R
            tp = min(ssl.price, current_price - (sl - current_price) * 1.5)
            log.info("[SMC] TP at SSL: %.2f", tp)
        else:
            risk = sl - current_price
            tp   = current_price - risk * 2.0

    # Validate RR
    risk_pts = abs(current_price - sl)
    reward_pts = abs(tp - current_price)
    rr_ratio = reward_pts / risk_pts if risk_pts > 0 else 0

    if rr_ratio < 1.3:
        log.info("[SMC] RR %.2f too low — skip", rr_ratio)
        return None

    # ── 12. Position sizing ───────────────────────────────────────────────
    risk_pct  = config.get("gold_risk_pct", 0.25)
    lot = calculate_lot_size(
        config.get("gold_account_balance", 1000),
        risk_pct,
        current_price, sl,
        current_atr,
        symbol,
        config.get("gold_lot_base", 0.01),
        config.get("gold_max_lot", 5.0),
    )
    risk_usdt   = config.get("gold_account_balance", 1000) * risk_pct / 100
    partial_tps = calculate_partial_tp(current_price, bias, sl, rr_ratio)

    # ── 13. Build reason string ───────────────────────────────────────────
    ob_info  = f"OB@{ob.price:.1f}" if (ob and ob.is_fresh) else "NO_OB"
    fvg      = ctx.nearest_bull_fvg() if bias == "BUY" else ctx.nearest_bear_fvg()
    fvg_info = f"FVG@{fvg.price:.1f}" if fvg else "NO_FVG"
    reason   = (
        f"{session} | SMC | Score:{smc_score} | {smc_reason} | "
        f"{m15_reason} | {ob_info} | {fvg_info} | "
        f"PD:{pd_zone} | RR:{rr_ratio:.1f} | RSI:{rsi_val:.0f} | VOL:{vol_ratio:.1f}x"
    )

    log.info(
        "[SMC] SIGNAL: %s | Score:%d | Lot:%.3f | SL:%.2f | TP:%.2f | RR:%.2f",
        bias, smc_score, lot, sl, tp, rr_ratio
    )

    return Signal(
        market="GOLD",
        symbol=symbol,
        action=bias,
        entry=current_price,
        sl=round(sl, 2),
        tp=round(tp, 2),
        lot_or_qty=lot,
        score=smc_score,
        reason=reason,
        sr_level=ob.price if (ob and ob.is_fresh) else current_price,
        sr_type="SUPPORT" if bias == "BUY" else "RESISTANCE",
        zone_strength=ob.strength if (ob and ob.is_fresh) else 0,
        trend_1h=ms_h4.trend,
        rsi=rsi_val,
        risk_usdt=risk_usdt,
        partial_tps=partial_tps,
        trailing_stop_atr_mult=1.5,
        rr_ratio=rr_ratio,
        atr_value=current_atr,
    )


# ── Combined mode: run both strategies, take the higher score ────────────────

def check_gold_signal_combined(config: dict) -> Optional[Signal]:
    """
    Run both the classic (EMA/RSI/MACD/SR) strategy AND the SMC strategy.
    Return whichever produces the higher-scoring signal, or None if neither fires.

    This is the recommended mode for live trading — SMC and classic often
    agree on the best setups (they just describe them differently).
    """
    from gold_strategy import check_gold_signal as classic_signal

    sig_classic = None
    sig_smc     = None

    try:
        sig_classic = classic_signal(config)
    except Exception as e:
        log.error("[COMBINED] Classic signal error: %s", e)

    try:
        sig_smc = check_gold_signal_smc(config)
    except Exception as e:
        log.error("[COMBINED] SMC signal error: %s", e)

    # [FIX] Explicitly check for Signal objects to avoid ambiguous DataFrame truth value errors
    # Both fired — take the higher score
    if sig_classic is not None and sig_smc is not None:
        if sig_classic.action != sig_smc.action:
            # Conflicting direction — skip (don't trade when strategies disagree)
            log.info("[COMBINED] Classic=%s SMC=%s conflict — skip",
                     sig_classic.action, sig_smc.action)
            return None
        winner = sig_classic if sig_classic.score >= sig_smc.score else sig_smc
        log.info("[COMBINED] Both fired same direction — taking higher score: %d (%s)",
                 winner.score, "classic" if winner is sig_classic else "SMC")
        return winner

    # Only one fired
    if sig_smc is not None:
        log.info("[COMBINED] SMC only: %s score=%d", sig_smc.action, sig_smc.score)
        return sig_smc
    if sig_classic is not None:
        log.info("[COMBINED] Classic only: %s score=%d", sig_classic.action, sig_classic.score)
        return sig_classic

    return None
