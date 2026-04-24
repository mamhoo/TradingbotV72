"""
config.py — v6.1 updated defaults

CHANGES from v6.0:
  GOLD_MAX_ENTRY_DIST_PCT  0.004 → 0.006  (anti-chase relaxed)
  GOLD_MIN_VOLUME_RATIO    1.3   → 1.3    (same, but now checks completed candle)
  GOLD_MIN_SCORE           55    → 55     (same, D1 counter-trend raises to 65)
  MAX_DAILY_LOSS_PCT        1.5  → 1.5    (unchanged)
"""

import os
from dotenv import load_dotenv

load_dotenv()


def parse_int(val, default=0):
    try:
        return int(val) if val else default
    except Exception:
        return default


def parse_float(val, default=0.0):
    try:
        return float(val) if val else default
    except Exception:
        return default


def parse_list(val):
    if not val:
        return []
    return [x.strip() for x in val.split(",") if x.strip()]


CONFIG = {
    # ── Telegram ──────────────────────────────────────────────────────────────
    "telegram_token":    os.getenv("TELEGRAM_TOKEN", ""),
    "telegram_chat_ids": parse_list(os.getenv("TELEGRAM_CHAT_IDS", "")),

    # ── MT5 (Gold) ────────────────────────────────────────────────────────────
    "mt5_login":    parse_int(os.getenv("MT5_LOGIN"), 0),
    "mt5_password": os.getenv("MT5_PASSWORD", ""),
    "mt5_server":   os.getenv("MT5_SERVER", ""),
    "mt5_symbol":   os.getenv("MT5_SYMBOL", "GOLD"),
    "mt5_magic":    parse_int(os.getenv("MT5_MAGIC"), 20250001),

    # ── Crypto Exchange ───────────────────────────────────────────────────────
    "exchange":       os.getenv("EXCHANGE", "binance"),
    "api_key":        os.getenv("API_KEY", ""),
    "api_secret":     os.getenv("API_SECRET", ""),
    "crypto_symbols": parse_list(os.getenv("CRYPTO_SYMBOLS", "")),

    # ── Gold Strategy ─────────────────────────────────────────────────────────
    "gold_lot_base":        parse_float(os.getenv("GOLD_LOT_BASE"),       0.01),
    "gold_account_balance": parse_float(os.getenv("GOLD_ACCOUNT_BALANCE"), 50),
    "gold_risk_pct":        parse_float(os.getenv("GOLD_RISK_PCT"),        1.0),
    "gold_rr_ratio":        parse_float(os.getenv("GOLD_RR_RATIO"),        1.5),
    "gold_sr_lookback":     parse_int(os.getenv("GOLD_SR_LOOKBACK"),       200),
    "gold_sr_touches":      parse_int(os.getenv("GOLD_SR_TOUCHES"),        2),
    "gold_sr_zone_pips":    parse_float(os.getenv("GOLD_SR_ZONE_PIPS"),    8),
    "gold_rsi_period":      parse_int(os.getenv("GOLD_RSI_PERIOD"),        14),
    "gold_rsi_oversold":    parse_int(os.getenv("GOLD_RSI_OVERSOLD"),      35),
    "gold_rsi_overbought":  parse_int(os.getenv("GOLD_RSI_OVERBOUGHT"),    65),
    "gold_ema_fast":        parse_int(os.getenv("GOLD_EMA_FAST"),          21),
    "gold_ema_slow":        parse_int(os.getenv("GOLD_EMA_SLOW"),          55),
    "gold_max_lot":         parse_float(os.getenv("GOLD_MAX_LOT"),         5.0),

    # ── Signal quality gates (v6.1 updated) ──────────────────────────────────
    "gold_min_volume_ratio":   parse_float(os.getenv("GOLD_MIN_VOLUME_RATIO"),   1.3),
    "gold_max_entry_dist_pct": parse_float(os.getenv("GOLD_MAX_ENTRY_DIST_PCT"), 0.006),  # v6.1: 0.004→0.006
    "gold_min_score":          parse_int(os.getenv("GOLD_MIN_SCORE"),            55),
    "gold_volume_filter":      True,

    # ── Spread Control ────────────────────────────────────────────────────────
    "gold_max_spread_pips": parse_int(os.getenv("GOLD_MAX_SPREAD_PIPS"), 80),

    # ── Crypto Risk ───────────────────────────────────────────────────────────
    "crypto_account_balance": parse_float(os.getenv("CRYPTO_ACCOUNT_BALANCE"), 1000),
    "crypto_risk_pct":        parse_float(os.getenv("CRYPTO_RISK_PCT"),         0.5),
    "crypto_max_risk_pct":    parse_float(os.getenv("CRYPTO_MAX_RISK_PCT"),     1.5),
    "crypto_rr_ratio":        parse_float(os.getenv("CRYPTO_RR_RATIO"),         2.0),
    "crypto_leverage":        parse_int(os.getenv("CRYPTO_LEVERAGE"),           1),

    # ── Global Risk ───────────────────────────────────────────────────────────
    "gold_scalp_rr":            parse_float(os.getenv("GOLD_SCALP_RR"), 1.5),
    "max_open_trades":          2,
    "max_trades_per_direction": 1,
    "max_daily_loss_pct":       parse_float(os.getenv("MAX_DAILY_LOSS_PCT"), 1.5),
}
