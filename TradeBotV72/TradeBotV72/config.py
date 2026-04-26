"""
config.py — v7.4 AGGRESSIVE SMALL ACCOUNT FLIP
Optimized for $30 balance to $30 daily profit.
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
    
    # ── Gold Strategy (AGGRESSIVE FLIP SETTINGS) ──────────────────────────────
    "gold_lot_base":        0.01,
    "gold_account_balance": 30.0,
    "gold_risk_pct":        10.0,   # High risk for small account
    "gold_rr_ratio":        1.5,    # Quick scalps
    "gold_sr_lookback":     200,
    "gold_sr_touches":      2,
    "gold_sr_zone_pips":    8,
    "gold_rsi_period":      7,      # Faster RSI
    "gold_rsi_oversold":    30,
    "gold_rsi_overbought":  70,
    "gold_ema_fast":        9,      # Faster EMA
    "gold_ema_slow":        21,
    "gold_max_lot":         0.05,
    
    # ── Signal quality gates ──────────────────────────────────────────────────
    "gold_min_volume_ratio":   1.0,  # No volume filter for high frequency
    "gold_max_entry_dist_pct": 0.01,
    "gold_min_score":          45,   # Lower score for more trades
    "gold_volume_filter":      False,
    
    # ── Spread Control ────────────────────────────────────────────────────────
    "gold_max_spread_pips": 100,
    
    # ── Global Risk ───────────────────────────────────────────────────────────
    "gold_scalp_rr":            1.5,
    "max_open_trades":          3,   # Allow more concurrent scalps
    "max_trades_per_direction": 2,
    "max_daily_loss_pct":       20.0, # High risk tolerance for flip
}
