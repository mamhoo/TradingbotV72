"""
session_config.py — 24/7 Global Market Coverage
Optimized for Thai Time (UTC+7) 
Targeting: Sydney, Tokyo, London, and New York sessions
"""

from datetime import datetime, timezone, timedelta
from typing import Tuple

# SESSION PARAMETERS
# Adjusted for XM Gold spreads and global liquidity cycles
# REWRITTEN v3.0 - stricter thresholds for quality setups only
SESSION_PARAMS = {
    "SYDNEY_TOKYO": {
        "min_score":        70,         # Stricter for lower Asian volume
        "max_spread_pips":  100,        # Tighter spread limit
        "macd_methods":     ["ZERO_CROSS"],  # Only best signals
        "rsi_sweet_range":  (40, 60),
        "atr_min_ratio":    0.0002,     # Require some volatility
        "min_volume_ratio": 1.05,       # Lower for Asian session
        "scan_interval":    60,         # Standard 60s scan
        "active":           True,
        "description":      "Sydney & Tokyo - Thai 04:00-14:00",
    },
    "LONDON": {
        "min_score":        65,
        "max_spread_pips":  70,
        "macd_methods":     ["ZERO_CROSS"],
        "rsi_sweet_range":  (38, 62),
        "atr_min_ratio":    0.0003,
        "min_volume_ratio": 1.10,       # Moderate for London session
        "scan_interval":    60,         # Standard 60s scan
        "active":           True,
        "description":      "London Open - Thai 14:00-19:00",
    },
    "LONDON_NY_OVERLAP": {
        "min_score":        60,         # Best session - can be slightly looser
        "max_spread_pips":  75,
        "macd_methods":     ["ZERO_CROSS"],
        "rsi_sweet_range":  (35, 65),
        "atr_min_ratio":    0.0004,     # Require high movement
        "min_volume_ratio": 1.15,       # Higher for London/NY overlap
        "scan_interval":    30,         # TURBO MODE: 30s scan
        "active":           True,
        "description":      "London+NY Overlap - Thai 19:00-23:00 (BEST)",
    },
    "NEW_YORK": {
        "min_score":        65,
        "max_spread_pips":  80,
        "macd_methods":     ["ZERO_CROSS"],
        "rsi_sweet_range":  (35, 65),
        "atr_min_ratio":    0.0003,
        "min_volume_ratio": 1.10,       # Moderate for New York session
        "scan_interval":    60,         # Standard 60s scan
        "active":           True,
        "description":      "NY Late Session - Thai 23:00-04:00",
    }
}

def get_current_session() -> Tuple[str, dict]:
    """
    Determines the trading session based on UTC hour to align with Thai Time (UTC+7).
    Covers 24 hours with no dead zones.
    """
    # Get current hour in UTC
    h_utc = datetime.now(timezone.utc).hour
    
    # 1. LONDON/NY OVERLAP: Thai 19:00 - 23:00 (UTC 12:00 - 16:00)
    if 12 <= h_utc < 16:
        return "LONDON_NY_OVERLAP", SESSION_PARAMS["LONDON_NY_OVERLAP"]
    
    # 2. NEW YORK: Thai 23:00 - 04:00 (UTC 16:00 - 21:00)
    elif 16 <= h_utc < 21:
        return "NEW_YORK", SESSION_PARAMS["NEW_YORK"]
    
    # 3. SYDNEY/TOKYO: Thai 04:00 - 14:00 (UTC 21:00 - 07:00)
    elif 21 <= h_utc or h_utc < 7:
        return "SYDNEY_TOKYO", SESSION_PARAMS["SYDNEY_TOKYO"]
        
    # 4. LONDON: Thai 14:00 - 19:00 (UTC 07:00 - 12:00)
    else:
        return "LONDON", SESSION_PARAMS["LONDON"]

def is_tradeable() -> Tuple[bool, str, dict]:
    """Returns if the current time is active, the session name, and its parameters."""
    name, params = get_current_session()
    return params["active"], name, params

def thai_time_str() -> str:
    """Helper to display the current time in Thai format for logs."""
    thai = datetime.now(timezone.utc) + timedelta(hours=7)
    return thai.strftime("%H:%M Thai")

if __name__ == "__main__":
    # Test Output
    tradeable, name, p = is_tradeable()
    print(f"Current Time: {thai_time_str()}")
    print(f"Session:      {name}")
    print(f"Tradeable:    {tradeable}")
    print(f"Description:  {p['description']}")