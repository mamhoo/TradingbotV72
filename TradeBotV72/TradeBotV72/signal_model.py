"""
signal_model.py — Signal data model with advanced features

Supports:
  - Partial take-profit levels
  - Trailing stop configuration
  - Pyramiding (add-to-position) levels
  - Signal metadata for analytics
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Tuple


@dataclass
class Signal:
    market: str           # "GOLD" or "CRYPTO"
    symbol: str
    action: str           # "BUY" or "SELL"
    entry: float
    sl: float
    tp: float             # Primary TP (TP1 for partial exit)
    lot_or_qty: float
    score: int            # confidence score 0–100
    reason: str           # human-readable reason
    sr_level: float       # nearest S/R level triggered
    sr_type: str          # "SUPPORT" or "RESISTANCE"
    zone_strength: int    # number of touches on this level
    trend_1h: str         # "UP" / "DOWN" / "NEUTRAL"
    rsi: float
    risk_usdt: float      # dollar amount at risk
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Advanced features
    partial_tps: List[Tuple[float, float]] = field(default_factory=list)  # [(price1, pct1), ...]
    trailing_stop_atr_mult: float = 1.5  # ATR multiplier for trailing
    rr_ratio: float = 1.5  # Risk:Reward ratio
    atr_value: float = 0.0  # ATR at entry for trailing calc
    pyramiding_enabled: bool = False
    pyramiding_levels: List[Tuple[float, float]] = field(default_factory=list)  # [(price, lot)]

    def rr(self) -> float:
        if abs(self.entry - self.sl) == 0:
            return 0.0
        return abs(self.tp - self.entry) / abs(self.entry - self.sl)

    def summary(self) -> str:
        emoji = "🟢" if self.action == "BUY" else "🔴"
        partial_info = ""
        if self.partial_tps:
            partial_info = f"\nPartial TP: {len(self.partial_tps)} levels | "
        return (
            f"{emoji} *{self.action} {self.symbol}*\n"
            f"━━━━━━━━━━━━━━\n"
            f"Entry : `{self.entry:.4f}`\n"
            f"SL    : `{self.sl:.4f}`\n"
            f"TP    : `{self.tp:.4f}`\n"
            f"R:R   : `1:{self.rr():.1f}`\n"
            f"━━━━━━━━━━━━━━\n"
            f"Zone  : {self.sr_type} @ {self.sr_level:.2f} ({self.zone_strength} touches)\n"
            f"Score : {self.score}/100\n"
            f"Trend : {self.trend_1h}\n"
            f"RSI   : {self.rsi:.1f}\n"
            f"Risk  : ${self.risk_usdt:.2f}\n"
            f"Lot   : {self.lot_or_qty}\n"
            f"{partial_info}Reason: {self.reason}\n"
            f"━━━━━━━━━━━━━━\n"
            f"🕐 {self.timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
        )

    def to_dict(self) -> dict:
        """Convert signal to dict for database storage."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "market": self.market,
            "symbol": self.symbol,
            "action": self.action,
            "entry": self.entry,
            "sl": self.sl,
            "tp": self.tp,
            "lot_qty": self.lot_or_qty,
            "score": self.score,
            "sr_type": self.sr_type,
            "sr_level": self.sr_level,
            "zone_strength": self.zone_strength,
            "trend_1h": self.trend_1h,
            "rsi": self.rsi,
            "risk_usdt": self.risk_usdt,
            "reason": self.reason,
            "rr_ratio": self.rr_ratio,
            "atr_value": self.atr_value,
        }
