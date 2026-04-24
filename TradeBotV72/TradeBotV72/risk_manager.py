"""
risk_manager.py — SUPER TRADER v7.0 Risk Management

FIXES from v6.0:
  [CRITICAL] _get_account_balance() was adding Gold + Crypto balances together,
             making the daily loss % denominator wrong. Gold $50 + Crypto $1000 = $1050
             denominator meant 1.5% limit was actually ~48% of Gold balance.
             Now uses per-market balance based on signal.market.
  [FIX]      can_trade() now passes signal.market to balance lookup
  [FIX]      status() shows per-market breakdown
"""

import logging
from datetime import date, datetime, timedelta
from typing import List, Optional
from signal_model import Signal

log = logging.getLogger(__name__)


class RiskManager:
    def __init__(self, config: dict):
        self.config = config
        self.open_trades: List[Signal] = []
        self.daily_pnl: float = 0.0
        self.daily_trades: int = 0
        self.last_reset: date = date.today()
        self.halted: bool = False

        # ── Circuit breaker limits ────────────────────────────────────────────
        self.consecutive_losses: int = 0
        self.max_consecutive_losses: int = 2

        self.consecutive_wins: int = 0

        # ── Trade count limits ────────────────────────────────────────────────
        self.max_open_trades: int = 2
        self.max_trades_per_direction: int = 1
        self.max_daily_trades: int = config.get("gold_max_daily_trades", 4)

        # ── Daily loss limit ──────────────────────────────────────────────────
        self.max_daily_loss_pct: float = config.get("max_daily_loss_pct", 1.5)

        # ── Per-symbol loss cap ───────────────────────────────────────────────
        self.symbol_pnl: dict = {}
        self.max_symbol_loss_pct: float = 1.0

        # ── ATR tracking ──────────────────────────────────────────────────────
        self.current_atr: float = 0.0
        self.normal_atr: float = 10.0

    # ── Daily reset ───────────────────────────────────────────────────────────

    def _reset_daily_if_needed(self):
        today = date.today()
        if today != self.last_reset:
            log.info(
                "[RISK] Daily reset | P&L: $%.2f | Trades: %d | "
                "Wins: %d | Losses: %d",
                self.daily_pnl, self.daily_trades,
                self.consecutive_wins, self.consecutive_losses,
            )
            self.daily_pnl = 0.0
            self.daily_trades = 0
            self.consecutive_losses = 0
            self.consecutive_wins = 0
            self.symbol_pnl = {}
            self.halted = False
            self.last_reset = today

    # ── [FIX] Per-market balance lookup ──────────────────────────────────────

    def _get_market_balance(self, market: str) -> float:
        """
        [FIX v6.1] Return the balance for the specific market being traded.
        Previously summed Gold + Crypto, inflating denominator for loss % checks.
        Gold $50 + Crypto $1000 = $1050 meant 1.5% limit was ~$15.75 not $0.75.
        """
        if market == "GOLD":
            return float(self.config.get("gold_account_balance", 0))
        elif market == "CRYPTO":
            return float(self.config.get("crypto_account_balance", 0))
        # Fallback: use whichever is non-zero
        gold = float(self.config.get("gold_account_balance", 0))
        crypto = float(self.config.get("crypto_account_balance", 0))
        return gold if gold > 0 else crypto

    def _get_total_balance(self) -> float:
        """Total across all markets — used only for status display."""
        return float(
            self.config.get("gold_account_balance", 0)
            + self.config.get("crypto_account_balance", 0)
        )

    # ── Risk multiplier ────────────────────────────────────────────────────────

    def get_risk_multiplier(self) -> float:
        base = 1.0
        if self.consecutive_wins >= 1:
            base += min(0.4, self.consecutive_wins * 0.15)
        if self.consecutive_losses >= 1:
            base -= min(0.5, self.consecutive_losses * 0.25)
        if self.current_atr > 0:
            vol_factor = min(1.0, self.normal_atr / max(self.current_atr, 0.1))
            base *= vol_factor
        return max(0.25, min(1.5, base))

    def get_adjusted_risk_pct(self, base_risk_pct: float, signal_score: int) -> float:
        if signal_score >= 80:
            score_mult = 1.5
        elif signal_score >= 65:
            score_mult = 1.25
        elif signal_score >= 55:
            score_mult = 1.0
        else:
            score_mult = 0.75
        streak_mult = self.get_risk_multiplier()
        adjusted = base_risk_pct * score_mult * streak_mult
        max_risk = self.config.get("crypto_max_risk_pct", 2.0)
        return min(adjusted, max_risk)

    # ── can_trade ──────────────────────────────────────────────────────────────

    def can_trade(self, signal: Signal) -> bool:
        self._reset_daily_if_needed()

        if self.halted:
            log.warning("[RISK] Bot halted — manual reset required")
            return False

        if self.consecutive_losses >= self.max_consecutive_losses:
            self.halted = True
            log.warning(
                "[RISK] Consecutive loss limit hit (%d/%d) — halting",
                self.consecutive_losses, self.max_consecutive_losses,
            )
            return False

        if len(self.open_trades) >= self.max_open_trades:
            log.info("[RISK] Max open trades reached (%d/%d)",
                     len(self.open_trades), self.max_open_trades)
            return False

        same_dir = [
            t for t in self.open_trades
            if t.symbol == signal.symbol and t.action == signal.action
        ]
        if len(same_dir) >= self.max_trades_per_direction:
            log.info(
                "[RISK] Max trades per direction reached: %s %s (%d/%d)",
                signal.symbol, signal.action,
                len(same_dir), self.max_trades_per_direction,
            )
            return False

        if self.daily_trades >= self.max_daily_trades:
            log.info("[RISK] Daily trade limit reached (%d/%d)",
                     self.daily_trades, self.max_daily_trades)
            return False

        # [FIX] Use per-market balance, not combined total
        account = self._get_market_balance(signal.market)
        if account > 0 and self.daily_pnl < 0:
            daily_loss_pct = abs(self.daily_pnl) / account * 100
            if daily_loss_pct >= self.max_daily_loss_pct:
                self.halted = True
                log.warning(
                    "[RISK] Daily loss limit hit: %.2f%% of %s balance $%.2f (limit %.2f%%) — halting",
                    daily_loss_pct, signal.market, account, self.max_daily_loss_pct,
                )
                return False

        # Per-symbol loss cap — also use market balance
        sym_pnl = self.symbol_pnl.get(signal.symbol, 0.0)
        if account > 0 and sym_pnl < 0:
            sym_loss_pct = abs(sym_pnl) / account * 100
            if sym_loss_pct >= self.max_symbol_loss_pct:
                log.warning(
                    "[RISK] Symbol %s daily loss cap hit: %.2f%% of $%.2f — skip",
                    signal.symbol, sym_loss_pct, account,
                )
                return False

        log.info(
            "[RISK] Trade allowed | Open:%d/%d | Daily:%d/%d | "
            "P&L:$%.2f | Wins:%d | Losses:%d",
            len(self.open_trades), self.max_open_trades,
            self.daily_trades, self.max_daily_trades,
            self.daily_pnl,
            self.consecutive_wins, self.consecutive_losses,
        )
        return True

    # ── Register trade ─────────────────────────────────────────────────────────

    def register_trade(self, signal: Signal):
        self.open_trades.append(signal)
        self.daily_trades += 1
        if hasattr(signal, "atr_value") and signal.atr_value > 0:
            self.current_atr = signal.atr_value
        log.info(
            "[RISK] Trade registered: %s %s | Open:%d/%d",
            signal.symbol, signal.action,
            len(self.open_trades), self.max_open_trades,
        )

    # ── Close trade ────────────────────────────────────────────────────────────

    def close_trade(self, symbol: str, pnl: float, action: str = None):
        candidates = [t for t in self.open_trades if t.symbol == symbol]
        if not candidates:
            log.warning("[RISK] close_trade: no open trade found for %s", symbol)
            return

        if action:
            match = [t for t in candidates if t.action == action]
            to_close = match[0] if match else candidates[0]
        else:
            to_close = candidates[0]

        self.open_trades.remove(to_close)
        self.daily_pnl += pnl
        self.symbol_pnl[symbol] = self.symbol_pnl.get(symbol, 0.0) + pnl

        if pnl > 0:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            result = "WIN"
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            result = "LOSS"

        log.info(
            "[RISK] Trade closed: %s %s | %s $%.2f | "
            "Daily:$%.2f | Win streak:%d | Loss streak:%d | Open:%d/%d",
            symbol, to_close.action, result, abs(pnl),
            self.daily_pnl,
            self.consecutive_wins, self.consecutive_losses,
            len(self.open_trades), self.max_open_trades,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def reset_halt(self):
        self.halted = False
        self.consecutive_losses = 0
        log.info("[RISK] Halt manually reset")

    def status(self) -> str:
        gold_bal   = float(self.config.get("gold_account_balance", 0))
        crypto_bal = float(self.config.get("crypto_account_balance", 0))
        total      = gold_bal + crypto_bal

        daily_loss_pct = (
            abs(self.daily_pnl) / gold_bal * 100
            if gold_bal > 0 and self.daily_pnl < 0
            else 0
        )
        open_list = (
            ", ".join(f"{t.symbol}({t.action})" for t in self.open_trades) or "none"
        )
        risk_mult = self.get_risk_multiplier()
        return (
            f"Open trades : {len(self.open_trades)}/{self.max_open_trades} — {open_list}\n"
            f"Daily P&L   : ${self.daily_pnl:.2f} ({daily_loss_pct:.2f}% of Gold bal)\n"
            f"Gold bal    : ${gold_bal:.2f}\n"
            f"Crypto bal  : ${crypto_bal:.2f}\n"
            f"Daily trades: {self.daily_trades}/{self.max_daily_trades}\n"
            f"Win streak  : {self.consecutive_wins}\n"
            f"Loss streak : {self.consecutive_losses}/{self.max_consecutive_losses}\n"
            f"Risk mult   : {risk_mult:.2f}x\n"
            f"Halted      : {self.halted}\n"
            f"Max dir/sym : {self.max_trades_per_direction}\n"
            f"Daily loss% : {self.max_daily_loss_pct}% limit (per-market)"
        )
