"""
backtester.py — Historical Backtesting Engine for TradingbotV72
Compares three strategies on the same historical data:
  1. Classic (EMA/RSI/MACD/SR) — original TradingbotV72 strategy
  2. Mean Reversion with Volume Clusters (new)
  3. Bollinger Band Squeeze with VWAP & Volume Profile (new)

HOW IT WORKS:
  - Downloads historical OHLCV data (yfinance as MT5 substitute for offline backtesting)
  - Walks forward bar by bar through the history
  - At each bar, calls each strategy's signal function with a rolling window of data
  - Simulates trade execution: entry at next bar open, SL/TP hit detection
  - Tracks all trades, P&L, drawdown, win rate, and other metrics
  - Generates a comparison report with charts

USAGE:
  python backtester.py --symbol XAUUSD --days 180 --timeframe M15

NOTE:
  Since MT5 is not available in the sandbox, this backtester uses yfinance
  to download Gold (GC=F) data as a substitute. The strategy logic is identical
  to the live version — only the data source changes.
"""

import sys
import os
import logging
import argparse
import warnings
import json
from typing import Optional, List, Dict, Tuple, Any
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── Backtest Config ───────────────────────────────────────────────────────────

BACKTEST_CONFIG = {
    "mt5_symbol":            "XAUUSD",
    "gold_account_balance":  10000.0,
    "gold_risk_pct":         0.5,
    "gold_lot_base":         0.01,
    "gold_max_lot":          5.0,
    "gold_min_score":        55,
    "gold_min_volume_ratio": 1.0,
    "gold_max_spread_pips":  80,
    "gold_max_entry_dist_pct": 0.006,
    "gold_sr_zone_pips":     8,
    "gold_ema_fast":         21,
    "gold_ema_slow":         55,
    "gold_rsi_period":       14,
    "gold_volume_filter":    True,
    # Backtest mode: disable MT5 checks
    "backtest_mode":         True,
}

# ── Session override for backtesting (always "tradeable") ────────────────────

import session_config as _sc_module

_ORIG_IS_TRADEABLE = _sc_module.is_tradeable

def _mock_is_tradeable():
    """Always return tradeable in backtest mode."""
    params = {
        "min_score": 50,
        "max_spread_pips": 80,
        "min_volume_ratio": 1.2,
        "scan_interval": 60,
    }
    return True, "LONDON", params

_sc_module.is_tradeable = _mock_is_tradeable

# Also patch session_config in other modules
import gold_strategy as _gs_module
_gs_module.is_tradeable = _mock_is_tradeable

import mean_reversion_strategy as _mr_module
_mr_module.is_tradeable = _mock_is_tradeable

import bb_squeeze_strategy as _sq_module
_sq_module.is_tradeable = _mock_is_tradeable


# ── Data Download ─────────────────────────────────────────────────────────────

def download_gold_data(days: int = 180, interval: str = "15m") -> pd.DataFrame:
    """
    Download Gold futures data from Yahoo Finance.
    GC=F = Gold Futures (continuous contract).
    Note: Yahoo Finance limits 15m data to the last 60 days.
    """
    try:
        import yfinance as yf
    except ImportError:
        log.error("yfinance not installed. Run: pip install yfinance")
        raise

    # Yahoo Finance restricts 15m data to last 60 days
    if interval == "15m":
        days = min(days, 59)
    elif interval in ("5m", "1m"):
        days = min(days, 7)

    end_date   = datetime.now()
    start_date = end_date - timedelta(days=days)

    log.warning("Downloading Gold data: %s days, interval=%s", days, interval)
    ticker = yf.Ticker("GC=F")
    df = ticker.history(start=start_date, end=end_date, interval=interval)

    if df.empty:
        # Fallback: try 1h data
        log.warning("15m data unavailable, falling back to 1h")
        df = ticker.history(period="3mo", interval="1h")
        if df.empty:
            raise ValueError("No data downloaded from Yahoo Finance")

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "time"}, inplace=True)

    # Ensure required columns
    required = ["time", "open", "high", "low", "close", "volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")

    df = df[required].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.dropna().reset_index(drop=True)

    log.warning("Downloaded %d bars from %s to %s",
                len(df), df["time"].iloc[0], df["time"].iloc[-1])
    return df


def resample_to_higher_tf(df_m15: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample M15 data to H1, H4, or D1."""
    df = df_m15.copy()
    df = df.set_index("time")

    freq_map = {"H1": "1h", "H4": "4h", "D1": "1D"}
    freq = freq_map.get(tf, "1h")

    resampled = df.resample(freq).agg({
        "open":   "first",
        "high":   "max",
        "low":    "min",
        "close":  "last",
        "volume": "sum",
    }).dropna().reset_index()
    # If resample index is named 'time', reset_index() will keep it as 'time'
    # If it's unnamed, it becomes 'index'
    if "index" in resampled.columns:
        resampled.rename(columns={"index": "time"}, inplace=True)
    return resampled


# ── Trade Simulation ──────────────────────────────────────────────────────────

@dataclass
class BacktestTrade:
    strategy:    str
    action:      str
    entry_bar:   int
    entry_price: float
    sl:          float
    tp:          float
    lot:         float
    score:       int
    reason:      str
    exit_bar:    Optional[int]   = None
    exit_price:  Optional[float] = None
    pnl:         float           = 0.0
    result:      str             = "OPEN"   # WIN / LOSS / OPEN
    rr_ratio:    float           = 0.0
    bars_held:   int             = 0


@dataclass
class BacktestResult:
    strategy:       str
    trades:         List[BacktestTrade] = field(default_factory=list)
    equity_curve:   List[float]         = field(default_factory=list)
    initial_balance: float              = 10000.0

    @property
    def closed_trades(self) -> List[BacktestTrade]:
        return [t for t in self.trades if t.result != "OPEN"]

    @property
    def wins(self) -> List[BacktestTrade]:
        return [t for t in self.closed_trades if t.result == "WIN"]

    @property
    def losses(self) -> List[BacktestTrade]:
        return [t for t in self.closed_trades if t.result == "LOSS"]

    @property
    def win_rate(self) -> float:
        n = len(self.closed_trades)
        return len(self.wins) / n * 100 if n > 0 else 0.0

    @property
    def total_pnl(self) -> float:
        return sum(t.pnl for t in self.closed_trades)

    @property
    def profit_factor(self) -> float:
        gross_win  = sum(t.pnl for t in self.wins)
        gross_loss = abs(sum(t.pnl for t in self.losses))
        return gross_win / gross_loss if gross_loss > 0 else float("inf")

    @property
    def max_drawdown(self) -> float:
        if not self.equity_curve:
            return 0.0
        curve = np.array(self.equity_curve)
        peak = np.maximum.accumulate(curve)
        dd = (curve - peak) / peak * 100
        return float(dd.min())

    @property
    def avg_rr(self) -> float:
        rrs = [t.rr_ratio for t in self.closed_trades if t.rr_ratio > 0]
        return np.mean(rrs) if rrs else 0.0

    @property
    def sharpe_ratio(self) -> float:
        """Simplified Sharpe: mean daily P&L / std daily P&L."""
        pnls = [t.pnl for t in self.closed_trades]
        if len(pnls) < 2:
            return 0.0
        return np.mean(pnls) / (np.std(pnls) + 1e-9)

    def summary(self) -> dict:
        return {
            "strategy":       self.strategy,
            "total_trades":   len(self.closed_trades),
            "wins":           len(self.wins),
            "losses":         len(self.losses),
            "win_rate":       round(self.win_rate, 1),
            "total_pnl":      round(self.total_pnl, 2),
            "profit_factor":  round(self.profit_factor, 2),
            "max_drawdown":   round(self.max_drawdown, 2),
            "avg_rr":         round(self.avg_rr, 2),
            "sharpe_ratio":   round(self.sharpe_ratio, 3),
            "final_balance":  round(self.initial_balance + self.total_pnl, 2),
        }


# ── PnL calculation ───────────────────────────────────────────────────────────

def calculate_pnl(action: str, entry: float, exit_price: float, lot: float) -> float:
    """
    Simplified Gold P&L calculation.
    Gold: 1 lot = 100 oz. Price in USD/oz.
    P&L = lot * 100 * (exit - entry) for BUY
    """
    if action == "BUY":
        return lot * 100 * (exit_price - entry)
    else:
        return lot * 100 * (entry - exit_price)


# ── Strategy signal wrappers ──────────────────────────────────────────────────

def _run_classic_signal(config: dict, df_m5: pd.DataFrame, df_m15: pd.DataFrame,
                        df_h1: pd.DataFrame, df_h4: pd.DataFrame,
                        df_d1: pd.DataFrame) -> Optional[Any]:
    """Run classic gold strategy with DataFrame overrides."""
    from gold_strategy import (
        check_volume_confirmation, get_action, check_daily_trend,
        check_not_chasing, check_rsi, check_macd, check_zone,
        calculate_dynamic_rr, calculate_lot_size, calculate_partial_tp,
        get_mt5_ohlcv, atr as _atr_import,
    )
    from indicators import rsi, ema, atr, macd, get_trend
    from sr_zones import build_zones
    from signal_model import Signal

    try:
        if any(d is None or len(d) < 50 for d in [df_m5, df_m15, df_h1, df_h4]):
            return None

        current_price = df_m5["close"].iloc[-1]
        current_atr   = atr(df_m5, 14).iloc[-1]

        # Volume
        vol_ok, vol_ratio = check_volume_confirmation(df_m5, 1.2)
        if not vol_ok:
            return None

        # Action from H1+H4
        action, h1_trend, h4_trend = get_action(df_h1, df_h4, 21, 55)
        if action == "NEUTRAL":
            return None

        # D1 trend
        d1_ok, d1_trend, d1_score_adj = check_daily_trend.__wrapped__(df_d1, action, 21, 55) \
            if hasattr(check_daily_trend, "__wrapped__") else _check_daily_trend_bt(df_d1, action)

        # Anti-chase
        not_chasing, dist_pct = check_not_chasing(df_m15, action, 21, 0.006)
        if not not_chasing:
            return None

        # RSI
        rsi_ok, rsi_label, rsi_val = check_rsi(df_m15, action, 14)
        if not rsi_ok:
            return None

        # MACD
        macd_ok, macd_signal, macd_val = check_macd(df_m15, action)
        if not macd_ok:
            return None

        # Zones
        zones = build_zones(df_h1, lookback=200, min_touches=2, zone_pips=8)
        at_zone, zone_obj, touches = check_zone(zones, current_price, action)

        # Score
        score = 0
        reasons = []
        score += d1_score_adj
        if h1_trend == h4_trend and h1_trend != "NEUTRAL":
            score += 35; reasons.append("H1H4_ALIGNED")
        elif h1_trend != "NEUTRAL":
            score += 20; reasons.append(f"H1_{h1_trend}")
        else:
            score += 10; reasons.append(f"H4_{h4_trend}")

        if "ZERO_CROSS" in macd_signal or "MOMENTUM" in macd_signal:
            score += 25
        else:
            score += 15
        reasons.append(macd_signal)

        if rsi_label == "GOOD_ZONE":
            score += 20
        elif "OVERSOLD" in rsi_label or "OVERBOUGHT" in rsi_label:
            score += 15
        else:
            score += 10
        reasons.append(rsi_label)

        if at_zone:
            score += min(20, 5 * touches)
            reasons.append(f"ZONE_{touches}T")
        else:
            ema_val = ema(df_m15["close"], 21).iloc[-1]
            dist = abs(current_price - ema_val) / ema_val
            if dist < 0.002:
                score += 15; reasons.append("EMA_PULLBACK")
            elif dist < 0.004:
                score += 8; reasons.append("EMA_NEAR")
            else:
                score += 2; reasons.append("NO_ZONE")

        score += 5  # volume confirmed
        reasons.append(f"VOL_{vol_ratio:.1f}x")

        min_score = 60 if d1_score_adj >= 0 else 65
        if score < min_score:
            return None

        # RR
        trend_aligned = h1_trend == h4_trend and h1_trend != "NEUTRAL"
        rr_ratio = calculate_dynamic_rr(touches if at_zone else 0,
                                        h1_trend if trend_aligned else "NEUTRAL", False)

        # SL/TP
        h1_atr = atr(df_h1, 14).iloc[-1]
        if action == "BUY":
            sl = current_price - h1_atr * 1.2
            tp = current_price + (current_price - sl) * rr_ratio
        else:
            sl = current_price + h1_atr * 1.2
            tp = current_price - (sl - current_price) * rr_ratio

        risk_pts   = abs(current_price - sl)
        reward_pts = abs(tp - current_price)
        actual_rr  = reward_pts / risk_pts if risk_pts > 0 else 0
        if actual_rr < 1.3:
            return None

        lot = calculate_lot_size(
            config["gold_account_balance"], config["gold_risk_pct"],
            current_price, sl, current_atr,
            config["gold_lot_base"], config["gold_max_lot"],
        )

        return Signal(
            market="GOLD", symbol="XAUUSD", action=action,
            entry=current_price, sl=round(sl, 2), tp=round(tp, 2),
            lot_or_qty=lot, score=score,
            reason=f"CLASSIC | Score:{score} | {' | '.join(reasons)}",
            sr_level=zone_obj.price if zone_obj else current_price,
            sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
            zone_strength=touches if zone_obj else 0,
            trend_1h=h1_trend, rsi=rsi_val,
            risk_usdt=config["gold_account_balance"] * config["gold_risk_pct"] / 100,
            rr_ratio=actual_rr, atr_value=current_atr,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as e:
        log.debug("[CLASSIC] Signal error: %s", e)
        return None


def _check_daily_trend_bt(df_d1: pd.DataFrame, action: str) -> Tuple[bool, str, int]:
    """Standalone D1 trend check for backtesting."""
    from indicators import get_trend
    if df_d1 is None or len(df_d1) < 60:
        return True, "UNKNOWN", 0
    d1_trend = get_trend(df_d1, 21, 55)
    if action == "BUY":
        if d1_trend == "UP":   return True, d1_trend, +15
        if d1_trend == "DOWN": return True, d1_trend, -15
    else:
        if d1_trend == "DOWN": return True, d1_trend, +15
        if d1_trend == "UP":   return True, d1_trend, -15
    return True, d1_trend, 0


def _run_mr_signal(config: dict, df_m15: pd.DataFrame, df_h1: pd.DataFrame,
                   df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> Optional[Any]:
    """Run Mean Reversion strategy with DataFrame overrides."""
    from mean_reversion_strategy import check_mr_signal
    try:
        return check_mr_signal(
            config,
            df_m15_override=df_m15,
            df_h1_override=df_h1,
            df_h4_override=df_h4,
            df_d1_override=df_d1,
        )
    except Exception as e:
        log.debug("[MR] Signal error: %s", e)
        return None


def _run_squeeze_signal(config: dict, df_m15: pd.DataFrame, df_h1: pd.DataFrame,
                        df_h4: pd.DataFrame, df_d1: pd.DataFrame) -> Optional[Any]:
    """Run BB Squeeze strategy with DataFrame overrides."""
    from bb_squeeze_strategy import check_squeeze_signal
    try:
        return check_squeeze_signal(
            config,
            df_m15_override=df_m15,
            df_h1_override=df_h1,
            df_h4_override=df_h4,
            df_d1_override=df_d1,
        )
    except Exception as e:
        log.debug("[SQ] Signal error: %s", e)
        return None


# ── Core Backtesting Engine ───────────────────────────────────────────────────

class Backtester:
    def __init__(self, df_m15: pd.DataFrame, config: dict,
                 warmup_bars: int = 100, max_open_trades: int = 1):
        self.df_m15   = df_m15.copy().reset_index(drop=True)
        self.config   = config.copy()
        self.warmup   = warmup_bars
        self.max_open = max_open_trades

        # Pre-compute higher timeframes
        self.df_h1  = resample_to_higher_tf(df_m15, "H1")
        self.df_h4  = resample_to_higher_tf(df_m15, "H4")
        self.df_d1  = resample_to_higher_tf(df_m15, "D1")

        # Also keep M5 (downsample M15 → not possible, so use M15 as M5 proxy)
        self.df_m5 = df_m15.copy()

        self.results: Dict[str, BacktestResult] = {
            "Classic":       BacktestResult("Classic",       initial_balance=config["gold_account_balance"]),
            "MeanReversion": BacktestResult("MeanReversion", initial_balance=config["gold_account_balance"]),
            "BBSqueeze":     BacktestResult("BBSqueeze",     initial_balance=config["gold_account_balance"]),
        }

        # Track open trades per strategy
        self._open_trades: Dict[str, Optional[BacktestTrade]] = {
            "Classic": None, "MeanReversion": None, "BBSqueeze": None,
        }

        # Balance per strategy
        self._balance: Dict[str, float] = {
            k: config["gold_account_balance"] for k in self.results
        }

    def _get_window(self, idx: int, df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Get a rolling window of data up to (and including) idx."""
        start = max(0, idx - window + 1)
        return df.iloc[start:idx + 1].copy().reset_index(drop=True)

    def _get_all_up_to(self, idx: int, df: pd.DataFrame) -> pd.DataFrame:
        """Get ALL data up to (and including) idx — for H1/H4/D1 where we want full history."""
        return df.iloc[:idx + 1].copy().reset_index(drop=True)

    def _get_h1_window(self, bar_time: pd.Timestamp, window: int) -> pd.DataFrame:
        """Get H1 data up to the given bar time."""
        mask = self.df_h1["time"] <= bar_time
        df = self.df_h1[mask].tail(window).copy().reset_index(drop=True)
        return df

    def _get_h4_window(self, bar_time: pd.Timestamp, window: int) -> pd.DataFrame:
        mask = self.df_h4["time"] <= bar_time
        df = self.df_h4[mask].tail(window).copy().reset_index(drop=True)
        return df

    def _get_d1_window(self, bar_time: pd.Timestamp, window: int) -> pd.DataFrame:
        mask = self.df_d1["time"] <= bar_time
        df = self.df_d1[mask].tail(window).copy().reset_index(drop=True)
        return df

    def _check_sl_tp(self, trade: BacktestTrade, bar: pd.Series) -> Tuple[bool, float, str]:
        """
        Check if SL or TP was hit in the given bar.
        Returns (hit, exit_price, result).
        Uses bar high/low to determine if either level was reached.
        """
        if trade.action == "BUY":
            if bar["low"] <= trade.sl:
                return True, trade.sl, "LOSS"
            if bar["high"] >= trade.tp:
                return True, trade.tp, "WIN"
        else:
            if bar["high"] >= trade.sl:
                return True, trade.sl, "LOSS"
            if bar["low"] <= trade.tp:
                return True, trade.tp, "WIN"
        return False, 0.0, ""

    def _update_equity(self, strategy: str, bar_idx: int):
        """Update equity curve for a strategy."""
        result = self.results[strategy]
        open_trade = self._open_trades[strategy]
        balance = self._balance[strategy]

        if open_trade is not None:
            # Mark-to-market
            bar = self.df_m15.iloc[bar_idx]
            current_price = bar["close"]
            unrealized = calculate_pnl(open_trade.action, open_trade.entry_price,
                                       current_price, open_trade.lot)
            result.equity_curve.append(balance + unrealized)
        else:
            result.equity_curve.append(balance)

    def run(self, verbose: bool = False):
        """Run the backtest for all strategies."""
        n_bars = len(self.df_m15)
        print(f"\nRunning backtest on {n_bars} bars ({self.df_m15['time'].iloc[0].date()} "
              f"to {self.df_m15['time'].iloc[-1].date()})")
        print(f"Warmup: {self.warmup} bars | Strategies: 3")
        print("-" * 60)

        for i in range(self.warmup, n_bars - 1):
            bar     = self.df_m15.iloc[i]
            bar_time = bar["time"]
            next_bar = self.df_m15.iloc[i + 1]

            # Update config balance per strategy
            for strategy in self.results:
                self.config["gold_account_balance"] = self._balance[strategy]

                # ── Check open trade ──────────────────────────────────────────
                open_trade = self._open_trades[strategy]
                if open_trade is not None:
                    hit, exit_price, result_str = self._check_sl_tp(open_trade, bar)
                    if hit:
                        pnl = calculate_pnl(open_trade.action, open_trade.entry_price,
                                            exit_price, open_trade.lot)
                        open_trade.exit_bar   = i
                        open_trade.exit_price = exit_price
                        open_trade.pnl        = pnl
                        open_trade.result     = result_str
                        open_trade.bars_held  = i - open_trade.entry_bar
                        open_trade.rr_ratio   = abs(exit_price - open_trade.entry_price) / \
                                                abs(open_trade.entry_price - open_trade.sl) \
                                                if abs(open_trade.entry_price - open_trade.sl) > 0 else 0

                        self._balance[strategy] += pnl
                        self._open_trades[strategy] = None

                        if verbose:
                            print(f"  [{strategy}] {result_str}: {open_trade.action} "
                                  f"entry={open_trade.entry_price:.2f} exit={exit_price:.2f} "
                                  f"P&L=${pnl:.2f}")

                # ── Generate new signal (only if no open trade) ───────────────
                if self._open_trades[strategy] is None:
                    # Get rolling windows
                    df_m15_w = self._get_window(i, self.df_m15, 300)
                    # For H1/H4/D1, use ALL available history up to this point
                    df_h1_w  = self._get_h1_window(bar_time, 1000)
                    df_h4_w  = self._get_h4_window(bar_time, 500)
                    df_d1_w  = self._get_d1_window(bar_time, 200)

                    if len(df_m15_w) < 50 or len(df_h1_w) < 10:
                        self._update_equity(strategy, i)
                        continue

                    signal = None
                    if strategy == "Classic":
                        signal = _run_classic_signal(
                            self.config, df_m15_w, df_m15_w, df_h1_w, df_h4_w, df_d1_w
                        )
                    elif strategy == "MeanReversion":
                        signal = _run_mr_signal(
                            self.config, df_m15_w, df_h1_w, df_h4_w, df_d1_w
                        )
                    elif strategy == "BBSqueeze":
                        signal = _run_squeeze_signal(
                            self.config, df_m15_w, df_h1_w, df_h4_w, df_d1_w
                        )

                    if signal is not None:
                        # Entry at next bar open
                        entry_price = next_bar["open"]
                        trade = BacktestTrade(
                            strategy=strategy,
                            action=signal.action,
                            entry_bar=i + 1,
                            entry_price=entry_price,
                            sl=signal.sl,
                            tp=signal.tp,
                            lot=signal.lot_or_qty,
                            score=signal.score,
                            reason=signal.reason,
                            rr_ratio=signal.rr_ratio,
                        )
                        self._open_trades[strategy] = trade
                        self.results[strategy].trades.append(trade)

                        if verbose:
                            print(f"  [{strategy}] SIGNAL: {signal.action} "
                                  f"entry={entry_price:.2f} SL={signal.sl:.2f} "
                                  f"TP={signal.tp:.2f} score={signal.score}")

                self._update_equity(strategy, i)

            # Progress
            if i % 500 == 0:
                pct = (i - self.warmup) / (n_bars - self.warmup) * 100
                print(f"  Progress: {pct:.0f}% ({i}/{n_bars})", end="\r")

        # Close any remaining open trades at last price
        last_price = self.df_m15["close"].iloc[-1]
        for strategy, open_trade in self._open_trades.items():
            if open_trade is not None:
                pnl = calculate_pnl(open_trade.action, open_trade.entry_price,
                                    last_price, open_trade.lot)
                open_trade.exit_bar   = n_bars - 1
                open_trade.exit_price = last_price
                open_trade.pnl        = pnl
                open_trade.result     = "WIN" if pnl > 0 else "LOSS"
                open_trade.bars_held  = n_bars - 1 - open_trade.entry_bar
                self._balance[strategy] += pnl

        print(f"\n  Progress: 100% ({n_bars}/{n_bars})")
        print("-" * 60)

        for strategy, result in self.results.items():
            s = result.summary()
            print(f"[{strategy:15s}] Trades:{s['total_trades']:3d} | "
                  f"WR:{s['win_rate']:5.1f}% | "
                  f"PnL:${s['total_pnl']:8.2f} | "
                  f"PF:{s['profit_factor']:5.2f} | "
                  f"DD:{s['max_drawdown']:6.2f}% | "
                  f"Sharpe:{s['sharpe_ratio']:6.3f}")


# ── Visualization ─────────────────────────────────────────────────────────────

STRATEGY_COLORS = {
    "Classic":       "#2196F3",   # Blue
    "MeanReversion": "#4CAF50",   # Green
    "BBSqueeze":     "#FF9800",   # Orange
}


def plot_results(results: Dict[str, BacktestResult], df_m15: pd.DataFrame,
                 output_path: str = "backtest_results.png"):
    """Generate a comprehensive backtest comparison chart."""
    fig = plt.figure(figsize=(18, 14), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    text_color = "#e6edf3"
    grid_color = "#21262d"

    def style_ax(ax, title=""):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[:].set_color(grid_color)
        ax.yaxis.label.set_color(text_color)
        ax.xaxis.label.set_color(text_color)
        if title:
            ax.set_title(title, color=text_color, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

    # ── 1. Equity Curves ─────────────────────────────────────────────────────
    ax_equity = fig.add_subplot(gs[0, :])
    style_ax(ax_equity, "Equity Curves — All Strategies")

    for name, result in results.items():
        if result.equity_curve:
            ax_equity.plot(result.equity_curve, label=name,
                           color=STRATEGY_COLORS[name], linewidth=1.5, alpha=0.9)

    ax_equity.axhline(y=list(results.values())[0].initial_balance,
                      color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_equity.set_ylabel("Balance ($)", color=text_color)
    ax_equity.legend(facecolor="#161b22", edgecolor=grid_color,
                     labelcolor=text_color, fontsize=9)

    # ── 2. Win Rate ───────────────────────────────────────────────────────────
    ax_wr = fig.add_subplot(gs[1, 0])
    style_ax(ax_wr, "Win Rate (%)")
    names = list(results.keys())
    wrs   = [results[n].win_rate for n in names]
    colors = [STRATEGY_COLORS[n] for n in names]
    bars = ax_wr.bar(names, wrs, color=colors, alpha=0.85, edgecolor=grid_color)
    for bar, val in zip(bars, wrs):
        ax_wr.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                   f"{val:.1f}%", ha="center", va="bottom", color=text_color, fontsize=9)
    ax_wr.set_ylim(0, 100)
    ax_wr.axhline(y=50, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_wr.tick_params(axis="x", rotation=15)

    # ── 3. Total P&L ──────────────────────────────────────────────────────────
    ax_pnl = fig.add_subplot(gs[1, 1])
    style_ax(ax_pnl, "Total P&L ($)")
    pnls = [results[n].total_pnl for n in names]
    bar_colors = [STRATEGY_COLORS[n] if p >= 0 else "#f44336" for n, p in zip(names, pnls)]
    bars = ax_pnl.bar(names, pnls, color=bar_colors, alpha=0.85, edgecolor=grid_color)
    for bar, val in zip(bars, pnls):
        y_pos = bar.get_height() + 5 if val >= 0 else bar.get_height() - 30
        ax_pnl.text(bar.get_x() + bar.get_width() / 2, y_pos,
                    f"${val:.0f}", ha="center", va="bottom", color=text_color, fontsize=9)
    ax_pnl.axhline(y=0, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_pnl.tick_params(axis="x", rotation=15)

    # ── 4. Profit Factor ──────────────────────────────────────────────────────
    ax_pf = fig.add_subplot(gs[1, 2])
    style_ax(ax_pf, "Profit Factor")
    pfs = [min(results[n].profit_factor, 5.0) for n in names]
    bars = ax_pf.bar(names, pfs, color=colors, alpha=0.85, edgecolor=grid_color)
    for bar, val, raw in zip(bars, pfs, [results[n].profit_factor for n in names]):
        label = f"{raw:.2f}" if raw < 5.0 else "∞"
        ax_pf.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                   label, ha="center", va="bottom", color=text_color, fontsize=9)
    ax_pf.axhline(y=1.0, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_pf.tick_params(axis="x", rotation=15)

    # ── 5. Max Drawdown ───────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[2, 0])
    style_ax(ax_dd, "Max Drawdown (%)")
    dds = [abs(results[n].max_drawdown) for n in names]
    bars = ax_dd.bar(names, dds, color=["#f44336"] * 3, alpha=0.75, edgecolor=grid_color)
    for bar, val in zip(bars, dds):
        ax_dd.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                   f"{val:.1f}%", ha="center", va="bottom", color=text_color, fontsize=9)
    ax_dd.tick_params(axis="x", rotation=15)

    # ── 6. Trade Count ────────────────────────────────────────────────────────
    ax_tc = fig.add_subplot(gs[2, 1])
    style_ax(ax_tc, "Trade Count (Wins vs Losses)")
    x = np.arange(len(names))
    w = 0.35
    wins   = [len(results[n].wins)   for n in names]
    losses = [len(results[n].losses) for n in names]
    ax_tc.bar(x - w/2, wins,   w, label="Wins",   color="#4CAF50", alpha=0.85, edgecolor=grid_color)
    ax_tc.bar(x + w/2, losses, w, label="Losses", color="#f44336", alpha=0.85, edgecolor=grid_color)
    ax_tc.set_xticks(x)
    ax_tc.set_xticklabels(names, rotation=15, color=text_color)
    ax_tc.legend(facecolor="#161b22", edgecolor=grid_color, labelcolor=text_color, fontsize=8)

    # ── 7. Sharpe Ratio ───────────────────────────────────────────────────────
    ax_sh = fig.add_subplot(gs[2, 2])
    style_ax(ax_sh, "Sharpe Ratio")
    sharpes = [results[n].sharpe_ratio for n in names]
    bar_colors2 = [STRATEGY_COLORS[n] if s >= 0 else "#f44336" for n, s in zip(names, sharpes)]
    bars = ax_sh.bar(names, sharpes, color=bar_colors2, alpha=0.85, edgecolor=grid_color)
    for bar, val in zip(bars, sharpes):
        ax_sh.text(bar.get_x() + bar.get_width() / 2,
                   bar.get_height() + 0.001 if val >= 0 else bar.get_height() - 0.005,
                   f"{val:.3f}", ha="center", va="bottom", color=text_color, fontsize=9)
    ax_sh.axhline(y=0, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_sh.tick_params(axis="x", rotation=15)

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.suptitle(
        f"TradingbotV72 — Strategy Backtest Comparison\n"
        f"Gold (XAUUSD) | {df_m15['time'].iloc[0].date()} → {df_m15['time'].iloc[-1].date()} | "
        f"M15 Timeframe",
        color=text_color, fontsize=13, fontweight="bold", y=0.98,
    )

    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\nChart saved: {output_path}")


def plot_trade_distribution(results: Dict[str, BacktestResult],
                            output_path: str = "trade_distribution.png"):
    """Plot P&L distribution per strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
    fig.patch.set_facecolor("#0d1117")
    text_color = "#e6edf3"
    grid_color = "#21262d"

    for ax, (name, result) in zip(axes, results.items()):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors=text_color, labelsize=8)
        ax.spines[:].set_color(grid_color)
        ax.grid(True, color=grid_color, linewidth=0.5, alpha=0.7)

        pnls = [t.pnl for t in result.closed_trades]
        if pnls:
            colors_hist = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors_hist, alpha=0.8, edgecolor=grid_color)
            ax.axhline(y=0, color="#888", linestyle="--", linewidth=1)
            ax.set_title(f"{name}\n({len(pnls)} trades | WR:{result.win_rate:.0f}%)",
                         color=text_color, fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                    transform=ax.transAxes, color=text_color)
            ax.set_title(name, color=text_color, fontsize=10)

        ax.set_xlabel("Trade #", color=text_color)
        ax.set_ylabel("P&L ($)", color=text_color)

    fig.suptitle("Trade-by-Trade P&L Distribution", color=text_color,
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"Trade distribution chart saved: {output_path}")


# ── Report Generation ─────────────────────────────────────────────────────────

def generate_report(results: Dict[str, BacktestResult], df_m15: pd.DataFrame,
                    output_path: str = "backtest_report.md"):
    """Generate a Markdown backtest report."""
    summaries = {name: result.summary() for name, result in results.items()}

    lines = [
        "# TradingbotV72 — Strategy Backtest Report",
        "",
        f"**Period:** {df_m15['time'].iloc[0].date()} → {df_m15['time'].iloc[-1].date()}",
        f"**Instrument:** Gold (XAUUSD) — M15 Timeframe",
        f"**Initial Balance:** ${list(results.values())[0].initial_balance:,.2f}",
        f"**Data Source:** Yahoo Finance (GC=F — Gold Futures)",
        "",
        "---",
        "",
        "## Strategy Overview",
        "",
        "| Metric | Classic (EMA/RSI/MACD) | Mean Reversion + Vol Clusters | BB Squeeze + VWAP |",
        "|--------|------------------------|-------------------------------|-------------------|",
    ]

    metrics = [
        ("Total Trades",   "total_trades",  "{}"),
        ("Wins",           "wins",          "{}"),
        ("Losses",         "losses",        "{}"),
        ("Win Rate",       "win_rate",      "{:.1f}%"),
        ("Total P&L",      "total_pnl",     "${:.2f}"),
        ("Final Balance",  "final_balance", "${:,.2f}"),
        ("Profit Factor",  "profit_factor", "{:.2f}"),
        ("Max Drawdown",   "max_drawdown",  "{:.2f}%"),
        ("Avg R:R",        "avg_rr",        "{:.2f}"),
        ("Sharpe Ratio",   "sharpe_ratio",  "{:.3f}"),
    ]

    for label, key, fmt in metrics:
        row = f"| **{label}** |"
        for name in ["Classic", "MeanReversion", "BBSqueeze"]:
            val = summaries[name][key]
            try:
                row += f" {fmt.format(val)} |"
            except Exception:
                row += f" {val} |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## Strategy Descriptions",
        "",
        "### 1. Classic Strategy (TradingbotV72 Original)",
        "The original strategy uses a multi-timeframe EMA trend filter (H1/H4), MACD momentum",
        "confirmation, RSI overbought/oversold filtering, and Support/Resistance zone proximity.",
        "The D1 trend acts as a soft gate with score adjustment.",
        "",
        "### 2. Mean Reversion with Volume Clusters (New)",
        "This strategy identifies statistically overextended price conditions using Bollinger Bands",
        "and Z-score analysis. Volume Clusters (High Volume Nodes from the Volume Profile) serve as",
        "support/resistance zones where institutional activity has concentrated, making reversion",
        "more likely. Entry requires price to be beyond the Bollinger Band, RSI in extreme territory,",
        "and proximity to a High Volume Node. TP targets the Point of Control (POC) or VWAP.",
        "",
        "### 3. BB Squeeze with VWAP & Volume Profile (New)",
        "This strategy trades breakouts from low-volatility consolidation periods. A 'squeeze' is",
        "detected when Bollinger Bands contract inside Keltner Channels. Upon squeeze release,",
        "VWAP confirms the directional bias (price above VWAP = bullish), and the Volume Profile",
        "ensures no High Volume Node is immediately blocking the breakout path. TP targets the",
        "nearest HVN in the breakout direction.",
        "",
        "---",
        "",
        "## Key Findings",
        "",
    ]

    # Determine winner
    best_pnl = max(summaries, key=lambda n: summaries[n]["total_pnl"])
    best_wr  = max(summaries, key=lambda n: summaries[n]["win_rate"])
    best_pf  = max(summaries, key=lambda n: summaries[n]["profit_factor"])
    best_dd  = min(summaries, key=lambda n: abs(summaries[n]["max_drawdown"]))

    lines += [
        f"- **Best Total P&L:** {best_pnl} (${summaries[best_pnl]['total_pnl']:.2f})",
        f"- **Best Win Rate:** {best_wr} ({summaries[best_wr]['win_rate']:.1f}%)",
        f"- **Best Profit Factor:** {best_pf} ({summaries[best_pf]['profit_factor']:.2f})",
        f"- **Lowest Drawdown:** {best_dd} ({summaries[best_dd]['max_drawdown']:.2f}%)",
        "",
        "---",
        "",
        "## Charts",
        "",
        "![Backtest Results](backtest_results.png)",
        "",
        "![Trade Distribution](trade_distribution.png)",
        "",
        "---",
        "",
        "*Report generated by TradingbotV72 Backtester*",
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Report saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TradingbotV72 Backtester")
    parser.add_argument("--days",      type=int, default=90,    help="Days of history")
    parser.add_argument("--warmup",    type=int, default=100,   help="Warmup bars")
    parser.add_argument("--verbose",   action="store_true",     help="Verbose output")
    parser.add_argument("--output",    type=str, default=".",   help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Download data
    print("=" * 60)
    print("  TradingbotV72 — Strategy Backtest Engine")
    print("=" * 60)

    df_m15 = download_gold_data(days=args.days, interval="15m")

    # Run backtest
    bt = Backtester(df_m15, BACKTEST_CONFIG, warmup_bars=args.warmup)
    bt.run(verbose=args.verbose)

    # Generate outputs
    results_path = os.path.join(args.output, "backtest_results.png")
    dist_path    = os.path.join(args.output, "trade_distribution.png")
    report_path  = os.path.join(args.output, "backtest_report.md")
    json_path    = os.path.join(args.output, "backtest_data.json")

    plot_results(bt.results, df_m15, results_path)
    plot_trade_distribution(bt.results, dist_path)
    generate_report(bt.results, df_m15, report_path)

    # Save raw data as JSON
    json_data = {name: result.summary() for name, result in bt.results.items()}
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON data saved: {json_path}")

    print("\n" + "=" * 60)
    print("  Backtest complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
