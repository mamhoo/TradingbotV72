"""
run_backtest.py — Self-Contained Backtest Runner for TradingbotV72
Uses Yahoo Finance 1h Gold data (up to 730 days available).
Mocks MetaTrader5 and runs all 3 strategies in parallel.

This script is designed to run standalone without MT5.
"""

import sys
import os
import types
import warnings
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Mock MetaTrader5 ──────────────────────────────────────────────────────────
mt5_mock = types.ModuleType("MetaTrader5")
for attr in ["TIMEFRAME_M1","TIMEFRAME_M5","TIMEFRAME_M15","TIMEFRAME_M30",
             "TIMEFRAME_H1","TIMEFRAME_H4","TIMEFRAME_D1",
             "POSITION_TYPE_BUY","POSITION_TYPE_SELL",
             "ORDER_TYPE_BUY","ORDER_TYPE_SELL",
             "TRADE_ACTION_DEAL","TRADE_ACTION_SLTP",
             "TRADE_RETCODE_DONE","DEAL_ENTRY_OUT"]:
    setattr(mt5_mock, attr, 0)
mt5_mock.initialize = lambda: False
mt5_mock.shutdown   = lambda: None
mt5_mock.account_info = lambda: None
mt5_mock.symbol_info  = lambda s: None
mt5_mock.symbol_info_tick = lambda s: None
mt5_mock.copy_rates_from_pos = lambda *a, **kw: None
mt5_mock.positions_get = lambda **kw: []
mt5_mock.last_error    = lambda: (0, "mock")
sys.modules["MetaTrader5"] = mt5_mock

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── Patch session_config to always return tradeable ──────────────────────────
import session_config as _sc
_MOCK_PARAMS = {"min_score": 50, "max_spread_pips": 80,
                "min_volume_ratio": 1.0, "scan_interval": 60}
_sc.is_tradeable = lambda: (True, "LONDON", _MOCK_PARAMS)
_sc.thai_time_str = lambda: datetime.now().strftime("%H:%M")

import gold_strategy as _gs
_gs.is_tradeable = _sc.is_tradeable
_gs.thai_time_str = _sc.thai_time_str

import mean_reversion_strategy as _mr
_mr.is_tradeable = _sc.is_tradeable
_mr.thai_time_str = _sc.thai_time_str

import bb_squeeze_strategy as _sq
_sq.is_tradeable = _sc.is_tradeable
_sq.thai_time_str = _sc.thai_time_str

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.ERROR)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
CONFIG = {
    "mt5_symbol":            "XAUUSD",
    "gold_account_balance":  10000.0,
    "gold_risk_pct":         0.5,
    "gold_lot_base":         0.01,
    "gold_max_lot":          5.0,
    "gold_min_score":        55,
    "gold_min_volume_ratio": 1.0,
    "gold_max_spread_pips":  80,
    "gold_max_entry_dist_pct": 0.008,
    "gold_sr_zone_pips":     8,
    "gold_ema_fast":         21,
    "gold_ema_slow":         55,
    "gold_rsi_period":       14,
    "gold_volume_filter":    True,
}

OUTPUT_DIR = "/home/ubuntu/backtest_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ── Data Download ─────────────────────────────────────────────────────────────

def download_data(period: str = "1y", interval: str = "1h") -> pd.DataFrame:
    import yfinance as yf
    print(f"Downloading Gold data: period={period}, interval={interval}")
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        raise ValueError("No data from Yahoo Finance")
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={"datetime": "time", "date": "time"}, inplace=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].dropna().reset_index(drop=True)
    df["time"] = pd.to_datetime(df["time"])
    # Add synthetic volume if zero
    if df["volume"].sum() == 0:
        df["volume"] = 1000 + np.random.randint(0, 500, len(df))
    print(f"Downloaded {len(df)} bars: {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}")
    return df


def resample(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    df2 = df.copy().set_index("time")
    r = df2.resample(freq).agg({"open":"first","high":"max","low":"min",
                                 "close":"last","volume":"sum"}).dropna().reset_index()
    return r


# ── Trade dataclass ───────────────────────────────────────────────────────────

@dataclass
class Trade:
    strategy:    str
    action:      str
    entry_bar:   int
    entry_price: float
    sl:          float
    tp:          float
    lot:         float
    score:       int
    exit_bar:    int   = -1
    exit_price:  float = 0.0
    pnl:         float = 0.0
    result:      str   = "OPEN"
    rr_ratio:    float = 0.0


def calc_pnl(action: str, entry: float, exit_p: float, lot: float) -> float:
    """Gold: 1 lot = 100 oz."""
    if action == "BUY":
        return lot * 100 * (exit_p - entry)
    return lot * 100 * (entry - exit_p)


# ── Signal wrappers ───────────────────────────────────────────────────────────

def run_classic(config: dict, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
                df_d1: pd.DataFrame) -> Optional[Any]:
    """Classic strategy using H1 as the primary timeframe."""
    from gold_strategy import (
        check_volume_confirmation, get_action, check_daily_trend,
        check_not_chasing, check_rsi, check_macd, check_zone,
        calculate_dynamic_rr, calculate_lot_size, calculate_partial_tp,
    )
    from indicators import rsi, ema, atr, macd, get_trend
    from sr_zones import build_zones
    from signal_model import Signal

    try:
        if len(df_h1) < 60:
            return None

        current_price = df_h1["close"].iloc[-1]
        current_atr   = atr(df_h1, 14).iloc[-1]

        # Volume (relaxed for H1)
        vol_ok, vol_ratio = check_volume_confirmation(df_h1, 1.0)

        # Action from H1+H4
        action, h1_trend, h4_trend = get_action(df_h1, df_h4, 21, 55)
        if action == "NEUTRAL":
            return None

        # D1 trend
        d1_trend = "UNKNOWN"
        d1_score_adj = 0
        if df_d1 is not None and len(df_d1) >= 30:
            d1_trend_str = get_trend(df_d1, 21, 55)
            d1_trend = d1_trend_str
            if (action == "BUY" and d1_trend == "UP") or (action == "SELL" and d1_trend == "DOWN"):
                d1_score_adj = +15
            elif (action == "BUY" and d1_trend == "DOWN") or (action == "SELL" and d1_trend == "UP"):
                d1_score_adj = -15

        # Anti-chase
        not_chasing, dist_pct = check_not_chasing(df_h1, action, 21, 0.008)
        if not not_chasing:
            return None

        # RSI
        rsi_ok, rsi_label, rsi_val = check_rsi(df_h1, action, 14)
        if not rsi_ok:
            return None

        # MACD
        macd_ok, macd_signal, macd_val = check_macd(df_h1, action)
        if not macd_ok:
            return None

        # Zones
        zones = build_zones(df_h1, lookback=100, min_touches=2, zone_pips=8)
        at_zone, zone_obj, touches = check_zone(zones, current_price, action)

        # Score
        score = d1_score_adj
        reasons = []
        if h1_trend == h4_trend and h1_trend != "NEUTRAL":
            score += 35; reasons.append("H1H4_ALIGNED")
        elif h1_trend != "NEUTRAL":
            score += 20; reasons.append(f"H1_{h1_trend}")
        else:
            score += 10; reasons.append(f"H4_{h4_trend}")

        score += 25 if ("ZERO_CROSS" in macd_signal or "MOMENTUM" in macd_signal) else 15
        reasons.append(macd_signal)

        if rsi_label == "GOOD_ZONE":
            score += 20
        elif "OVERSOLD" in rsi_label or "OVERBOUGHT" in rsi_label:
            score += 15
        else:
            score += 10
        reasons.append(rsi_label)

        if at_zone:
            score += min(20, 5 * touches); reasons.append(f"ZONE_{touches}T")
        else:
            ev = ema(df_h1["close"], 21).iloc[-1]
            d = abs(current_price - ev) / ev
            if d < 0.002: score += 15; reasons.append("EMA_PULLBACK")
            elif d < 0.004: score += 8; reasons.append("EMA_NEAR")
            else: score += 2; reasons.append("NO_ZONE")

        score += 5; reasons.append(f"VOL_{vol_ratio:.1f}x")

        min_score = 65 if d1_score_adj < 0 else 60
        if score < min_score:
            return None

        trend_aligned = h1_trend == h4_trend and h1_trend != "NEUTRAL"
        rr_ratio = calculate_dynamic_rr(touches if at_zone else 0,
                                        h1_trend if trend_aligned else "NEUTRAL", False)

        h1_atr = atr(df_h1, 14).iloc[-1]
        if action == "BUY":
            sl = current_price - h1_atr * 1.2
            tp = current_price + (current_price - sl) * rr_ratio
        else:
            sl = current_price + h1_atr * 1.2
            tp = current_price - (sl - current_price) * rr_ratio

        rr_actual = abs(tp - current_price) / abs(current_price - sl) if abs(current_price - sl) > 0 else 0
        if rr_actual < 1.3:
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
            reason=f"CLASSIC|Score:{score}|{' '.join(reasons)}",
            sr_level=zone_obj.price if zone_obj else current_price,
            sr_type="SUPPORT" if action == "BUY" else "RESISTANCE",
            zone_strength=touches if zone_obj else 0,
            trend_1h=h1_trend, rsi=rsi_val,
            risk_usdt=config["gold_account_balance"] * config["gold_risk_pct"] / 100,
            rr_ratio=rr_actual, atr_value=current_atr,
            timestamp=datetime.now(timezone.utc),
        )
    except Exception as e:
        log.debug("[CLASSIC] %s", e)
        return None


def run_mr(config: dict, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
           df_d1: pd.DataFrame) -> Optional[Any]:
    from mean_reversion_strategy import check_mr_signal
    try:
        return check_mr_signal(config,
                               df_m15_override=df_h1,
                               df_h1_override=df_h1,
                               df_h4_override=df_h4,
                               df_d1_override=df_d1)
    except Exception as e:
        log.debug("[MR] %s", e)
        return None


def run_sq(config: dict, df_h1: pd.DataFrame, df_h4: pd.DataFrame,
           df_d1: pd.DataFrame) -> Optional[Any]:
    from bb_squeeze_strategy import check_squeeze_signal
    try:
        return check_squeeze_signal(config,
                                    df_m15_override=df_h1,
                                    df_h1_override=df_h1,
                                    df_h4_override=df_h4,
                                    df_d1_override=df_d1)
    except Exception as e:
        log.debug("[SQ] %s", e)
        return None


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    def __init__(self, df_h1: pd.DataFrame, config: dict, warmup: int = 100):
        self.df   = df_h1.copy().reset_index(drop=True)
        self.cfg  = config.copy()
        self.wu   = warmup
        self.df_h4 = resample(df_h1, "4h")
        self.df_d1 = resample(df_h1, "1D")

        self.strategies = ["Classic", "MeanReversion", "BBSqueeze"]
        self.trades:   Dict[str, List[Trade]] = {s: [] for s in self.strategies}
        self.equity:   Dict[str, List[float]] = {s: [] for s in self.strategies}
        self.balance:  Dict[str, float]       = {s: config["gold_account_balance"] for s in self.strategies}
        self.open_t:   Dict[str, Optional[Trade]] = {s: None for s in self.strategies}

    def _h4_up_to(self, t: pd.Timestamp) -> pd.DataFrame:
        return self.df_h4[self.df_h4["time"] <= t].copy().reset_index(drop=True)

    def _d1_up_to(self, t: pd.Timestamp) -> pd.DataFrame:
        return self.df_d1[self.df_d1["time"] <= t].copy().reset_index(drop=True)

    def run(self):
        n = len(self.df)
        print(f"\nBacktest: {n} H1 bars | {self.df['time'].iloc[0].date()} → {self.df['time'].iloc[-1].date()}")
        print(f"Warmup: {self.wu} bars | Strategies: {len(self.strategies)}")
        print("-" * 70)

        for i in range(self.wu, n - 1):
            bar      = self.df.iloc[i]
            bar_time = bar["time"]
            next_bar = self.df.iloc[i + 1]

            df_h1_w = self.df.iloc[:i + 1].copy().reset_index(drop=True)
            df_h4_w = self._h4_up_to(bar_time)
            df_d1_w = self._d1_up_to(bar_time)

            for strat in self.strategies:
                self.cfg["gold_account_balance"] = self.balance[strat]

                # Check open trade
                ot = self.open_t[strat]
                if ot is not None:
                    hit, exit_p, res = self._check_exit(ot, bar)
                    if hit:
                        pnl = calc_pnl(ot.action, ot.entry_price, exit_p, ot.lot)
                        ot.exit_bar   = i
                        ot.exit_price = exit_p
                        ot.pnl        = pnl
                        ot.result     = res
                        ot.rr_ratio   = abs(exit_p - ot.entry_price) / max(abs(ot.entry_price - ot.sl), 1e-9)
                        self.balance[strat] += pnl
                        self.open_t[strat]   = None

                # Generate signal
                if self.open_t[strat] is None and len(df_h1_w) >= 60:
                    sig = None
                    if strat == "Classic":
                        sig = run_classic(self.cfg, df_h1_w, df_h4_w, df_d1_w)
                    elif strat == "MeanReversion":
                        sig = run_mr(self.cfg, df_h1_w, df_h4_w, df_d1_w)
                    elif strat == "BBSqueeze":
                        sig = run_sq(self.cfg, df_h1_w, df_h4_w, df_d1_w)

                    if sig is not None:
                        entry_p = next_bar["open"]
                        t = Trade(
                            strategy=strat, action=sig.action,
                            entry_bar=i + 1, entry_price=entry_p,
                            sl=sig.sl, tp=sig.tp, lot=sig.lot_or_qty,
                            score=sig.score, rr_ratio=sig.rr_ratio,
                        )
                        self.open_t[strat] = t
                        self.trades[strat].append(t)

                # Equity mark-to-market
                ot2 = self.open_t[strat]
                if ot2 is not None:
                    unreal = calc_pnl(ot2.action, ot2.entry_price, bar["close"], ot2.lot)
                    self.equity[strat].append(self.balance[strat] + unreal)
                else:
                    self.equity[strat].append(self.balance[strat])

            if i % 200 == 0:
                pct = (i - self.wu) / (n - self.wu) * 100
                print(f"  Progress: {pct:.0f}%  ({i}/{n})", end="\r")

        # Close remaining open trades
        last_p = self.df["close"].iloc[-1]
        for strat in self.strategies:
            ot = self.open_t[strat]
            if ot is not None:
                pnl = calc_pnl(ot.action, ot.entry_price, last_p, ot.lot)
                ot.exit_bar = n - 1; ot.exit_price = last_p
                ot.pnl = pnl; ot.result = "WIN" if pnl > 0 else "LOSS"
                self.balance[strat] += pnl

        print(f"\n  Progress: 100%  ({n}/{n})")
        print("-" * 70)
        self._print_summary()

    def _check_exit(self, t: Trade, bar: pd.Series) -> Tuple[bool, float, str]:
        if t.action == "BUY":
            if bar["low"]  <= t.sl: return True, t.sl, "LOSS"
            if bar["high"] >= t.tp: return True, t.tp, "WIN"
        else:
            if bar["high"] >= t.sl: return True, t.sl, "LOSS"
            if bar["low"]  <= t.tp: return True, t.tp, "WIN"
        return False, 0.0, ""

    def _closed(self, strat: str) -> List[Trade]:
        return [t for t in self.trades[strat] if t.result != "OPEN"]

    def _wins(self, strat: str) -> List[Trade]:
        return [t for t in self._closed(strat) if t.result == "WIN"]

    def _losses(self, strat: str) -> List[Trade]:
        return [t for t in self._closed(strat) if t.result == "LOSS"]

    def win_rate(self, s: str) -> float:
        c = self._closed(s); return len(self._wins(s)) / len(c) * 100 if c else 0.0

    def total_pnl(self, s: str) -> float:
        return sum(t.pnl for t in self._closed(s))

    def profit_factor(self, s: str) -> float:
        gw = sum(t.pnl for t in self._wins(s))
        gl = abs(sum(t.pnl for t in self._losses(s)))
        return gw / gl if gl > 0 else float("inf")

    def max_dd(self, s: str) -> float:
        eq = np.array(self.equity[s])
        if len(eq) == 0: return 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-9) * 100
        return float(dd.min())

    def sharpe(self, s: str) -> float:
        pnls = [t.pnl for t in self._closed(s)]
        if len(pnls) < 2: return 0.0
        return float(np.mean(pnls) / (np.std(pnls) + 1e-9))

    def avg_rr(self, s: str) -> float:
        rrs = [t.rr_ratio for t in self._closed(s) if t.rr_ratio > 0]
        return float(np.mean(rrs)) if rrs else 0.0

    def summary(self, s: str) -> dict:
        init = CONFIG["gold_account_balance"]
        return {
            "strategy":      s,
            "total_trades":  len(self._closed(s)),
            "wins":          len(self._wins(s)),
            "losses":        len(self._losses(s)),
            "win_rate":      round(self.win_rate(s), 1),
            "total_pnl":     round(self.total_pnl(s), 2),
            "profit_factor": round(min(self.profit_factor(s), 99.0), 2),
            "max_drawdown":  round(self.max_dd(s), 2),
            "avg_rr":        round(self.avg_rr(s), 2),
            "sharpe_ratio":  round(self.sharpe(s), 3),
            "final_balance": round(init + self.total_pnl(s), 2),
        }

    def _print_summary(self):
        for s in self.strategies:
            sm = self.summary(s)
            pf = f"{sm['profit_factor']:.2f}" if sm['profit_factor'] < 99 else "∞"
            print(f"[{s:15s}] Trades:{sm['total_trades']:3d} | "
                  f"WR:{sm['win_rate']:5.1f}% | "
                  f"PnL:${sm['total_pnl']:8.2f} | "
                  f"PF:{pf:>5} | "
                  f"DD:{sm['max_drawdown']:6.2f}% | "
                  f"Sharpe:{sm['sharpe_ratio']:6.3f} | "
                  f"Balance:${sm['final_balance']:,.2f}")


# ── Visualization ─────────────────────────────────────────────────────────────

COLORS = {"Classic": "#2196F3", "MeanReversion": "#4CAF50", "BBSqueeze": "#FF9800"}
BG     = "#0d1117"
PANEL  = "#161b22"
GRID   = "#21262d"
TEXT   = "#e6edf3"


def make_charts(bt: Backtester, df: pd.DataFrame):
    fig = plt.figure(figsize=(20, 15), facecolor=BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.50, wspace=0.38)

    def sax(ax, title=""):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.yaxis.label.set_color(TEXT)
        ax.xaxis.label.set_color(TEXT)
        if title: ax.set_title(title, color=TEXT, fontsize=10, fontweight="bold", pad=8)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # ── Equity curves ─────────────────────────────────────────────────────────
    ax_eq = fig.add_subplot(gs[0, :])
    sax(ax_eq, "Equity Curves — All Strategies")
    init_bal = CONFIG["gold_account_balance"]
    for s in bt.strategies:
        if bt.equity[s]:
            ax_eq.plot(bt.equity[s], label=s, color=COLORS[s], linewidth=1.8, alpha=0.9)
    ax_eq.axhline(y=init_bal, color="#666", linestyle="--", linewidth=1, alpha=0.5)
    ax_eq.set_ylabel("Balance ($)", color=TEXT)
    ax_eq.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=10)

    names = bt.strategies

    # ── Win Rate ──────────────────────────────────────────────────────────────
    ax_wr = fig.add_subplot(gs[1, 0])
    sax(ax_wr, "Win Rate (%)")
    wrs = [bt.win_rate(s) for s in names]
    bars = ax_wr.bar(names, wrs, color=[COLORS[s] for s in names], alpha=0.85, edgecolor=GRID)
    for b, v in zip(bars, wrs):
        ax_wr.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                   f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_wr.set_ylim(0, 100)
    ax_wr.axhline(50, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax_wr.tick_params(axis="x", rotation=15)

    # ── Total P&L ─────────────────────────────────────────────────────────────
    ax_pnl = fig.add_subplot(gs[1, 1])
    sax(ax_pnl, "Total P&L ($)")
    pnls = [bt.total_pnl(s) for s in names]
    bc   = [COLORS[s] if p >= 0 else "#f44336" for s, p in zip(names, pnls)]
    bars = ax_pnl.bar(names, pnls, color=bc, alpha=0.85, edgecolor=GRID)
    for b, v in zip(bars, pnls):
        ax_pnl.text(b.get_x() + b.get_width()/2, b.get_height() + (10 if v >= 0 else -50),
                    f"${v:.0f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_pnl.axhline(0, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax_pnl.tick_params(axis="x", rotation=15)

    # ── Profit Factor ─────────────────────────────────────────────────────────
    ax_pf = fig.add_subplot(gs[1, 2])
    sax(ax_pf, "Profit Factor")
    pfs = [min(bt.profit_factor(s), 5.0) for s in names]
    raw_pfs = [bt.profit_factor(s) for s in names]
    bars = ax_pf.bar(names, pfs, color=[COLORS[s] for s in names], alpha=0.85, edgecolor=GRID)
    for b, v, rv in zip(bars, pfs, raw_pfs):
        lbl = f"{rv:.2f}" if rv < 99 else "∞"
        ax_pf.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05,
                   lbl, ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_pf.axhline(1.0, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax_pf.tick_params(axis="x", rotation=15)

    # ── Max Drawdown ──────────────────────────────────────────────────────────
    ax_dd = fig.add_subplot(gs[2, 0])
    sax(ax_dd, "Max Drawdown (%)")
    dds = [abs(bt.max_dd(s)) for s in names]
    bars = ax_dd.bar(names, dds, color=["#f44336"]*3, alpha=0.75, edgecolor=GRID)
    for b, v in zip(bars, dds):
        ax_dd.text(b.get_x() + b.get_width()/2, b.get_height() + 0.1,
                   f"{v:.1f}%", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_dd.tick_params(axis="x", rotation=15)

    # ── Trade Count ───────────────────────────────────────────────────────────
    ax_tc = fig.add_subplot(gs[2, 1])
    sax(ax_tc, "Trade Count (Wins vs Losses)")
    x = np.arange(len(names)); w = 0.35
    wins   = [len(bt._wins(s))   for s in names]
    losses = [len(bt._losses(s)) for s in names]
    ax_tc.bar(x - w/2, wins,   w, label="Wins",   color="#4CAF50", alpha=0.85, edgecolor=GRID)
    ax_tc.bar(x + w/2, losses, w, label="Losses", color="#f44336", alpha=0.85, edgecolor=GRID)
    ax_tc.set_xticks(x); ax_tc.set_xticklabels(names, rotation=15, color=TEXT)
    ax_tc.legend(facecolor=PANEL, edgecolor=GRID, labelcolor=TEXT, fontsize=8)

    # ── Sharpe Ratio ──────────────────────────────────────────────────────────
    ax_sh = fig.add_subplot(gs[2, 2])
    sax(ax_sh, "Sharpe Ratio")
    sharpes = [bt.sharpe(s) for s in names]
    bc2 = [COLORS[s] if v >= 0 else "#f44336" for s, v in zip(names, sharpes)]
    bars = ax_sh.bar(names, sharpes, color=bc2, alpha=0.85, edgecolor=GRID)
    for b, v in zip(bars, sharpes):
        ax_sh.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                   f"{v:.3f}", ha="center", va="bottom", color=TEXT, fontsize=9)
    ax_sh.axhline(0, color="#888", linestyle="--", linewidth=1, alpha=0.5)
    ax_sh.tick_params(axis="x", rotation=15)

    fig.suptitle(
        f"TradingbotV72 — Strategy Backtest Comparison\n"
        f"Gold (XAUUSD) | {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()} | H1 Timeframe",
        color=TEXT, fontsize=13, fontweight="bold", y=0.99,
    )

    out = os.path.join(OUTPUT_DIR, "backtest_results.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Chart saved: {out}")
    return out


def make_pnl_chart(bt: Backtester) -> str:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=BG)
    fig.patch.set_facecolor(BG)

    for ax, s in zip(axes, bt.strategies):
        ax.set_facecolor(PANEL)
        ax.tick_params(colors=TEXT, labelsize=8)
        ax.spines[:].set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

        pnls = [t.pnl for t in bt._closed(s)]
        if pnls:
            colors = ["#4CAF50" if p > 0 else "#f44336" for p in pnls]
            ax.bar(range(len(pnls)), pnls, color=colors, alpha=0.8, edgecolor=GRID)
            ax.axhline(0, color="#888", linestyle="--", linewidth=1)
            ax.set_title(f"{s}\n({len(pnls)} trades | WR:{bt.win_rate(s):.0f}%)",
                         color=TEXT, fontsize=10, fontweight="bold")
        else:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center",
                    transform=ax.transAxes, color=TEXT, fontsize=12)
            ax.set_title(s, color=TEXT, fontsize=10)
        ax.set_xlabel("Trade #", color=TEXT)
        ax.set_ylabel("P&L ($)", color=TEXT)

    fig.suptitle("Trade-by-Trade P&L Distribution", color=TEXT, fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, "trade_distribution.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    print(f"Trade distribution chart saved: {out}")
    return out


# ── Report ────────────────────────────────────────────────────────────────────

def make_report(bt: Backtester, df: pd.DataFrame) -> str:
    sums = {s: bt.summary(s) for s in bt.strategies}

    lines = [
        "# TradingbotV72 — Strategy Backtest Report",
        "",
        f"**Period:** {df['time'].iloc[0].date()} → {df['time'].iloc[-1].date()}",
        f"**Instrument:** Gold (XAUUSD) — H1 Timeframe (Yahoo Finance GC=F)",
        f"**Initial Balance:** ${CONFIG['gold_account_balance']:,.2f}",
        f"**Risk per Trade:** {CONFIG['gold_risk_pct']}% of balance",
        "",
        "---",
        "",
        "## Performance Summary",
        "",
        "| Metric | Classic (EMA/RSI/MACD) | Mean Reversion + Vol Clusters | BB Squeeze + VWAP |",
        "|--------|:----------------------:|:-----------------------------:|:-----------------:|",
    ]

    metrics = [
        ("Total Trades",  "total_trades",  "{}"),
        ("Wins / Losses", None,            None),
        ("Win Rate",      "win_rate",      "{:.1f}%"),
        ("Total P&L",     "total_pnl",     "${:.2f}"),
        ("Final Balance", "final_balance", "${:,.2f}"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Max Drawdown",  "max_drawdown",  "{:.2f}%"),
        ("Avg R:R",       "avg_rr",        "{:.2f}"),
        ("Sharpe Ratio",  "sharpe_ratio",  "{:.3f}"),
    ]

    for label, key, fmt in metrics:
        if key is None:
            row = f"| **{label}** |"
            for s in ["Classic", "MeanReversion", "BBSqueeze"]:
                row += f" {sums[s]['wins']}W / {sums[s]['losses']}L |"
        else:
            row = f"| **{label}** |"
            for s in ["Classic", "MeanReversion", "BBSqueeze"]:
                val = sums[s][key]
                try:
                    row += f" {fmt.format(val)} |"
                except Exception:
                    row += f" {val} |"
        lines.append(row)

    best_pnl = max(sums, key=lambda s: sums[s]["total_pnl"])
    best_wr  = max(sums, key=lambda s: sums[s]["win_rate"])
    best_pf  = max(sums, key=lambda s: sums[s]["profit_factor"])
    best_dd  = min(sums, key=lambda s: abs(sums[s]["max_drawdown"]))

    lines += [
        "",
        "---",
        "",
        "## Strategy Descriptions",
        "",
        "### 1. Classic Strategy (TradingbotV72 Original)",
        "",
        "The original strategy uses a multi-timeframe EMA trend filter (H1/H4), MACD momentum",
        "confirmation, RSI overbought/oversold filtering, and Support/Resistance zone proximity.",
        "The D1 trend acts as a soft gate with a score adjustment of ±15 points. This is a",
        "trend-following strategy that performs best in directional markets.",
        "",
        "### 2. Mean Reversion with Volume Clusters (New)",
        "",
        "This strategy identifies statistically overextended price conditions using Bollinger Bands",
        "and Z-score analysis. Volume Clusters (High Volume Nodes from the Volume Profile) serve as",
        "support/resistance zones where institutional activity has concentrated, making reversion",
        "more likely. Entry requires price to be beyond the Bollinger Band, RSI in extreme territory,",
        "and proximity to a High Volume Node. TP targets the Point of Control (POC) or VWAP.",
        "This strategy is selective by design — it only fires when multiple extreme conditions align.",
        "",
        "### 3. BB Squeeze with VWAP & Volume Profile (New)",
        "",
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
        f"- **Best Total P&L:** {best_pnl} (${sums[best_pnl]['total_pnl']:.2f})",
        f"- **Best Win Rate:** {best_wr} ({sums[best_wr]['win_rate']:.1f}%)",
        f"- **Best Profit Factor:** {best_pf} ({sums[best_pf]['profit_factor']:.2f})",
        f"- **Lowest Drawdown:** {best_dd} ({sums[best_dd]['max_drawdown']:.2f}%)",
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
        "*Report generated by TradingbotV72 Backtester — Manus AI*",
    ]

    out = os.path.join(OUTPUT_DIR, "backtest_report.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Report saved: {out}")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("  TradingbotV72 — Strategy Backtest Engine")
    print("=" * 70)

    df_h1 = download_data(period="1y", interval="1h")

    bt = Backtester(df_h1, CONFIG, warmup=100)
    bt.run()

    make_charts(bt, df_h1)
    make_pnl_chart(bt)
    make_report(bt, df_h1)

    # Save JSON
    json_out = os.path.join(OUTPUT_DIR, "backtest_data.json")
    with open(json_out, "w") as f:
        json.dump({s: bt.summary(s) for s in bt.strategies}, f, indent=2)
    print(f"JSON data saved: {json_out}")

    print("\n" + "=" * 70)
    print("  Backtest complete! Results in:", OUTPUT_DIR)
    print("=" * 70)
