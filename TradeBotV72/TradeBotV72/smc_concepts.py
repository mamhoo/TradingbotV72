"""
smc_concepts.py — Smart Money Concepts (SMC) Detection Engine

Implements the four core SMC building blocks:
  1. Market Structure  — BOS (Break of Structure) and CHoCH (Change of Character)
  2. Order Blocks (OB) — Last bearish candle before bullish impulse (demand OB)
                         Last bullish candle before bearish impulse (supply OB)
  3. Fair Value Gaps   — 3-candle imbalance where candle[i-1].high < candle[i+1].low (bull FVG)
                         or candle[i-1].low > candle[i+1].high (bear FVG)
  4. Liquidity Levels  — Equal highs/lows and swing point clusters (stop-hunt magnets)

Design principles:
  - Pure pandas/numpy — no external dependencies
  - Each function returns a typed dataclass or list, not raw dicts
  - All functions accept a standard OHLCV DataFrame (time, open, high, low, close, volume)
  - Compatible with botV6 signal pipeline — outputs plug into score_signal()
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class MarketStructure:
    trend: str              # "BULLISH" / "BEARISH" / "RANGING"
    last_event: str         # "BOS_UP" / "BOS_DOWN" / "CHOCH_UP" / "CHOCH_DOWN" / "NONE"
    last_event_price: float # price level of the last BOS/CHoCH
    last_event_idx: int     # bar index of the last event
    swing_highs: List[Tuple[int, float]] = field(default_factory=list)
    swing_lows:  List[Tuple[int, float]] = field(default_factory=list)

    def is_bullish(self)  -> bool: return self.trend == "BULLISH"
    def is_bearish(self)  -> bool: return self.trend == "BEARISH"
    def just_broke_up(self)   -> bool: return self.last_event in ("BOS_UP",   "CHOCH_UP")
    def just_broke_down(self) -> bool: return self.last_event in ("BOS_DOWN", "CHOCH_DOWN")
    def is_choch(self)    -> bool: return "CHOCH" in self.last_event


@dataclass
class OrderBlock:
    ob_type: str            # "DEMAND" (bullish OB) or "SUPPLY" (bearish OB)
    high: float
    low: float
    price: float            # midpoint
    origin_idx: int         # bar index where OB formed
    is_fresh: bool          # True if price has not yet returned to mitigate it
    strength: int           # 0-100 based on impulse size and volume
    volume_at_origin: float # volume of the OB candle

    def __repr__(self):
        return f"OB({self.ob_type} {self.low:.2f}-{self.high:.2f} fresh={self.is_fresh} str={self.strength})"


@dataclass
class FairValueGap:
    gap_type: str           # "BULL_FVG" or "BEAR_FVG"
    high: float             # top of gap
    low: float              # bottom of gap
    price: float            # midpoint
    origin_idx: int         # middle candle index
    is_filled: bool         # True if price has traded through the gap
    fill_pct: float         # 0.0–1.0, how much of the gap has been filled

    def __repr__(self):
        return f"FVG({self.gap_type} {self.low:.2f}-{self.high:.2f} filled={self.fill_pct:.0%})"


@dataclass
class LiquidityLevel:
    liq_type: str           # "BSL" (buy-side) or "SSL" (sell-side)
    price: float            # the liquidity level price
    origin_idx: int
    touches: int            # how many times price tested this level
    swept: bool             # True if price has taken out this liquidity

    def __repr__(self):
        return f"LIQ({self.liq_type} @ {self.price:.2f} swept={self.swept})"


@dataclass
class SMCContext:
    """All SMC data for one symbol/timeframe scan — passed into the signal scorer."""
    structure:   MarketStructure
    demand_obs:  List[OrderBlock]      # bullish OBs below current price
    supply_obs:  List[OrderBlock]      # bearish OBs above current price
    bull_fvgs:   List[FairValueGap]    # unfilled bull gaps below (magnet)
    bear_fvgs:   List[FairValueGap]    # unfilled bear gaps above (magnet)
    bsl_levels:  List[LiquidityLevel]  # buy-side liquidity above
    ssl_levels:  List[LiquidityLevel]  # sell-side liquidity below
    current_price: float

    def nearest_demand_ob(self) -> Optional[OrderBlock]:
        below = [ob for ob in self.demand_obs if ob.price < self.current_price and ob.is_fresh]
        return max(below, key=lambda ob: ob.price) if below else None

    def nearest_supply_ob(self) -> Optional[OrderBlock]:
        above = [ob for ob in self.supply_obs if ob.price > self.current_price and ob.is_fresh]
        return min(above, key=lambda ob: ob.price) if above else None

    def nearest_bull_fvg(self) -> Optional[FairValueGap]:
        unfilled = [f for f in self.bull_fvgs if not f.is_filled and f.price < self.current_price]
        return max(unfilled, key=lambda f: f.price) if unfilled else None

    def nearest_bear_fvg(self) -> Optional[FairValueGap]:
        unfilled = [f for f in self.bear_fvgs if not f.is_filled and f.price > self.current_price]
        return min(unfilled, key=lambda f: f.price) if unfilled else None


# ── 1. Market Structure: BOS and CHoCH ────────────────────────────────────────

def find_swing_points_smc(df: pd.DataFrame, window: int = 5) -> Tuple[List, List]:
    """
    Identify swing highs and lows with tolerance for float precision.
    Returns (swing_highs, swing_lows) as lists of (index, price).
    """
    highs = df["high"].values
    lows  = df["low"].values
    n     = len(df)

    swing_highs, swing_lows = [], []

    for i in range(window, n - window):
        wh = highs[i - window:i + window + 1]
        wl = lows[i - window:i + window + 1]
        tol_h = highs[i] * 0.00001
        tol_l = lows[i]  * 0.00001
        if abs(highs[i] - max(wh)) <= tol_h:
            swing_highs.append((i, highs[i]))
        if abs(lows[i]  - min(wl)) <= tol_l:
            swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def analyze_market_structure(df: pd.DataFrame, window: int = 5) -> MarketStructure:
    """
    Determine trend and last BOS/CHoCH event.

    Rules:
    - BOS UP   : price closes above the last confirmed swing HIGH → trend is BULLISH continuation
    - BOS DOWN : price closes below the last confirmed swing LOW  → trend is BEARISH continuation
    - CHoCH UP : in a downtrend, price closes above the last swing HIGH → potential reversal UP
    - CHoCH DN : in an uptrend,  price closes below the last swing LOW  → potential reversal DOWN

    The difference between BOS and CHoCH:
    - BOS = break in the direction of the current trend (continuation)
    - CHoCH = break AGAINST the current trend (reversal signal)
    """
    if len(df) < window * 3:
        return MarketStructure("RANGING", "NONE", 0.0, 0, [], [])

    swing_highs, swing_lows = find_swing_points_smc(df, window)
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure("RANGING", "NONE", 0.0, 0, swing_highs, swing_lows)

    closes = df["close"].values
    n      = len(closes)

    # Build structure by scanning BOS/CHoCH events
    trend       = "RANGING"
    last_event  = "NONE"
    last_ep     = 0.0
    last_ei     = 0

    # Track last confirmed swing high/low
    last_sh_price = swing_highs[-2][1] if len(swing_highs) >= 2 else swing_highs[-1][1]
    last_sl_price = swing_lows[-2][1]  if len(swing_lows)  >= 2 else swing_lows[-1][1]

    # Determine initial trend from older swings
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Simple: compare last two swing highs and lows
        if swing_highs[-1][1] > swing_highs[-2][1] and swing_lows[-1][1] > swing_lows[-2][1]:
            trend = "BULLISH"   # HH + HL
        elif swing_highs[-1][1] < swing_highs[-2][1] and swing_lows[-1][1] < swing_lows[-2][1]:
            trend = "BEARISH"   # LH + LL
        else:
            trend = "RANGING"

    # Check current close against last swing levels
    current_close = closes[-1]

    if current_close > last_sh_price:
        if trend == "BEARISH":
            last_event = "CHOCH_UP"    # was bearish, now breaking above last high = reversal
        else:
            last_event = "BOS_UP"      # continuation
        last_ep = last_sh_price
        last_ei = swing_highs[-1][0]
        trend   = "BULLISH"

    elif current_close < last_sl_price:
        if trend == "BULLISH":
            last_event = "CHOCH_DOWN"  # was bullish, now breaking below last low = reversal
        else:
            last_event = "BOS_DOWN"    # continuation
        last_ep = last_sl_price
        last_ei = swing_lows[-1][0]
        trend   = "BEARISH"

    return MarketStructure(
        trend=trend,
        last_event=last_event,
        last_event_price=last_ep,
        last_event_idx=last_ei,
        swing_highs=swing_highs,
        swing_lows=swing_lows,
    )


# ── 2. Order Blocks ───────────────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame, lookback: int = 50,
                      min_impulse_pct: float = 0.003) -> List[OrderBlock]:
    """
    Detect Order Blocks (OBs).

    Demand OB (bullish): The last BEARISH candle immediately before a strong
    bullish impulse move. Smart money accumulated here.

    Supply OB (bearish): The last BULLISH candle immediately before a strong
    bearish impulse move. Smart money distributed here.

    min_impulse_pct: the impulse after the OB must be at least this % of price
    to qualify (default 0.3% = ~$9 on Gold at $3000).
    """
    df   = df.tail(lookback).copy().reset_index(drop=True)
    obs  = []
    n    = len(df)
    current_price = df["close"].iloc[-1]

    opens  = df["open"].values
    closes = df["close"].values
    highs  = df["high"].values
    lows   = df["low"].values
    vols   = df["volume"].values if "volume" in df.columns else np.ones(n)

    for i in range(1, n - 2):
        # ── Demand OB: bearish candle followed by bullish impulse ──────────
        if closes[i] < opens[i]:   # bearish candle
            # Check for impulse: next 1-3 candles move up strongly
            impulse_high = max(highs[i+1:min(i+4, n)])
            impulse_size = (impulse_high - highs[i]) / closes[i]

            if impulse_size >= min_impulse_pct:
                ob_high = highs[i]
                ob_low  = lows[i]
                ob_mid  = (ob_high + ob_low) / 2

                # Is it fresh? Price hasn't returned to mitigate yet
                future_lows = lows[i+1:]
                is_fresh = not any(l < ob_low for l in future_lows)
                # Partially mitigated check
                min_future = min(future_lows) if len(future_lows) > 0 else ob_low
                fill_depth = max(0.0, ob_high - min_future) / (ob_high - ob_low + 1e-9)
                is_fresh   = fill_depth < 0.5   # mitigated if price went >50% into OB

                strength = _score_ob(impulse_size, vols[i], np.mean(vols), is_fresh)

                obs.append(OrderBlock(
                    ob_type="DEMAND",
                    high=ob_high,
                    low=ob_low,
                    price=ob_mid,
                    origin_idx=i,
                    is_fresh=is_fresh,
                    strength=strength,
                    volume_at_origin=vols[i],
                ))

        # ── Supply OB: bullish candle followed by bearish impulse ──────────
        elif closes[i] > opens[i]:  # bullish candle
            impulse_low  = min(lows[i+1:min(i+4, n)])
            impulse_size = (lows[i] - impulse_low) / closes[i]

            if impulse_size >= min_impulse_pct:
                ob_high = highs[i]
                ob_low  = lows[i]
                ob_mid  = (ob_high + ob_low) / 2

                future_highs = highs[i+1:]
                max_future   = max(future_highs) if len(future_highs) > 0 else ob_high
                fill_depth   = max(0.0, max_future - ob_low) / (ob_high - ob_low + 1e-9)
                is_fresh     = fill_depth < 0.5

                strength = _score_ob(impulse_size, vols[i], np.mean(vols), is_fresh)

                obs.append(OrderBlock(
                    ob_type="SUPPLY",
                    high=ob_high,
                    low=ob_low,
                    price=ob_mid,
                    origin_idx=i,
                    is_fresh=is_fresh,
                    strength=strength,
                    volume_at_origin=vols[i],
                ))

    # Filter to relevant side only, sort by strength
    demand_obs = sorted([ob for ob in obs if ob.ob_type == "DEMAND" and ob.price < current_price],
                        key=lambda o: o.strength, reverse=True)
    supply_obs = sorted([ob for ob in obs if ob.ob_type == "SUPPLY" and ob.price > current_price],
                        key=lambda o: o.strength, reverse=True)

    return demand_obs + supply_obs


def _score_ob(impulse_size: float, ob_vol: float, avg_vol: float, is_fresh: bool) -> int:
    """Score an OB 0-100."""
    impulse_score = min(40, int(impulse_size / 0.001 * 10))   # 10pts per 0.1% impulse
    vol_score     = min(30, int(30 * (ob_vol / max(avg_vol, 1))))
    fresh_score   = 30 if is_fresh else 0
    return min(100, impulse_score + vol_score + fresh_score)


# ── 3. Fair Value Gaps (FVG / Imbalance) ─────────────────────────────────────

def find_fair_value_gaps(df: pd.DataFrame, lookback: int = 50,
                         min_gap_pct: float = 0.001) -> List[FairValueGap]:
    """
    Detect Fair Value Gaps (FVGs / Imbalances).

    Bull FVG: candle[i-1].high < candle[i+1].low
    The gap between them was never traded — price will likely return to fill it.

    Bear FVG: candle[i-1].low > candle[i+1].high
    Same idea going down.

    min_gap_pct: minimum gap size as % of price (default 0.1%).
    """
    df   = df.tail(lookback).copy().reset_index(drop=True)
    fvgs = []
    n    = len(df)
    current_price = df["close"].iloc[-1]

    highs  = df["high"].values
    lows   = df["low"].values
    closes = df["close"].values

    for i in range(1, n - 1):
        # Bull FVG: gap between candle[i-1] high and candle[i+1] low
        gap_low  = highs[i - 1]
        gap_high = lows[i + 1]

        if gap_high > gap_low:  # valid bull gap
            gap_size = (gap_high - gap_low) / closes[i]
            if gap_size >= min_gap_pct:
                # Check fill: has price traded back into the gap?
                future_lows = lows[i + 2:] if i + 2 < n else []
                min_future  = min(future_lows) if len(future_lows) > 0 else gap_high
                if min_future < gap_low:
                    is_filled = True
                    fill_pct  = 1.0
                elif min_future < gap_high:
                    is_filled = False
                    fill_pct  = (gap_high - min_future) / (gap_high - gap_low)
                else:
                    is_filled = False
                    fill_pct  = 0.0

                fvgs.append(FairValueGap(
                    gap_type="BULL_FVG",
                    high=gap_high,
                    low=gap_low,
                    price=(gap_high + gap_low) / 2,
                    origin_idx=i,
                    is_filled=is_filled,
                    fill_pct=fill_pct,
                ))

        # Bear FVG: gap between candle[i-1] low and candle[i+1] high
        gap_high2 = lows[i - 1]
        gap_low2  = highs[i + 1]

        if gap_high2 > gap_low2:  # valid bear gap
            gap_size = (gap_high2 - gap_low2) / closes[i]
            if gap_size >= min_gap_pct:
                future_highs = highs[i + 2:] if i + 2 < n else []
                max_future   = max(future_highs) if len(future_highs) > 0 else gap_low2
                if max_future > gap_high2:
                    is_filled = True
                    fill_pct  = 1.0
                elif max_future > gap_low2:
                    is_filled = False
                    fill_pct  = (max_future - gap_low2) / (gap_high2 - gap_low2)
                else:
                    is_filled = False
                    fill_pct  = 0.0

                fvgs.append(FairValueGap(
                    gap_type="BEAR_FVG",
                    high=gap_high2,
                    low=gap_low2,
                    price=(gap_high2 + gap_low2) / 2,
                    origin_idx=i,
                    is_filled=is_filled,
                    fill_pct=fill_pct,
                ))

    bull_fvgs = sorted(
        [f for f in fvgs if f.gap_type == "BULL_FVG" and not f.is_filled and f.price < current_price],
        key=lambda f: f.price, reverse=True
    )
    bear_fvgs = sorted(
        [f for f in fvgs if f.gap_type == "BEAR_FVG" and not f.is_filled and f.price > current_price],
        key=lambda f: f.price
    )

    return bull_fvgs + bear_fvgs


# ── 4. Liquidity Levels ───────────────────────────────────────────────────────

def find_liquidity_levels(df: pd.DataFrame, lookback: int = 100,
                          tolerance_pct: float = 0.001,
                          min_touches: int = 2) -> List[LiquidityLevel]:
    """
    Identify liquidity pools: clusters of equal highs (buy-side) or equal lows (sell-side).

    Equal highs = stop losses of sellers sitting just above → BSL (buy-side liquidity)
    Equal lows  = stop losses of buyers sitting just below  → SSL (sell-side liquidity)

    These are price magnets — smart money will sweep these before reversing.
    """
    df    = df.tail(lookback).copy().reset_index(drop=True)
    n     = len(df)
    highs = df["high"].values
    lows  = df["low"].values
    current_price = df["close"].iloc[-1]

    # Find swing points for liquidity pools
    swing_highs, swing_lows = find_swing_points_smc(df, window=3)

    liq_levels = []

    # ── Buy-side liquidity: cluster of equal/near highs ───────────────────
    if swing_highs:
        sh_prices = np.array([p for _, p in swing_highs])
        sh_idx    = [i for i, _ in swing_highs]

        # Group into clusters
        visited = [False] * len(sh_prices)
        for i in range(len(sh_prices)):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, len(sh_prices)):
                if not visited[j]:
                    if abs(sh_prices[i] - sh_prices[j]) / sh_prices[i] <= tolerance_pct:
                        cluster.append(j)
                        visited[j] = True

            if len(cluster) >= min_touches:
                cluster_price = np.mean(sh_prices[cluster])
                latest_idx    = max(sh_idx[k] for k in cluster)
                # Check if swept (price has closed above)
                future_closes = df["close"].values[latest_idx + 1:]
                swept = any(c > cluster_price for c in future_closes)

                liq_levels.append(LiquidityLevel(
                    liq_type="BSL",
                    price=cluster_price,
                    origin_idx=latest_idx,
                    touches=len(cluster),
                    swept=swept,
                ))

    # ── Sell-side liquidity: cluster of equal/near lows ───────────────────
    if swing_lows:
        sl_prices = np.array([p for _, p in swing_lows])
        sl_idx    = [i for i, _ in swing_lows]

        visited = [False] * len(sl_prices)
        for i in range(len(sl_prices)):
            if visited[i]:
                continue
            cluster = [i]
            visited[i] = True
            for j in range(i + 1, len(sl_prices)):
                if not visited[j]:
                    if abs(sl_prices[i] - sl_prices[j]) / sl_prices[i] <= tolerance_pct:
                        cluster.append(j)
                        visited[j] = True

            if len(cluster) >= min_touches:
                cluster_price = np.mean(sl_prices[cluster])
                latest_idx    = max(sl_idx[k] for k in cluster)
                future_closes = df["close"].values[latest_idx + 1:]
                swept = any(c < cluster_price for c in future_closes)

                liq_levels.append(LiquidityLevel(
                    liq_type="SSL",
                    price=cluster_price,
                    origin_idx=latest_idx,
                    touches=len(cluster),
                    swept=swept,
                ))

    # Filter to relevant (un-swept), split by side
    bsl = sorted([l for l in liq_levels if l.liq_type == "BSL"
                  and not l.swept and l.price > current_price],
                 key=lambda l: l.price)
    ssl = sorted([l for l in liq_levels if l.liq_type == "SSL"
                  and not l.swept and l.price < current_price],
                 key=lambda l: l.price, reverse=True)

    return bsl + ssl


# ── 5. Full SMC context builder ───────────────────────────────────────────────

def build_smc_context(df: pd.DataFrame,
                      lookback_structure: int = 100,
                      lookback_ob: int = 50,
                      lookback_fvg: int = 50,
                      lookback_liq: int = 100) -> SMCContext:
    """
    Run all SMC detectors and return a unified SMCContext.
    Call this once per scan on your H1 DataFrame.
    """
    current_price = df["close"].iloc[-1]

    structure = analyze_market_structure(df.tail(lookback_structure), window=5)
    all_obs   = find_order_blocks(df, lookback=lookback_ob)
    all_fvgs  = find_fair_value_gaps(df, lookback=lookback_fvg)
    all_liq   = find_liquidity_levels(df, lookback=lookback_liq)

    demand_obs = [ob for ob in all_obs if ob.ob_type == "DEMAND"]
    supply_obs = [ob for ob in all_obs if ob.ob_type == "SUPPLY"]
    bull_fvgs  = [f  for f  in all_fvgs if f.gap_type  == "BULL_FVG"]
    bear_fvgs  = [f  for f  in all_fvgs if f.gap_type  == "BEAR_FVG"]
    bsl        = [l  for l  in all_liq  if l.liq_type  == "BSL"]
    ssl        = [l  for l  in all_liq  if l.liq_type  == "SSL"]

    return SMCContext(
        structure=structure,
        demand_obs=demand_obs,
        supply_obs=supply_obs,
        bull_fvgs=bull_fvgs,
        bear_fvgs=bear_fvgs,
        bsl_levels=bsl,
        ssl_levels=ssl,
        current_price=current_price,
    )


# ── 6. SMC Signal Scorer ──────────────────────────────────────────────────────

def score_smc_signal(ctx: SMCContext, action: str,
                     structure_timeframe: str = "H1") -> Tuple[int, str]:
    """
    Score an SMC setup 0-100. Designed to ADD to or REPLACE the existing
    zone score in gold_strategy.py / crypto_strategy.py.

    Scoring breakdown (total 100):
      Market structure alignment   : 35 pts
      Order Block confluence       : 30 pts
      FVG confluence               : 20 pts
      Liquidity context            : 15 pts

    Returns (score, reason_string)
    """
    score   = 0
    reasons = []

    ms = ctx.structure

    # ── Market structure (35 pts) ─────────────────────────────────────────
    if action == "BUY":
        if ms.trend == "BULLISH":
            score += 25
            reasons.append("MS_BULL")
        elif ms.just_broke_up():
            score += 35
            label = "CHOCH_UP" if ms.is_choch() else "BOS_UP"
            reasons.append(label)
        elif ms.trend == "RANGING":
            score += 10
            reasons.append("MS_RANGE")
        else:
            score += 0
            reasons.append("MS_BEAR_COUNTER")
    else:  # SELL
        if ms.trend == "BEARISH":
            score += 25
            reasons.append("MS_BEAR")
        elif ms.just_broke_down():
            score += 35
            label = "CHOCH_DN" if ms.is_choch() else "BOS_DN"
            reasons.append(label)
        elif ms.trend == "RANGING":
            score += 10
            reasons.append("MS_RANGE")
        else:
            score += 0
            reasons.append("MS_BULL_COUNTER")

    # ── Order Block confluence (30 pts) ───────────────────────────────────
    if action == "BUY":
        ob = ctx.nearest_demand_ob()
        if ob:
            dist_pct = abs(ctx.current_price - ob.price) / ctx.current_price
            if dist_pct <= 0.003:       # price is AT the demand OB
                pts = min(30, int(ob.strength * 0.3))
                score += pts
                reasons.append(f"DEMAND_OB_str{ob.strength}")
            elif dist_pct <= 0.006:     # price is near the demand OB
                pts = min(15, int(ob.strength * 0.15))
                score += pts
                reasons.append(f"DEMAND_OB_near")
        else:
            reasons.append("NO_DEMAND_OB")
    else:
        ob = ctx.nearest_supply_ob()
        if ob:
            dist_pct = abs(ctx.current_price - ob.price) / ctx.current_price
            if dist_pct <= 0.003:
                pts = min(30, int(ob.strength * 0.3))
                score += pts
                reasons.append(f"SUPPLY_OB_str{ob.strength}")
            elif dist_pct <= 0.006:
                pts = min(15, int(ob.strength * 0.15))
                score += pts
                reasons.append(f"SUPPLY_OB_near")
        else:
            reasons.append("NO_SUPPLY_OB")

    # ── FVG confluence (20 pts) ───────────────────────────────────────────
    if action == "BUY":
        fvg = ctx.nearest_bull_fvg()
        if fvg:
            dist_pct = abs(ctx.current_price - fvg.price) / ctx.current_price
            if dist_pct <= 0.004:
                score += 20
                reasons.append(f"BULL_FVG_at")
            elif dist_pct <= 0.008:
                score += 10
                reasons.append(f"BULL_FVG_near")
        else:
            reasons.append("NO_BULL_FVG")
    else:
        fvg = ctx.nearest_bear_fvg()
        if fvg:
            dist_pct = abs(ctx.current_price - fvg.price) / ctx.current_price
            if dist_pct <= 0.004:
                score += 20
                reasons.append(f"BEAR_FVG_at")
            elif dist_pct <= 0.008:
                score += 10
                reasons.append(f"BEAR_FVG_near")
        else:
            reasons.append("NO_BEAR_FVG")

    # ── Liquidity context (15 pts) ────────────────────────────────────────
    # BUY: we want SSL (stop hunt below) to have just been swept = smart money grabbed liquidity
    # SELL: we want BSL (stop hunt above) to have just been swept
    if action == "BUY":
        # Recent SSL sweep = smart money hunted longs below, now reversing up
        recent_swept_ssl = [l for l in ctx.ssl_levels if l.swept]
        if recent_swept_ssl:
            score += 15
            reasons.append("SSL_SWEPT")
        # Or: price heading toward unswept BSL above = target
        elif ctx.bsl_levels:
            score += 5
            reasons.append("BSL_TARGET")
    else:
        recent_swept_bsl = [l for l in ctx.bsl_levels if l.swept]
        if recent_swept_bsl:
            score += 15
            reasons.append("BSL_SWEPT")
        elif ctx.ssl_levels:
            score += 5
            reasons.append("SSL_TARGET")

    return min(100, score), " | ".join(reasons)


# ── 7. Entry refinement: premium/discount zones ───────────────────────────────

def get_premium_discount(df: pd.DataFrame, lookback: int = 50) -> Tuple[str, float]:
    """
    Determine if price is in premium (above 50% of range) or discount (below 50%).

    SMC rule:
    - BUY in discount (below equilibrium) at demand OBs
    - SELL in premium (above equilibrium) at supply OBs

    Returns (zone, equilibrium_price)
    """
    df_range     = df.tail(lookback)
    range_high   = df_range["high"].max()
    range_low    = df_range["low"].min()
    equilibrium  = (range_high + range_low) / 2
    current      = df["close"].iloc[-1]

    if current > equilibrium:
        return "PREMIUM", equilibrium
    elif current < equilibrium:
        return "DISCOUNT", equilibrium
    return "EQUILIBRIUM", equilibrium


def check_entry_confirmation(df_m15: pd.DataFrame, action: str) -> Tuple[bool, str]:
    """
    M15 entry confirmation:
    - BOS on M15 in the direction of trade
    - Or displacement candle (large body, closes well off the low/high)
    Returns (confirmed, reason)
    """
    ms_m15 = analyze_market_structure(df_m15.tail(40), window=3)

    if action == "BUY":
        if ms_m15.just_broke_up():
            return True, f"M15_{ms_m15.last_event}"
        # Displacement: large bullish candle in last 3 bars
        last3 = df_m15.tail(3)
        for _, row in last3.iterrows():
            body  = row["close"] - row["open"]
            range_ = row["high"] - row["low"]
            if body > 0 and range_ > 0 and (body / range_) > 0.6:
                return True, "M15_DISPLACEMENT_BULL"
    else:
        if ms_m15.just_broke_down():
            return True, f"M15_{ms_m15.last_event}"
        last3 = df_m15.tail(3)
        for _, row in last3.iterrows():
            body  = row["open"] - row["close"]
            range_ = row["high"] - row["low"]
            if body > 0 and range_ > 0 and (body / range_) > 0.6:
                return True, "M15_DISPLACEMENT_BEAR"

    return False, "NO_M15_CONFIRM"
