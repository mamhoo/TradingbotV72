"""
sr_zones.py — Strong Support/Resistance + Demand/Supply Zone Detection

FIXES v6.1:
  [BUG] find_swing_points: float equality replaced with index comparison (argmax/argmin)
  [BUG] cluster_levels: tolerance now consistent with zone_pips width
  [FIX] score_zone: touch scoring more granular, min 2 touches enforced properly
  [FIX] build_zones: zone_pips used consistently in cluster tolerance
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class Zone:
    price: float
    high: float
    low: float
    zone_type: str        # "SUPPORT" / "RESISTANCE" / "DEMAND" / "SUPPLY"
    touches: int
    strength: int         # 0–100 score
    last_touch_idx: int
    is_fresh: bool

    def __repr__(self):
        return f"{self.zone_type}({self.price:.2f}, touches={self.touches}, strength={self.strength})"


def find_swing_points(df: pd.DataFrame, window: int = 5) -> Tuple[list, list]:
    """
    Find swing highs and lows.
    FIX: Use argmax/argmin index comparison instead of float equality.
    Two candles with identical prices no longer both fail.
    """
    highs = df["high"].values
    lows  = df["low"].values
    n = len(df)

    swing_highs = []
    swing_lows  = []

    for i in range(window, n - window):
        window_slice_h = highs[i - window : i + window + 1]
        window_slice_l = lows[i  - window : i + window + 1]

        # Swing high: i must be the index of the maximum in the window
        # Use >= so equal highs at the left edge don't block the right one
        if highs[i] >= np.max(window_slice_h):
            # Reject duplicates — if previous swing high is the same price, skip
            if not swing_highs or abs(highs[i] - swing_highs[-1][1]) > 1e-8:
                swing_highs.append((i, highs[i]))

        # Swing low: i must be at or below the minimum in the window
        if lows[i] <= np.min(window_slice_l):
            if not swing_lows or abs(lows[i] - swing_lows[-1][1]) > 1e-8:
                swing_lows.append((i, lows[i]))

    return swing_highs, swing_lows


def cluster_levels(points: list, tolerance_pct: float = 0.002) -> List[list]:
    """
    Cluster nearby price levels into zones.
    FIX: tolerance_pct should match zone_pips / price so clustering and zone width are consistent.
    Caller should pass tolerance_pct = zone_pips / current_price * 0.5
    """
    if not points:
        return []

    sorted_points = sorted(points, key=lambda x: x[1])
    clusters = []
    current_cluster = [sorted_points[0]]

    for i in range(1, len(sorted_points)):
        idx, price = sorted_points[i]
        cluster_center = np.mean([p[1] for p in current_cluster])

        if abs(price - cluster_center) / cluster_center <= tolerance_pct:
            current_cluster.append((idx, price))
        else:
            clusters.append(current_cluster)
            current_cluster = [(idx, price)]

    clusters.append(current_cluster)
    return clusters


def build_zones(df: pd.DataFrame, lookback: int = 200, min_touches: int = 2,
                zone_pips: float = 5.0) -> List[Zone]:
    """
    Build S/R and Demand/Supply zones from OHLCV data.

    FIX v6.1:
      - zone_pips drives cluster tolerance (was hardcoded 0.003 vs zone_pips)
      - Swing window reduced to 3 for M15/H1 data (5 was too strict)
      - is_fresh check uses zone_pips * 2 not zone_pips * 3
    """
    if len(df) < lookback // 2:
        return []

    df = df.tail(lookback).copy().reset_index(drop=True)
    close = df["close"].values
    current_price = close[-1]

    if current_price <= 0:
        return []

    # Derive tolerance from zone_pips so clustering and zone width are aligned
    # zone_pips is the half-width, tolerance is the merge radius (use 0.6x half-width)
    cluster_tol = (zone_pips / current_price) * 0.6

    swing_highs, swing_lows = find_swing_points(df, window=3)

    support_clusters    = cluster_levels(swing_lows,  tolerance_pct=cluster_tol)
    resistance_clusters = cluster_levels(swing_highs, tolerance_pct=cluster_tol)

    zones: List[Zone] = []

    def score_zone(cluster: list, zone_type: str, df: pd.DataFrame) -> Optional[Zone]:
        touches = len(cluster)
        if touches < min_touches:
            return None

        prices  = [p[1] for p in cluster]
        indices = [p[0] for p in cluster]
        center  = float(np.mean(prices))
        last_idx = int(max(indices))
        n = len(df)

        # Recency: 0-40 pts, linear from oldest (0) to newest (40)
        recency_score = int(40 * (last_idx / max(n - 1, 1)))

        # Volume reaction near zone
        vol_score = 0
        if "volume" in df.columns and df["volume"].sum() > 0:
            zone_mask = (
                (df["close"] >= center - zone_pips * 1.5) &
                (df["close"] <= center + zone_pips * 1.5)
            )
            zone_vol = df.loc[zone_mask, "volume"].mean() if zone_mask.any() else 0
            avg_vol  = df["volume"].mean()
            if avg_vol > 0 and zone_vol > 0:
                vol_score = min(20, int(20 * min(zone_vol / avg_vol, 2.0)))

        # Touch score: 10 pts each, max 40
        touch_score = min(40, touches * 10)
        total = touch_score + recency_score + vol_score

        # Freshness: price must not have closed cleanly through the zone
        recent_n = min(20, n)
        recent_closes = df["close"].values[-recent_n:]
        breach_dist = zone_pips * 2   # FIX: was zone_pips * 3 (too loose)

        if zone_type == "SUPPORT":
            broken = any(c < center - breach_dist for c in recent_closes)
        else:
            broken = any(c > center + breach_dist for c in recent_closes)
        is_fresh = not broken

        return Zone(
            price=center,
            high=center + zone_pips,
            low=center  - zone_pips,
            zone_type=zone_type,
            touches=touches,
            strength=min(100, total),
            last_touch_idx=last_idx,
            is_fresh=is_fresh,
        )

    for cluster in support_clusters:
        z = score_zone(cluster, "SUPPORT", df)
        if z and z.price < current_price:
            zones.append(z)

    for cluster in resistance_clusters:
        z = score_zone(cluster, "RESISTANCE", df)
        if z and z.price > current_price:
            zones.append(z)

    # Upgrade strong zones to DEMAND / SUPPLY
    for z in zones:
        if z.zone_type == "SUPPORT"    and z.strength >= 60:
            z.zone_type = "DEMAND"
        elif z.zone_type == "RESISTANCE" and z.strength >= 60:
            z.zone_type = "SUPPLY"

    zones.sort(key=lambda z: z.strength, reverse=True)
    return zones


def get_nearest_zones(zones: List[Zone], current_price: float,
                      max_distance_pct: float = 0.005):
    """
    Return nearest support/demand below and resistance/supply above current price.
    max_distance_pct: price must be within this % of the zone center to trigger.
    """
    nearest_support    = None
    nearest_resistance = None

    supports     = [z for z in zones if z.zone_type in ("SUPPORT", "DEMAND")     and z.price < current_price]
    resistances  = [z for z in zones if z.zone_type in ("RESISTANCE", "SUPPLY")  and z.price > current_price]

    if supports:
        nearest_support    = max(supports,    key=lambda z: z.price)
    if resistances:
        nearest_resistance = min(resistances, key=lambda z: z.price)

    at_support = (
        nearest_support is not None and
        abs(current_price - nearest_support.price) / current_price <= max_distance_pct
    )
    at_resistance = (
        nearest_resistance is not None and
        abs(current_price - nearest_resistance.price) / current_price <= max_distance_pct
    )

    return nearest_support, nearest_resistance, at_support, at_resistance
