"""
indicators_ext.py — Extended Technical Indicators for TradingbotV72

New indicators added for:
  - Mean Reversion with Volume Clusters strategy
  - Bollinger Band Squeeze with VWAP & Volume Profile strategy

Indicators:
  - vwap()               : Volume Weighted Average Price (session-reset or rolling)
  - volume_profile()     : Builds a price/volume histogram (Point of Control, HVN, LVN)
  - keltner_channels()   : Keltner Channels (EMA ± ATR multiplier)
  - bb_squeeze()         : Bollinger Band Squeeze detector (BB inside KC)
  - volume_clusters()    : Identifies high-volume price clusters for mean reversion
  - zscore()             : Z-score of price relative to a rolling mean
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


# ── VWAP ─────────────────────────────────────────────────────────────────────

def vwap(df: pd.DataFrame, anchor: str = "rolling", period: int = 50) -> pd.Series:
    """
    Compute VWAP.

    Parameters
    ----------
    df      : DataFrame with columns [open, high, low, close, volume]
    anchor  : "rolling" (period-bar rolling VWAP) or "session" (cumulative from bar 0)
    period  : number of bars for rolling VWAP (ignored for session mode)

    Returns
    -------
    pd.Series of VWAP values
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    tp_vol = typical_price * df["volume"]

    if anchor == "session":
        cumulative_tp_vol = tp_vol.cumsum()
        cumulative_vol = df["volume"].cumsum()
        return cumulative_tp_vol / cumulative_vol.replace(0, np.nan)
    else:
        rolling_tp_vol = tp_vol.rolling(period).sum()
        rolling_vol = df["volume"].rolling(period).sum()
        return rolling_tp_vol / rolling_vol.replace(0, np.nan)


def vwap_bands(df: pd.DataFrame, period: int = 50, std_mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    VWAP with upper and lower standard deviation bands.

    Returns
    -------
    (vwap_upper, vwap_mid, vwap_lower)
    """
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    vwap_mid = vwap(df, anchor="rolling", period=period)
    deviation = typical_price.rolling(period).std()
    upper = vwap_mid + std_mult * deviation
    lower = vwap_mid - std_mult * deviation
    return upper, vwap_mid, lower


# ── Volume Profile ────────────────────────────────────────────────────────────

class VolumeProfileResult:
    """Result of a volume profile calculation."""
    def __init__(self, poc: float, hvn_levels: List[float], lvn_levels: List[float],
                 profile: pd.Series, price_bins: pd.Index):
        self.poc = poc                    # Point of Control (highest volume price)
        self.hvn_levels = hvn_levels      # High Volume Nodes (top 20% by volume)
        self.lvn_levels = lvn_levels      # Low Volume Nodes (bottom 20% by volume)
        self.profile = profile            # Full volume profile Series
        self.price_bins = price_bins      # Price bin edges

    def nearest_hvn(self, current_price: float) -> Optional[float]:
        """Return the nearest HVN to current_price."""
        if not self.hvn_levels:
            return None
        return min(self.hvn_levels, key=lambda x: abs(x - current_price))

    def nearest_lvn(self, current_price: float) -> Optional[float]:
        """Return the nearest LVN to current_price."""
        if not self.lvn_levels:
            return None
        return min(self.lvn_levels, key=lambda x: abs(x - current_price))

    def is_at_hvn(self, current_price: float, tolerance_pct: float = 0.002) -> bool:
        """Check if current_price is within tolerance of any HVN."""
        for hvn in self.hvn_levels:
            if abs(current_price - hvn) / hvn <= tolerance_pct:
                return True
        return False

    def is_at_lvn(self, current_price: float, tolerance_pct: float = 0.002) -> bool:
        """Check if current_price is within tolerance of any LVN."""
        for lvn in self.lvn_levels:
            if abs(current_price - lvn) / lvn <= tolerance_pct:
                return True
        return False


def volume_profile(df: pd.DataFrame, bins: int = 30,
                   hvn_pct: float = 0.20, lvn_pct: float = 0.20) -> VolumeProfileResult:
    """
    Build a Volume Profile (price histogram weighted by volume).

    Parameters
    ----------
    df       : DataFrame with [high, low, close, volume]
    bins     : number of price bins for the histogram
    hvn_pct  : top fraction of bins classified as High Volume Nodes
    lvn_pct  : bottom fraction of bins classified as Low Volume Nodes

    Returns
    -------
    VolumeProfileResult with POC, HVN list, LVN list, and full profile
    """
    price_min = df["low"].min()
    price_max = df["high"].max()

    if price_max <= price_min:
        # Degenerate case: flat price
        mid = (price_max + price_min) / 2
        empty = pd.Series([0.0], index=pd.Index([mid]))
        return VolumeProfileResult(mid, [mid], [mid], empty, empty.index)

    bin_edges = np.linspace(price_min, price_max, bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Distribute each bar's volume across the bins it spans
    volume_bins = np.zeros(bins)
    for _, row in df.iterrows():
        bar_low  = row["low"]
        bar_high = row["high"]
        bar_vol  = row["volume"]
        if bar_vol <= 0:
            continue
        # Find bins that overlap with this bar
        overlap_mask = (bin_edges[1:] > bar_low) & (bin_edges[:-1] < bar_high)
        n_overlap = overlap_mask.sum()
        if n_overlap > 0:
            volume_bins[overlap_mask] += bar_vol / n_overlap

    profile = pd.Series(volume_bins, index=pd.Index(bin_centers))

    # Point of Control
    poc = float(profile.idxmax())

    # HVN: top hvn_pct bins by volume
    n_hvn = max(1, int(bins * hvn_pct))
    hvn_levels = sorted(profile.nlargest(n_hvn).index.tolist())

    # LVN: bottom lvn_pct bins by volume (but must have some volume)
    active_bins = profile[profile > 0]
    if len(active_bins) > 0:
        n_lvn = max(1, int(len(active_bins) * lvn_pct))
        lvn_levels = sorted(active_bins.nsmallest(n_lvn).index.tolist())
    else:
        lvn_levels = []

    return VolumeProfileResult(poc, hvn_levels, lvn_levels, profile, profile.index)


# ── Keltner Channels ──────────────────────────────────────────────────────────

def keltner_channels(df: pd.DataFrame, ema_period: int = 20,
                     atr_period: int = 10, mult: float = 1.5) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Keltner Channels: EMA ± mult * ATR.

    Returns
    -------
    (upper, mid, lower) as pd.Series
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    mid = close.ewm(span=ema_period, adjust=False).mean()

    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr_val = tr.ewm(span=atr_period, adjust=False).mean()

    upper = mid + mult * atr_val
    lower = mid - mult * atr_val
    return upper, mid, lower


# ── Bollinger Band Squeeze ────────────────────────────────────────────────────

def bb_squeeze(df: pd.DataFrame, bb_period: int = 20, bb_std: float = 2.0,
               kc_period: int = 20, kc_atr_period: int = 10,
               kc_mult: float = 1.5) -> pd.Series:
    """
    Bollinger Band Squeeze Detector.

    A squeeze occurs when the Bollinger Bands are entirely inside the Keltner Channels.
    This indicates a period of low volatility that typically precedes a breakout.

    Returns
    -------
    pd.Series of bool: True = squeeze is active, False = no squeeze
    """
    close = df["close"]

    # Bollinger Bands
    bb_mid   = close.rolling(bb_period).mean()
    bb_std_v = close.rolling(bb_period).std()
    bb_upper = bb_mid + bb_std * bb_std_v
    bb_lower = bb_mid - bb_std * bb_std_v

    # Keltner Channels
    kc_upper, kc_mid, kc_lower = keltner_channels(df, kc_period, kc_atr_period, kc_mult)

    # Squeeze: BB is inside KC
    squeeze = (bb_upper <= kc_upper) & (bb_lower >= kc_lower)
    return squeeze


def bb_squeeze_momentum(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """
    Squeeze Momentum Oscillator (simplified Lazybear-style).
    Measures momentum as delta of price from the midpoint of the highest high/lowest low
    and the SMA of that midpoint.

    Returns
    -------
    pd.Series of momentum values (positive = bullish, negative = bearish)
    """
    close = df["close"]
    high  = df["high"]
    low   = df["low"]

    # Midpoint of highest high / lowest low over the period
    highest_high = high.rolling(period).max()
    lowest_low   = low.rolling(period).min()
    mid_hl = (highest_high + lowest_low) / 2

    # SMA of close
    sma_close = close.rolling(period).mean()

    # Delta from midpoint
    delta = close - (mid_hl + sma_close) / 2

    # Smooth the delta
    momentum = delta.rolling(period).mean()
    return momentum


# ── Volume Clusters ───────────────────────────────────────────────────────────

class VolumeCluster:
    """Represents a significant volume cluster (price zone with high traded volume)."""
    def __init__(self, price: float, volume: float, cluster_type: str):
        self.price = price          # Center price of the cluster
        self.volume = volume        # Total volume at this cluster
        self.cluster_type = cluster_type  # "SUPPORT" or "RESISTANCE" or "NEUTRAL"

    def __repr__(self):
        return f"VolumeCluster({self.cluster_type} @ {self.price:.2f}, vol={self.volume:.0f})"


def volume_clusters(df: pd.DataFrame, bins: int = 30,
                    top_n: int = 5, current_price: Optional[float] = None) -> List[VolumeCluster]:
    """
    Identify significant volume clusters from the volume profile.

    Clusters below current_price are labeled SUPPORT, above are RESISTANCE.

    Parameters
    ----------
    df            : OHLCV DataFrame
    bins          : number of price bins
    top_n         : number of top clusters to return
    current_price : current price for support/resistance classification

    Returns
    -------
    List of VolumeCluster objects sorted by volume descending
    """
    vp = volume_profile(df, bins=bins)
    profile = vp.profile

    if current_price is None:
        current_price = df["close"].iloc[-1]

    # Get top N bins by volume
    top_bins = profile.nlargest(top_n)
    clusters = []
    for price_level, vol in top_bins.items():
        if price_level < current_price * 0.9995:
            ctype = "SUPPORT"
        elif price_level > current_price * 1.0005:
            ctype = "RESISTANCE"
        else:
            ctype = "NEUTRAL"
        clusters.append(VolumeCluster(float(price_level), float(vol), ctype))

    return sorted(clusters, key=lambda c: c.volume, reverse=True)


# ── Z-Score ───────────────────────────────────────────────────────────────────

def zscore(series: pd.Series, period: int = 20) -> pd.Series:
    """
    Rolling Z-score: (price - mean) / std over a rolling window.

    A Z-score > 2 suggests overbought; < -2 suggests oversold.
    """
    rolling_mean = series.rolling(period).mean()
    rolling_std  = series.rolling(period).std()
    return (series - rolling_mean) / rolling_std.replace(0, np.nan)


# ── Convenience: price distance from VWAP ────────────────────────────────────

def vwap_distance_pct(df: pd.DataFrame, period: int = 50) -> pd.Series:
    """
    Returns the percentage distance of close from VWAP.
    Positive = price above VWAP, Negative = price below VWAP.
    """
    vwap_mid = vwap(df, anchor="rolling", period=period)
    return (df["close"] - vwap_mid) / vwap_mid * 100
