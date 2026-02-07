"""
POI (Points of Interest) Manager — In-Memory HTF Level Tracker

Tracks key structural levels from H1 klines:
- Previous Day High/Low (PDH/PDL)
- H1 Swing Highs/Lows
- H1 Breaker Blocks
- H1 FVGs, BPR zones, Inverse FVGs

Used by ICT strategies (Silver Bullet, Unicorn, Turtle Soup)
for multi-timeframe POI alignment and confirmation scoring.

Storage: in-memory dict keyed by symbol. Refreshed each scan cycle.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class POIType(Enum):
    """Types of Points of Interest."""
    PDH = "PDH"                    # Previous Day High
    PDL = "PDL"                    # Previous Day Low
    SWING_HIGH_H1 = "SWING_HIGH"  # H1 Swing High
    SWING_LOW_H1 = "SWING_LOW"    # H1 Swing Low
    BREAKER_H1 = "BREAKER"        # H1 Breaker Block
    FVG_H1 = "FVG"                # H1 Fair Value Gap
    BPR_H1 = "BPR"                # Balanced Price Range
    IFVG_H1 = "IFVG"              # Inverse FVG (filled → flipped)


@dataclass
class POI:
    """A single Point of Interest level."""
    poi_type: POIType
    price_high: float           # Upper boundary
    price_low: float            # Lower boundary
    created_at: datetime = field(default_factory=datetime.utcnow)
    direction: str = "NEUTRAL"  # BULLISH / BEARISH / NEUTRAL
    strength: float = 1.0       # 0.0 - 1.0
    touched: bool = False       # Has price reached this level
    
    @property
    def mid(self) -> float:
        return (self.price_high + self.price_low) / 2
    
    @property
    def size_pct(self) -> float:
        if self.mid == 0:
            return 0
        return abs(self.price_high - self.price_low) / self.mid * 100
    
    @property
    def age_hours(self) -> float:
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    def is_price_at_level(self, price: float, tolerance_pct: float = 0.3) -> bool:
        """Check if price is at this POI level (within tolerance)."""
        tol = self.mid * tolerance_pct / 100
        return (self.price_low - tol) <= price <= (self.price_high + tol)
    
    def __repr__(self) -> str:
        return f"POI({self.poi_type.value} {self.price_low:.4f}-{self.price_high:.4f} {self.direction})"


# Max ages for different POI types
POI_MAX_AGE_HOURS = {
    POIType.PDH: 48,
    POIType.PDL: 48,
    POIType.SWING_HIGH_H1: 72,
    POIType.SWING_LOW_H1: 72,
    POIType.BREAKER_H1: 48,
    POIType.FVG_H1: 24,
    POIType.BPR_H1: 24,
    POIType.IFVG_H1: 36,
}


class POIManager:
    """
    In-memory manager for Points of Interest across all symbols.
    
    Refreshed every scan cycle from H1 klines. No persistence needed.
    Memory: ~50 symbols × ~20 POIs × 100 bytes ≈ 100KB
    """
    
    def __init__(self):
        self._pois: Dict[str, List[POI]] = {}
        self._last_update: Dict[str, datetime] = {}
        self.swing_lookback = 10  # H1 candles for swing detection
        self.max_pois_per_symbol = 30
    
    def update(self, symbol: str, df_h1: pd.DataFrame) -> List[POI]:
        """
        Refresh POI levels for a symbol from H1 klines.
        
        Args:
            symbol: Trading pair
            df_h1: H1 OHLCV DataFrame (expects columns: open, high, low, close, volume)
            
        Returns:
            List of detected POI levels
        """
        if df_h1.empty or len(df_h1) < 30:
            return self._pois.get(symbol, [])
        
        pois: List[POI] = []
        
        # 1. Previous Day High/Low
        pdh_pdl = self._calc_pdh_pdl(df_h1)
        pois.extend(pdh_pdl)
        
        # 2. H1 Swing Highs/Lows
        swings = self._calc_swing_levels(df_h1)
        pois.extend(swings)
        
        # 3. H1 Breaker Blocks
        breakers = self._calc_breaker_blocks(df_h1)
        pois.extend(breakers)
        
        # 4. H1 FVGs
        fvgs = self._calc_fvgs(df_h1)
        pois.extend(fvgs)
        
        # 5. BPR zones (overlapping bullish+bearish FVGs)
        bprs = self._calc_bpr_zones(fvgs)
        pois.extend(bprs)
        
        # 6. Inverse FVGs (filled FVGs → flipped polarity)
        ifvgs = self._calc_inverse_fvgs(df_h1, fvgs)
        pois.extend(ifvgs)
        
        # Prune stale and limit count
        pois = self._prune_pois(pois)
        
        self._pois[symbol] = pois
        self._last_update[symbol] = datetime.utcnow()
        
        logger.debug(f"POI update {symbol}: {len(pois)} levels")
        return pois
    
    def get_pois(self, symbol: str) -> List[POI]:
        """Get all POI levels for a symbol."""
        return self._pois.get(symbol, [])
    
    def get_pois_by_type(self, symbol: str, poi_type: POIType) -> List[POI]:
        """Get POI levels of specific type."""
        return [p for p in self.get_pois(symbol) if p.poi_type == poi_type]
    
    def get_pdh_pdl(self, symbol: str) -> Tuple[Optional[float], Optional[float]]:
        """Get Previous Day High/Low as (pdh, pdl) tuple."""
        pdh_list = self.get_pois_by_type(symbol, POIType.PDH)
        pdl_list = self.get_pois_by_type(symbol, POIType.PDL)
        pdh = pdh_list[0].price_high if pdh_list else None
        pdl = pdl_list[0].price_low if pdl_list else None
        return pdh, pdl
    
    def check_price_at_poi(self, symbol: str, price: float, tolerance_pct: float = 0.3) -> List[POI]:
        """Check if price is at any POI level."""
        return [p for p in self.get_pois(symbol) if p.is_price_at_level(price, tolerance_pct)]
    
    def get_nearest_poi(self, symbol: str, price: float, direction: str = "ANY") -> Optional[POI]:
        """Get the nearest POI to current price."""
        pois = self.get_pois(symbol)
        if direction != "ANY":
            pois = [p for p in pois if p.direction == direction or p.direction == "NEUTRAL"]
        if not pois:
            return None
        return min(pois, key=lambda p: abs(p.mid - price))
    
    # --- Internal calculation methods ---
    
    def _calc_pdh_pdl(self, df_h1: pd.DataFrame) -> List[POI]:
        """Calculate Previous Day High/Low from H1 candles."""
        pois = []
        
        if len(df_h1) < 24:
            return pois
        
        # Use last 24-48 H1 candles for "previous day"
        # Take candles from 24-48h ago as "previous day"
        if len(df_h1) >= 48:
            prev_day = df_h1.iloc[-48:-24]
        else:
            prev_day = df_h1.iloc[:-24]
        
        if prev_day.empty:
            return pois
        
        pdh = float(prev_day['high'].max())
        pdl = float(prev_day['low'].min())
        
        # Estimate timestamp
        ts = prev_day.index[-1] if isinstance(prev_day.index, pd.DatetimeIndex) else datetime.utcnow() - timedelta(hours=24)
        if not isinstance(ts, datetime):
            ts = datetime.utcnow() - timedelta(hours=24)
        
        pois.append(POI(
            poi_type=POIType.PDH,
            price_high=pdh,
            price_low=pdh * 0.999,  # Thin zone
            created_at=ts,
            direction="BEARISH",  # PDH acts as resistance
            strength=0.9
        ))
        
        pois.append(POI(
            poi_type=POIType.PDL,
            price_high=pdl * 1.001,
            price_low=pdl,
            created_at=ts,
            direction="BULLISH",  # PDL acts as support
            strength=0.9
        ))
        
        return pois
    
    def _calc_swing_levels(self, df_h1: pd.DataFrame) -> List[POI]:
        """Find H1 swing highs and lows."""
        pois = []
        highs = df_h1['high'].values
        lows = df_h1['low'].values
        lb = self.swing_lookback
        
        for i in range(lb, len(highs) - lb):
            window_h = highs[i - lb:i + lb + 1]
            window_l = lows[i - lb:i + lb + 1]
            
            # Swing High
            if highs[i] == max(window_h):
                ts = self._get_timestamp(df_h1, i)
                pois.append(POI(
                    poi_type=POIType.SWING_HIGH_H1,
                    price_high=float(highs[i]),
                    price_low=float(highs[i]) * 0.999,
                    created_at=ts,
                    direction="BEARISH",
                    strength=0.8
                ))
            
            # Swing Low
            if lows[i] == min(window_l):
                ts = self._get_timestamp(df_h1, i)
                pois.append(POI(
                    poi_type=POIType.SWING_LOW_H1,
                    price_high=float(lows[i]) * 1.001,
                    price_low=float(lows[i]),
                    created_at=ts,
                    direction="BULLISH",
                    strength=0.8
                ))
        
        # Keep only last 5 swing highs and 5 swing lows
        swing_highs = [p for p in pois if p.poi_type == POIType.SWING_HIGH_H1]
        swing_lows = [p for p in pois if p.poi_type == POIType.SWING_LOW_H1]
        return swing_highs[-5:] + swing_lows[-5:]
    
    def _calc_breaker_blocks(self, df_h1: pd.DataFrame) -> List[POI]:
        """
        Find H1 Breaker Blocks.
        A breaker = a swing high/low that was broken, then flips role.
        Bearish Breaker: swing high broken up → retests as support → fails = resistance
        Bullish Breaker: swing low broken down → retests as resistance → fails = support
        """
        pois = []
        if len(df_h1) < 30:
            return pois
        
        opens = df_h1['open'].values
        closes = df_h1['close'].values
        highs = df_h1['high'].values
        lows = df_h1['low'].values
        
        # Bearish breaker: prior bullish candle before strong down move
        for i in range(len(df_h1) - 3, max(len(df_h1) - 30, 2), -1):
            # Bullish candle followed by bearish breakdown
            if closes[i] > opens[i]:  # Bullish candle
                if (i + 2 < len(df_h1) and 
                    closes[i + 1] < opens[i + 1] and  # Next is bearish
                    closes[i + 2] < lows[i]):          # Broke below
                    ts = self._get_timestamp(df_h1, i)
                    pois.append(POI(
                        poi_type=POIType.BREAKER_H1,
                        price_high=float(highs[i]),
                        price_low=float(lows[i]),
                        created_at=ts,
                        direction="BEARISH",
                        strength=0.85
                    ))
                    if len([p for p in pois if p.poi_type == POIType.BREAKER_H1]) >= 3:
                        break
        
        # Bullish breaker: prior bearish candle before strong up move
        for i in range(len(df_h1) - 3, max(len(df_h1) - 30, 2), -1):
            if closes[i] < opens[i]:  # Bearish candle
                if (i + 2 < len(df_h1) and
                    closes[i + 1] > opens[i + 1] and  # Next is bullish
                    closes[i + 2] > highs[i]):         # Broke above
                    ts = self._get_timestamp(df_h1, i)
                    pois.append(POI(
                        poi_type=POIType.BREAKER_H1,
                        price_high=float(highs[i]),
                        price_low=float(lows[i]),
                        created_at=ts,
                        direction="BULLISH",
                        strength=0.85
                    ))
                    if len([p for p in pois if p.direction == "BULLISH" and p.poi_type == POIType.BREAKER_H1]) >= 3:
                        break
        
        return pois
    
    def _calc_fvgs(self, df_h1: pd.DataFrame) -> List[POI]:
        """
        Detect H1 Fair Value Gaps (3-candle imbalance pattern).
        Bearish FVG: c0.low > c2.high (gap above)
        Bullish FVG: c0.high < c2.low (gap below)
        """
        pois = []
        if len(df_h1) < 3:
            return pois
        
        highs = df_h1['high'].values
        lows = df_h1['low'].values
        closes = df_h1['close'].values
        current_price = float(closes[-1])
        
        for i in range(2, len(df_h1)):
            c0_high = float(highs[i - 2])
            c0_low = float(lows[i - 2])
            c2_high = float(highs[i])
            c2_low = float(lows[i])
            
            # Bearish FVG: gap above current price
            if c0_low > c2_high:
                gap_size_pct = (c0_low - c2_high) / current_price * 100
                if 0.1 <= gap_size_pct <= 2.0:
                    ts = self._get_timestamp(df_h1, i - 1)
                    # Check if already filled
                    filled = current_price >= c0_low
                    if not filled:
                        pois.append(POI(
                            poi_type=POIType.FVG_H1,
                            price_high=c0_low,
                            price_low=c2_high,
                            created_at=ts,
                            direction="BEARISH",
                            strength=0.75
                        ))
            
            # Bullish FVG: gap below current price
            if c0_high < c2_low:
                gap_size_pct = (c2_low - c0_high) / current_price * 100
                if 0.1 <= gap_size_pct <= 2.0:
                    ts = self._get_timestamp(df_h1, i - 1)
                    filled = current_price <= c0_high
                    if not filled:
                        pois.append(POI(
                            poi_type=POIType.FVG_H1,
                            price_high=c2_low,
                            price_low=c0_high,
                            created_at=ts,
                            direction="BULLISH",
                            strength=0.75
                        ))
        
        # Keep only recent FVGs
        return pois[-6:]
    
    def _calc_bpr_zones(self, fvg_pois: List[POI]) -> List[POI]:
        """
        Find Balanced Price Ranges: overlapping bullish + bearish FVGs.
        BPR = intersection zone = very strong S/R level.
        """
        pois = []
        fvgs = [p for p in fvg_pois if p.poi_type == POIType.FVG_H1]
        
        bullish_fvgs = [f for f in fvgs if f.direction == "BULLISH"]
        bearish_fvgs = [f for f in fvgs if f.direction == "BEARISH"]
        
        for b_fvg in bullish_fvgs:
            for s_fvg in bearish_fvgs:
                # Check for overlap
                overlap_low = max(b_fvg.price_low, s_fvg.price_low)
                overlap_high = min(b_fvg.price_high, s_fvg.price_high)
                
                if overlap_low < overlap_high:
                    # Found BPR
                    ts = max(b_fvg.created_at, s_fvg.created_at)
                    pois.append(POI(
                        poi_type=POIType.BPR_H1,
                        price_high=overlap_high,
                        price_low=overlap_low,
                        created_at=ts,
                        direction="NEUTRAL",  # BPR works both ways
                        strength=0.95  # Very strong level
                    ))
        
        return pois[:3]  # Max 3 BPR zones
    
    def _calc_inverse_fvgs(self, df_h1: pd.DataFrame, fvg_pois: List[POI]) -> List[POI]:
        """
        Track filled FVGs and flip their polarity (Inverse FVG).
        A filled bullish FVG becomes bearish S/R (and vice versa).
        """
        pois = []
        if len(df_h1) < 3:
            return pois
        
        current_price = float(df_h1['close'].values[-1])
        highs = df_h1['high'].values
        lows = df_h1['low'].values
        
        # Also scan for FVGs that WERE filled (re-scan all candles)
        for i in range(2, min(len(df_h1), 50)):  # Look back 50 H1 candles
            c0_high = float(highs[i - 2])
            c0_low = float(lows[i - 2])
            c2_high = float(highs[i])
            c2_low = float(lows[i])
            
            # Check bearish FVG that got filled
            if c0_low > c2_high:
                gap_top = c0_low
                gap_bottom = c2_high
                gap_size_pct = (gap_top - gap_bottom) / current_price * 100
                if 0.1 <= gap_size_pct <= 2.0:
                    # Check if filled: any subsequent candle's high >= gap_top
                    filled = False
                    for j in range(i + 1, len(df_h1)):
                        if float(highs[j]) >= gap_top:
                            filled = True
                            break
                    
                    if filled:
                        ts = self._get_timestamp(df_h1, i - 1)
                        # Inverse: bearish FVG filled → becomes BULLISH support
                        pois.append(POI(
                            poi_type=POIType.IFVG_H1,
                            price_high=gap_top,
                            price_low=gap_bottom,
                            created_at=ts,
                            direction="BULLISH",  # Flipped
                            strength=0.7
                        ))
            
            # Check bullish FVG that got filled
            if c0_high < c2_low:
                gap_top = c2_low
                gap_bottom = c0_high
                gap_size_pct = (gap_top - gap_bottom) / current_price * 100
                if 0.1 <= gap_size_pct <= 2.0:
                    filled = False
                    for j in range(i + 1, len(df_h1)):
                        if float(lows[j]) <= gap_bottom:
                            filled = True
                            break
                    
                    if filled:
                        ts = self._get_timestamp(df_h1, i - 1)
                        # Inverse: bullish FVG filled → becomes BEARISH resistance
                        pois.append(POI(
                            poi_type=POIType.IFVG_H1,
                            price_high=gap_top,
                            price_low=gap_bottom,
                            created_at=ts,
                            direction="BEARISH",  # Flipped
                            strength=0.7
                        ))
        
        return pois[-4:]  # Keep recent
    
    def _prune_pois(self, pois: List[POI]) -> List[POI]:
        """Remove stale POIs and limit total count."""
        result = []
        for p in pois:
            max_age = POI_MAX_AGE_HOURS.get(p.poi_type, 48)
            if p.age_hours <= max_age:
                result.append(p)
        
        # Sort by strength (strongest first) and limit
        result.sort(key=lambda p: p.strength, reverse=True)
        return result[:self.max_pois_per_symbol]
    
    def _get_timestamp(self, df: pd.DataFrame, idx: int) -> datetime:
        """Extract timestamp from DataFrame at index."""
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                return df.index[idx].to_pydatetime()
            elif 'timestamp' in df.columns:
                ts = df['timestamp'].iloc[idx]
                if isinstance(ts, (int, float)):
                    if ts > 10000000000:
                        ts = ts / 1000
                    return datetime.utcfromtimestamp(ts)
                return ts
        except Exception:
            pass
        return datetime.utcnow() - timedelta(hours=max(0, len(df) - idx))
    
    def clear(self, symbol: Optional[str] = None):
        """Clear POI data."""
        if symbol:
            self._pois.pop(symbol, None)
            self._last_update.pop(symbol, None)
        else:
            self._pois.clear()
            self._last_update.clear()
