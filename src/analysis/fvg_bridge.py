"""
FVG Bridge — Bridges IE Trade FVG detector into main strategy pipeline.

Provides utility functions to:
- Convert DataFrame klines → Candle objects for FVGDetector
- Find BPR zones (overlapping bullish+bearish FVGs)
- Find IFVG levels (filled FVGs with flipped polarity)
- Check if a price falls within an FVG zone

This avoids duplicating FVG logic — single source of truth in ie_trade.fvg_detector.
"""

import logging
from typing import List, Optional, Tuple
from datetime import datetime

import pandas as pd
import numpy as np

from ..ie_trade.fvg_detector import FVGDetector, FVG, FVGType, Candle, candles_from_api_data
from ..ie_trade.config import DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class FVGBridge:
    """
    Bridge between IE Trade FVG detection and main strategy pipeline.
    
    Usage:
        bridge = FVGBridge()
        fvgs = bridge.detect_from_dataframe(df_h1, symbol, "1h", "SHORT")
        bpr_zones = bridge.find_bpr_zones(symbol, df_h1)
        is_in = bridge.check_price_in_fvg(price, fvgs)
    """
    
    def __init__(self):
        self.detector = FVGDetector(config=DEFAULT_CONFIG)
    
    def df_to_candles(self, df: pd.DataFrame) -> List[Candle]:
        """Convert a DataFrame to list of Candle objects."""
        candles = []
        for i in range(len(df)):
            row = df.iloc[i]
            
            # Get timestamp
            if isinstance(df.index, pd.DatetimeIndex):
                ts = df.index[i].to_pydatetime()
            elif 'timestamp' in df.columns:
                ts_val = row.get('timestamp', 0)
                if isinstance(ts_val, (int, float)):
                    if ts_val > 10000000000:
                        ts_val = ts_val / 1000
                    ts = datetime.utcfromtimestamp(ts_val)
                else:
                    ts = ts_val if isinstance(ts_val, datetime) else datetime.utcnow()
            else:
                ts = datetime.utcnow()
            
            candles.append(Candle(
                timestamp=ts,
                open=float(row['open']),
                high=float(row['high']),
                low=float(row['low']),
                close=float(row['close']),
                volume=float(row.get('volume', 0))
            ))
        
        return candles
    
    def detect_from_dataframe(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        direction: str
    ) -> List[FVG]:
        """
        Detect FVGs from a DataFrame.
        
        Args:
            df: OHLCV DataFrame
            symbol: Trading pair
            timeframe: "1h" or "5m"
            direction: "LONG" or "SHORT"
            
        Returns:
            List of detected FVGs (valid, unfilled)
        """
        if df.empty or len(df) < 3:
            return []
        
        candles = self.df_to_candles(df)
        
        # Get swing range for premium/discount zone validation
        swing_high, swing_low = self.detector._find_swing_range(candles)
        
        fvgs = self.detector.detect_fvgs(
            candles=candles,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            swing_high=swing_high,
            swing_low=swing_low
        )
        
        # Return only valid, unfilled FVGs
        return [f for f in fvgs if f.is_valid and not f.filled]
    
    def detect_all_fvgs(
        self,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str
    ) -> Tuple[List[FVG], List[FVG]]:
        """
        Detect both bullish and bearish FVGs.
        
        Returns:
            (bullish_fvgs, bearish_fvgs)
        """
        bullish = self.detect_from_dataframe(df, symbol, timeframe, "LONG")
        bearish = self.detect_from_dataframe(df, symbol, timeframe, "SHORT")
        return bullish, bearish
    
    def find_bpr_zones(
        self,
        symbol: str,
        df_h1: pd.DataFrame
    ) -> List[dict]:
        """
        Find Balanced Price Ranges: overlapping bullish + bearish FVGs.
        
        Returns:
            List of {'high': float, 'low': float, 'mid': float, 'strength': float}
        """
        bullish_fvgs, bearish_fvgs = self.detect_all_fvgs(df_h1, symbol, "1h")
        
        bpr_zones = []
        for b_fvg in bullish_fvgs:
            for s_fvg in bearish_fvgs:
                overlap_low = max(b_fvg.bottom, s_fvg.bottom)
                overlap_high = min(b_fvg.top, s_fvg.top)
                
                if overlap_low < overlap_high:
                    bpr_zones.append({
                        'high': overlap_high,
                        'low': overlap_low,
                        'mid': (overlap_high + overlap_low) / 2,
                        'strength': 0.95
                    })
        
        return bpr_zones[:3]
    
    def find_ifvg_levels(
        self,
        symbol: str,
        df: pd.DataFrame,
        timeframe: str = "1h"
    ) -> List[dict]:
        """
        Find Inverse FVGs (filled FVGs with flipped polarity).
        
        A filled bearish FVG becomes bullish support zone.
        A filled bullish FVG becomes bearish resistance zone.
        
        Returns:
            List of {'high': float, 'low': float, 'direction': str, 'strength': float}
        """
        if df.empty or len(df) < 5:
            return []
        
        candles = self.df_to_candles(df)
        current_price = candles[-1].close
        ifvg_levels = []
        
        for i in range(2, min(len(candles), 50)):
            c0 = candles[i - 2]
            c2 = candles[i]
            
            # Bearish FVG
            if c0.low > c2.high:
                gap_top = c0.low
                gap_bottom = c2.high
                size_pct = (gap_top - gap_bottom) / current_price * 100
                if 0.1 <= size_pct <= 2.0:
                    # Check if filled by subsequent candles
                    filled = any(candles[j].high >= gap_top for j in range(i + 1, len(candles)))
                    if filled:
                        ifvg_levels.append({
                            'high': gap_top,
                            'low': gap_bottom,
                            'direction': 'BULLISH',  # Flipped
                            'strength': 0.70
                        })
            
            # Bullish FVG
            if c0.high < c2.low:
                gap_top = c2.low
                gap_bottom = c0.high
                size_pct = (gap_top - gap_bottom) / current_price * 100
                if 0.1 <= size_pct <= 2.0:
                    filled = any(candles[j].low <= gap_bottom for j in range(i + 1, len(candles)))
                    if filled:
                        ifvg_levels.append({
                            'high': gap_top,
                            'low': gap_bottom,
                            'direction': 'BEARISH',  # Flipped
                            'strength': 0.70
                        })
        
        return ifvg_levels[-4:]
    
    def check_price_in_fvg(self, price: float, fvgs: List[FVG]) -> Optional[FVG]:
        """Check if price is inside any FVG. Returns the FVG or None."""
        for fvg in fvgs:
            if fvg.is_price_in_gap(price):
                return fvg
        return None
    
    def find_best_entry_fvg(
        self,
        df_m5: pd.DataFrame,
        symbol: str,
        direction: str,
        current_price: float
    ) -> Optional[Tuple[float, float, float]]:
        """
        Find best M5 FVG for entry.
        
        Returns:
            (entry_price, zone_top, zone_bottom) or None
        """
        fvgs = self.detect_from_dataframe(df_m5, symbol, "5m", direction)
        
        if not fvgs:
            return None
        
        # Find FVG closest to current price (most actionable)
        best = min(fvgs, key=lambda f: abs(f.mid - current_price))
        return self.detector.get_entry_zone(best)
