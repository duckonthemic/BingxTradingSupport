"""
Market Structure Shift (MSS) Detector

Detects MSS patterns on M5 timeframe:
- Break of recent Swing High/Low
- Displacement validation (strong candle body)
- Creates new FVG for entry

MSS is the confirmation signal after price enters H1 FVG.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
from enum import Enum

from .config import IETradeConfig, DEFAULT_CONFIG
from .fvg_detector import Candle, FVG, FVGType, FVGDetector


logger = logging.getLogger(__name__)


class MSSType(Enum):
    """Market Structure Shift type"""
    BEARISH = "BEARISH"  # Break of Swing Low - confirms short
    BULLISH = "BULLISH"  # Break of Swing High - confirms long


@dataclass
class SwingPoint:
    """Swing High or Swing Low point"""
    price: float
    timestamp: datetime
    candle_index: int
    is_high: bool  # True = Swing High, False = Swing Low
    
    @property
    def type_str(self) -> str:
        return "High" if self.is_high else "Low"


@dataclass
class MSS:
    """Market Structure Shift data"""
    
    mss_type: MSSType
    symbol: str
    timeframe: str
    
    # Break details
    break_price: float          # Price level that was broken
    break_candle_index: int     # Candle that caused the break
    break_timestamp: datetime
    
    # Displacement candle info
    displacement_body_ratio: float  # Body as % of range
    displacement_size: float        # Size of displacement move
    
    # Swing points
    swing_high: Optional[SwingPoint] = None
    swing_low: Optional[SwingPoint] = None
    
    # Entry FVG created by displacement
    entry_fvg: Optional[FVG] = None
    
    # Validation
    is_valid: bool = True
    invalidation_reason: str = ""
    
    def __str__(self) -> str:
        return (f"MSS({self.mss_type.value} {self.symbol} {self.timeframe} "
                f"break={self.break_price:.4f} disp={self.displacement_body_ratio:.1%})")


class MSSDetector:
    """
    Market Structure Shift detector with ICT methodology
    
    MSS Pattern for SHORT:
    1. Price is making Higher Highs on M5 (pullback into H1 FVG)
    2. Suddenly breaks below recent Swing Low
    3. Break candle has strong body (>70% displacement)
    4. Creates new FVG for entry
    
    MSS Pattern for LONG:
    1. Price is making Lower Lows on M5 (pullback into H1 FVG)
    2. Suddenly breaks above recent Swing High
    3. Break candle has strong body (>70% displacement)
    4. Creates new FVG for entry
    """
    
    def __init__(self, config: IETradeConfig = DEFAULT_CONFIG):
        self.config = config
        self.fvg_detector = FVGDetector(config)
    
    def find_swing_points(
        self,
        candles: List[Candle],
        lookback: int = 5
    ) -> List[SwingPoint]:
        """
        Find swing highs and lows in candle data
        
        A Swing High is a candle with:
        - Higher high than {lookback} candles before AND after
        
        A Swing Low is a candle with:
        - Lower low than {lookback} candles before AND after
        """
        swings = []
        
        if len(candles) < (lookback * 2 + 1):
            return swings
        
        for i in range(lookback, len(candles) - lookback):
            candle = candles[i]
            
            # Check for Swing High
            is_swing_high = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and candles[j].high >= candle.high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swings.append(SwingPoint(
                    price=candle.high,
                    timestamp=candle.timestamp,
                    candle_index=i,
                    is_high=True
                ))
            
            # Check for Swing Low
            is_swing_low = True
            for j in range(i - lookback, i + lookback + 1):
                if j != i and candles[j].low <= candle.low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swings.append(SwingPoint(
                    price=candle.low,
                    timestamp=candle.timestamp,
                    candle_index=i,
                    is_high=False
                ))
        
        # Sort by time
        swings.sort(key=lambda x: x.timestamp)
        
        return swings
    
    def detect_mss(
        self,
        candles: List[Candle],
        symbol: str,
        direction: str,  # "LONG" or "SHORT"
        in_fvg_zone: bool = True  # Whether price is in H1 FVG
    ) -> Optional[MSS]:
        """
        Detect Market Structure Shift on M5 timeframe
        
        Args:
            candles: M5 candle data (oldest first)
            symbol: Trading pair
            direction: "LONG" or "SHORT"
            in_fvg_zone: Whether price has entered H1 FVG
            
        Returns:
            MSS object if detected, None otherwise
        """
        if not in_fvg_zone:
            return None
        
        if len(candles) < 15:  # Need enough history
            return None
        
        # Find swing points
        swings = self.find_swing_points(
            candles, 
            lookback=self.config.MSS_SWING_LOOKBACK
        )
        
        if len(swings) < 2:
            return None
        
        # Look for MSS in recent candles
        max_lookback = min(self.config.MSS_MAX_LOOKBACK, len(candles) - 1)
        
        for i in range(len(candles) - 1, len(candles) - max_lookback - 1, -1):
            candle = candles[i]
            
            mss = self._check_mss_at_candle(
                candle=candle,
                candle_index=i,
                candles=candles,
                swings=swings,
                symbol=symbol,
                direction=direction
            )
            
            if mss and mss.is_valid:
                # Try to find entry FVG created by displacement
                mss.entry_fvg = self._find_displacement_fvg(
                    candles=candles,
                    mss_index=i,
                    symbol=symbol,
                    direction=direction
                )
                return mss
        
        return None
    
    def _check_mss_at_candle(
        self,
        candle: Candle,
        candle_index: int,
        candles: List[Candle],
        swings: List[SwingPoint],
        symbol: str,
        direction: str
    ) -> Optional[MSS]:
        """Check if a specific candle creates MSS"""
        
        if direction == "SHORT":
            # Looking for Bearish MSS (break of Swing Low)
            
            # Find most recent Swing Low before this candle
            swing_low = None
            for s in reversed(swings):
                if not s.is_high and s.candle_index < candle_index:
                    swing_low = s
                    break
            
            if not swing_low:
                return None
            
            # Check if candle breaks below Swing Low
            if candle.close < swing_low.price:
                # Validate displacement (strong candle body)
                if candle.body_ratio >= self.config.MSS_MIN_BODY_RATIO:
                    # Find recent Swing High for SL placement
                    swing_high = None
                    for s in reversed(swings):
                        if s.is_high and s.candle_index < candle_index:
                            swing_high = s
                            break
                    
                    return MSS(
                        mss_type=MSSType.BEARISH,
                        symbol=symbol,
                        timeframe="5m",
                        break_price=swing_low.price,
                        break_candle_index=candle_index,
                        break_timestamp=candle.timestamp,
                        displacement_body_ratio=candle.body_ratio,
                        displacement_size=abs(candle.close - candle.open),
                        swing_high=swing_high,
                        swing_low=swing_low,
                        is_valid=True
                    )
                else:
                    # Weak break - not a valid MSS
                    return MSS(
                        mss_type=MSSType.BEARISH,
                        symbol=symbol,
                        timeframe="5m",
                        break_price=swing_low.price,
                        break_candle_index=candle_index,
                        break_timestamp=candle.timestamp,
                        displacement_body_ratio=candle.body_ratio,
                        displacement_size=abs(candle.close - candle.open),
                        is_valid=False,
                        invalidation_reason=f"Weak displacement ({candle.body_ratio:.1%} < 70%)"
                    )
        
        elif direction == "LONG":
            # Looking for Bullish MSS (break of Swing High)
            
            # Find most recent Swing High before this candle
            swing_high = None
            for s in reversed(swings):
                if s.is_high and s.candle_index < candle_index:
                    swing_high = s
                    break
            
            if not swing_high:
                return None
            
            # Check if candle breaks above Swing High
            if candle.close > swing_high.price:
                # Validate displacement
                if candle.body_ratio >= self.config.MSS_MIN_BODY_RATIO:
                    # Find recent Swing Low for SL placement
                    swing_low = None
                    for s in reversed(swings):
                        if not s.is_high and s.candle_index < candle_index:
                            swing_low = s
                            break
                    
                    return MSS(
                        mss_type=MSSType.BULLISH,
                        symbol=symbol,
                        timeframe="5m",
                        break_price=swing_high.price,
                        break_candle_index=candle_index,
                        break_timestamp=candle.timestamp,
                        displacement_body_ratio=candle.body_ratio,
                        displacement_size=abs(candle.close - candle.open),
                        swing_high=swing_high,
                        swing_low=swing_low,
                        is_valid=True
                    )
                else:
                    return MSS(
                        mss_type=MSSType.BULLISH,
                        symbol=symbol,
                        timeframe="5m",
                        break_price=swing_high.price,
                        break_candle_index=candle_index,
                        break_timestamp=candle.timestamp,
                        displacement_body_ratio=candle.body_ratio,
                        displacement_size=abs(candle.close - candle.open),
                        is_valid=False,
                        invalidation_reason=f"Weak displacement ({candle.body_ratio:.1%} < 70%)"
                    )
        
        return None
    
    def _find_displacement_fvg(
        self,
        candles: List[Candle],
        mss_index: int,
        symbol: str,
        direction: str
    ) -> Optional[FVG]:
        """
        Find the FVG created by the MSS displacement move
        
        This is the M5 FVG we use for entry
        """
        # Look at candles around MSS
        start_idx = max(0, mss_index - 2)
        end_idx = min(len(candles), mss_index + 3)
        
        recent_candles = candles[start_idx:end_idx]
        
        if len(recent_candles) < 3:
            return None
        
        # Detect FVGs in this small window
        fvgs = self.fvg_detector.detect_fvgs(
            candles=recent_candles,
            symbol=symbol,
            timeframe="5m",
            direction=direction
        )
        
        # Return most recent valid FVG
        valid_fvgs = [f for f in fvgs if f.is_valid]
        
        if valid_fvgs:
            return valid_fvgs[0]
        
        return None
    
    def get_sl_level(self, mss: MSS, direction: str) -> Optional[float]:
        """
        Get Stop Loss level from MSS
        
        For SHORT: SL above recent Swing High
        For LONG: SL below recent Swing Low
        """
        if direction == "SHORT":
            if mss.swing_high:
                # Add small buffer above swing high
                buffer = mss.swing_high.price * 0.001  # 0.1%
                return mss.swing_high.price + buffer
        elif direction == "LONG":
            if mss.swing_low:
                # Add small buffer below swing low
                buffer = mss.swing_low.price * 0.001
                return mss.swing_low.price - buffer
        
        return None
