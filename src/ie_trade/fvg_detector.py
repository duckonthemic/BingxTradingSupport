"""
Fair Value Gap (FVG) Detector

Detects FVG patterns on H1 and M5 timeframes with:
- Age validation (max 24h)
- Premium/Discount zone filtering
- Size validation
- Fill status tracking
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from enum import Enum

from .config import IETradeConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class FVGType(Enum):
    """FVG direction type"""
    BULLISH = "BULLISH"  # Gap below - price needs to drop to fill
    BEARISH = "BEARISH"  # Gap above - price needs to rise to fill


class ZoneType(Enum):
    """Premium/Discount zone classification"""
    PREMIUM = "PREMIUM"      # Above 50% - good for shorts
    DISCOUNT = "DISCOUNT"    # Below 50% - good for longs
    EQUILIBRIUM = "EQUILIBRIUM"  # Around 50% - neutral


@dataclass
class FVG:
    """Fair Value Gap data structure"""
    
    # Core properties
    fvg_type: FVGType
    top: float          # Upper boundary of gap
    bottom: float       # Lower boundary of gap
    mid: float          # 50% level of gap (OTE entry)
    
    # Metadata
    symbol: str
    timeframe: str      # "1h" or "5m"
    created_at: datetime
    candle_index: int   # Index in candle array
    
    # State
    filled: bool = False
    fill_percentage: float = 0.0  # How much of gap has been filled
    
    # Zone info
    zone_type: Optional[ZoneType] = None
    fib_level: float = 0.0  # Position in swing range (0-1)
    
    # Validation
    size_pct: float = 0.0  # Gap size as % of price
    is_valid: bool = True
    invalidation_reason: str = ""
    
    @property
    def size(self) -> float:
        """Gap size in price units"""
        return abs(self.top - self.bottom)
    
    @property
    def age_hours(self) -> float:
        """Age of FVG in hours"""
        return (datetime.utcnow() - self.created_at).total_seconds() / 3600
    
    def is_fresh(self, max_age_hours: int = 24) -> bool:
        """Check if FVG is still fresh (not stale)"""
        return self.age_hours <= max_age_hours
    
    def check_fill(self, current_price: float) -> bool:
        """Check if price has filled this FVG"""
        if self.fvg_type == FVGType.BEARISH:
            # Bearish FVG: filled when price rises into gap
            if current_price >= self.bottom:
                self.fill_percentage = min(1.0, (current_price - self.bottom) / self.size)
                if current_price >= self.top:
                    self.filled = True
        else:
            # Bullish FVG: filled when price drops into gap  
            if current_price <= self.top:
                self.fill_percentage = min(1.0, (self.top - current_price) / self.size)
                if current_price <= self.bottom:
                    self.filled = True
        return self.filled
    
    def is_price_in_gap(self, current_price: float) -> bool:
        """Check if current price is within the FVG"""
        return self.bottom <= current_price <= self.top
    
    def __str__(self) -> str:
        age_str = f"{self.age_hours:.1f}h"
        zone_str = self.zone_type.value if self.zone_type else "?"
        return (f"FVG({self.fvg_type.value} {self.symbol} {self.timeframe} "
                f"${self.bottom:.4f}-${self.top:.4f} age={age_str} zone={zone_str})")


@dataclass
class Candle:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    @property
    def is_bullish(self) -> bool:
        return self.close > self.open
    
    @property
    def is_bearish(self) -> bool:
        return self.close < self.open
    
    @property
    def body(self) -> float:
        return abs(self.close - self.open)
    
    @property
    def range(self) -> float:
        return self.high - self.low
    
    @property
    def body_ratio(self) -> float:
        """Body as percentage of total range"""
        if self.range == 0:
            return 0
        return self.body / self.range


class FVGDetector:
    """
    Fair Value Gap detector with ICT methodology
    
    FVG Pattern:
    - 3 consecutive candles
    - Gap between candle[0] and candle[2] 
    - Candle[1] creates the imbalance
    
    Bearish FVG (for Short entries):
    - candle[0].low > candle[2].high
    - Gap is above current price
    - Price needs to rise to fill
    
    Bullish FVG (for Long entries):
    - candle[0].high < candle[2].low
    - Gap is below current price
    - Price needs to drop to fill
    """
    
    def __init__(self, config: IETradeConfig = DEFAULT_CONFIG):
        self.config = config
        self.active_fvgs: List[FVG] = []
    
    def detect_fvgs(
        self,
        candles: List[Candle],
        symbol: str,
        timeframe: str,
        direction: str,  # "LONG" or "SHORT"
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None
    ) -> List[FVG]:
        """
        Detect all FVGs in candle data
        
        Args:
            candles: List of OHLCV candles (oldest first)
            symbol: Trading pair symbol
            timeframe: "1h" or "5m"
            direction: "LONG" for bullish FVGs, "SHORT" for bearish FVGs
            swing_high: Recent swing high for zone calculation
            swing_low: Recent swing low for zone calculation
            
        Returns:
            List of detected FVGs, sorted by recency
        """
        fvgs = []
        
        if len(candles) < 3:
            return fvgs
        
        # Detect based on direction
        for i in range(2, len(candles)):
            c0 = candles[i - 2]  # First candle
            c1 = candles[i - 1]  # Middle candle (creates imbalance)
            c2 = candles[i]      # Third candle
            
            fvg = None
            
            if direction == "SHORT":
                # Bearish FVG: gap above (price needs to rise to fill)
                # c0.low > c2.high creates a gap
                if c0.low > c2.high:
                    gap_top = c0.low
                    gap_bottom = c2.high
                    gap_mid = (gap_top + gap_bottom) / 2
                    
                    fvg = FVG(
                        fvg_type=FVGType.BEARISH,
                        top=gap_top,
                        bottom=gap_bottom,
                        mid=gap_mid,
                        symbol=symbol,
                        timeframe=timeframe,
                        created_at=c1.timestamp,
                        candle_index=i - 1
                    )
                    
            elif direction == "LONG":
                # Bullish FVG: gap below (price needs to drop to fill)
                # c0.high < c2.low creates a gap
                if c0.high < c2.low:
                    gap_top = c2.low
                    gap_bottom = c0.high
                    gap_mid = (gap_top + gap_bottom) / 2
                    
                    fvg = FVG(
                        fvg_type=FVGType.BULLISH,
                        top=gap_top,
                        bottom=gap_bottom,
                        mid=gap_mid,
                        symbol=symbol,
                        timeframe=timeframe,
                        created_at=c1.timestamp,
                        candle_index=i - 1
                    )
            
            if fvg:
                # Validate and enrich FVG
                self._validate_fvg(fvg, candles[-1].close, swing_high, swing_low)
                fvgs.append(fvg)
        
        # Sort by recency (newest first)
        fvgs.sort(key=lambda x: x.created_at, reverse=True)
        
        return fvgs
    
    def _validate_fvg(
        self, 
        fvg: FVG, 
        current_price: float,
        swing_high: Optional[float],
        swing_low: Optional[float]
    ) -> None:
        """Validate FVG and calculate zone info"""
        
        # 1. Calculate size percentage
        fvg.size_pct = (fvg.size / current_price) * 100
        
        # 2. Check size bounds
        if fvg.size_pct < self.config.FVG_MIN_SIZE_PCT:
            fvg.is_valid = False
            fvg.invalidation_reason = f"Too small ({fvg.size_pct:.3f}%)"
            return
        
        if fvg.size_pct > self.config.FVG_MAX_SIZE_PCT:
            fvg.is_valid = False
            fvg.invalidation_reason = f"Too large ({fvg.size_pct:.3f}%)"
            return
        
        # 3. Check age
        if not fvg.is_fresh(self.config.FVG_MAX_AGE_HOURS):
            fvg.is_valid = False
            fvg.invalidation_reason = f"Too old ({fvg.age_hours:.1f}h)"
            return
        
        # 4. Calculate Premium/Discount zone
        if swing_high is not None and swing_low is not None:
            swing_range = swing_high - swing_low
            if swing_range > 0:
                # Calculate FVG position in swing range (0 = bottom, 1 = top)
                fvg.fib_level = (fvg.mid - swing_low) / swing_range
                
                # Classify zone
                if fvg.fib_level >= self.config.PREMIUM_THRESHOLD:
                    fvg.zone_type = ZoneType.PREMIUM
                elif fvg.fib_level <= self.config.DISCOUNT_THRESHOLD:
                    fvg.zone_type = ZoneType.DISCOUNT
                else:
                    fvg.zone_type = ZoneType.EQUILIBRIUM
                
                # Validate zone for direction
                if fvg.fvg_type == FVGType.BEARISH:
                    # Bearish FVG should be in Premium zone
                    if fvg.zone_type != ZoneType.PREMIUM:
                        fvg.is_valid = False
                        fvg.invalidation_reason = f"Not in Premium zone (fib={fvg.fib_level:.2f})"
                        return
                else:
                    # Bullish FVG should be in Discount zone
                    if fvg.zone_type != ZoneType.DISCOUNT:
                        fvg.is_valid = False
                        fvg.invalidation_reason = f"Not in Discount zone (fib={fvg.fib_level:.2f})"
                        return
        
        # 5. Check if already filled
        if fvg.check_fill(current_price):
            fvg.is_valid = False
            fvg.invalidation_reason = "Already filled"
            return
        
        fvg.is_valid = True
    
    def find_best_fvg(
        self,
        candles: List[Candle],
        symbol: str,
        timeframe: str,
        direction: str,
        current_price: float
    ) -> Optional[FVG]:
        """
        Find the best (most recent, valid, unfilled) FVG
        
        Args:
            candles: OHLCV candle data
            symbol: Trading pair
            timeframe: "1h" or "5m"
            direction: "LONG" or "SHORT"
            current_price: Current market price
            
        Returns:
            Best FVG or None if no valid FVG found
        """
        # Calculate swing points for zone validation
        swing_high, swing_low = self._find_swing_range(candles)
        
        # Detect all FVGs
        fvgs = self.detect_fvgs(
            candles=candles,
            symbol=symbol,
            timeframe=timeframe,
            direction=direction,
            swing_high=swing_high,
            swing_low=swing_low
        )
        
        # Filter valid, unfilled FVGs
        valid_fvgs = [f for f in fvgs if f.is_valid and not f.filled]
        
        if not valid_fvgs:
            return None
        
        # Return most recent valid FVG
        return valid_fvgs[0]
    
    def _find_swing_range(
        self, 
        candles: List[Candle],
        lookback: int = 50
    ) -> Tuple[float, float]:
        """Find swing high and low from recent candles"""
        if not candles:
            return 0, 0
        
        recent = candles[-lookback:] if len(candles) >= lookback else candles
        
        swing_high = max(c.high for c in recent)
        swing_low = min(c.low for c in recent)
        
        return swing_high, swing_low
    
    def update_fvg_status(self, fvg: FVG, current_price: float) -> bool:
        """
        Update FVG fill status with current price
        
        Returns:
            True if price is now in the FVG (potential entry zone)
        """
        # Check if filled
        fvg.check_fill(current_price)
        
        # Check if price is in the gap
        return fvg.is_price_in_gap(current_price)
    
    def get_entry_zone(self, fvg: FVG) -> Tuple[float, float, float]:
        """
        Get entry zone from FVG
        
        Returns:
            Tuple of (entry_price, zone_top, zone_bottom)
        """
        # Entry at 50% of FVG (OTE - Optimal Trade Entry)
        entry = fvg.mid
        return entry, fvg.top, fvg.bottom


def candles_from_api_data(data: List[dict]) -> List[Candle]:
    """
    Convert API candle data to Candle objects
    
    Expected format: [timestamp, open, high, low, close, volume]
    or dict with keys: timestamp/time, open, high, low, close, volume
    """
    candles = []
    
    for item in data:
        if isinstance(item, (list, tuple)):
            # Array format: [timestamp, open, high, low, close, volume]
            ts = item[0]
            if isinstance(ts, (int, float)):
                # Milliseconds to datetime
                if ts > 10000000000:
                    ts = ts / 1000
                timestamp = datetime.utcfromtimestamp(ts)
            else:
                timestamp = ts
            
            candles.append(Candle(
                timestamp=timestamp,
                open=float(item[1]),
                high=float(item[2]),
                low=float(item[3]),
                close=float(item[4]),
                volume=float(item[5]) if len(item) > 5 else 0
            ))
        elif isinstance(item, dict):
            # Dict format
            ts = item.get('timestamp') or item.get('time') or item.get('t')
            if isinstance(ts, (int, float)):
                if ts > 10000000000:
                    ts = ts / 1000
                timestamp = datetime.utcfromtimestamp(ts)
            else:
                timestamp = ts
            
            candles.append(Candle(
                timestamp=timestamp,
                open=float(item.get('open') or item.get('o')),
                high=float(item.get('high') or item.get('h')),
                low=float(item.get('low') or item.get('l')),
                close=float(item.get('close') or item.get('c')),
                volume=float(item.get('volume') or item.get('v') or 0)
            ))
    
    # Sort by timestamp (oldest first)
    candles.sort(key=lambda x: x.timestamp)
    
    return candles
