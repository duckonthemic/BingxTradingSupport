"""
Market Context Manager v2.0.
Monitors BTC trend, BTC.D, and market conditions to determine overall market state.

Features:
- BTC Trend tracking (H4 EMA34/89)
- BTC.D (Dominance) monitoring
- Context freshness check (max 2 minutes old)
- Auto-pause on unfavorable conditions
"""

import asyncio
import logging
from enum import Enum
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional

from ..config import config
from ..storage.redis_client import RedisClient
from ..ingestion.rest_client import BingXRestClient
from ..analysis.indicators import IndicatorCalculator

logger = logging.getLogger(__name__)

# Context freshness threshold (2 minutes)
CONTEXT_MAX_AGE_SECONDS = 120


class MarketState(Enum):
    """Overall market state for trading decisions."""
    FAVORABLE = "FAVORABLE"   # Good for alerts
    CAUTION = "CAUTION"       # Reduced confidence
    WARNING = "WARNING"       # BTC weak - allow SHORT Diamond only
    DANGER = "DANGER"         # BTC dumping - block LONG, allow SHORT Diamond
    CRASH = "CRASH"           # Emergency - block all


class BTCTrend(Enum):
    """BTC trend classification."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"


@dataclass
class MarketContext:
    """Current market context data."""
    timestamp: int
    
    # BTC data
    btc_price: float
    btc_ema34_h4: float
    btc_ema89_h4: float
    btc_ema200_h4: float = 0.0  # For Market Regime detection
    btc_change_1h_pct: float = 0.0
    btc_change_4h_pct: float = 0.0
    btc_trend: str = "SIDEWAYS"
    market_regime: str = "NEUTRAL"  # BULLISH/BEARISH/NEUTRAL based on EMA200
    
    # Market state
    market_state: str = "CAUTION"
    alerts_allowed: bool = True
    long_allowed: bool = True
    short_allowed: bool = True
    short_diamond_allowed: bool = True  # Diamond SHORT allowed even in DANGER
    ema89_distance_pct: float = 0.0  # Distance from EMA89 for monitoring
    
    # Context freshness
    is_fresh: bool = True
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> "MarketContext":
        return cls(**data)
    
    @property
    def age_seconds(self) -> float:
        """Get age of context in seconds."""
        now_ms = int(datetime.now().timestamp() * 1000)
        return (now_ms - self.timestamp) / 1000
    
    def check_freshness(self) -> bool:
        """Check if context is still fresh (< 2 minutes old)."""
        return self.age_seconds < CONTEXT_MAX_AGE_SECONDS


class ContextManager:
    """
    Manages market context by monitoring BTC and overall market conditions.
    Updates context every 30 seconds and stores in Redis.
    """
    
    def __init__(self, redis_client: RedisClient):
        self.redis = redis_client
        self.rest_client = BingXRestClient()
        self.indicator_calc = IndicatorCalculator()
        self.update_interval = config.timing.context_update_interval
        self._running = False
        self._current_context: Optional[MarketContext] = None
    
    async def start(self):
        """Start the context manager background loop."""
        self._running = True
        await self.rest_client.connect()
        
        logger.info(f"🌐 Context Manager started (update every {self.update_interval}s)")
        
        while self._running:
            try:
                await self._update_context()
            except Exception as e:
                logger.error(f"Context update failed: {e}")
            
            await asyncio.sleep(self.update_interval)
    
    async def stop(self):
        """Stop the context manager."""
        self._running = False
        await self.rest_client.disconnect()
        logger.info("Context Manager stopped")
    
    async def _update_context(self):
        """Fetch data and update market context."""
        # Fetch BTC data
        btc_ticker = await self.rest_client.get_ticker("BTC-USDT")
        if not btc_ticker:
            logger.warning("Failed to fetch BTC ticker")
            return
        
        # Fetch BTC klines for EMA calculation
        klines_h4 = await self.rest_client.get_klines("BTC-USDT", "4h", 100)
        if not klines_h4:
            logger.warning("Failed to fetch BTC klines")
            return
        
        # Calculate EMAs
        btc_price = float(btc_ticker.get("lastPrice", 0))
        
        # Convert klines to DataFrame for EMA calculation
        import pandas as pd
        import ta
        
        # BingX klines format may vary, extract close prices
        if klines_h4 and isinstance(klines_h4[0], list):
            # Array format: [time, open, high, low, close, volume, ...]
            closes = [float(k[4]) for k in klines_h4]  # index 4 = close
        elif klines_h4 and isinstance(klines_h4[0], dict):
            closes = [float(k.get('close', 0)) for k in klines_h4]
        else:
            logger.warning("Unknown klines format")
            return
        
        df = pd.DataFrame({'close': closes})
        
        ema34 = float(ta.trend.ema_indicator(df['close'], window=34).iloc[-1])
        ema89 = float(ta.trend.ema_indicator(df['close'], window=89).iloc[-1])
        
        # Calculate EMA200 for Market Regime detection
        if len(closes) >= 200:
            ema200 = float(ta.trend.ema_indicator(df['close'], window=200).iloc[-1])
        else:
            # Fallback to EMA89 if not enough data
            ema200 = ema89
        
        # Calculate 1h change
        change_str = str(btc_ticker.get("priceChangePercent", "0")).replace("%", "")
        btc_change_1h = float(change_str) if change_str else 0
        
        # Determine BTC trend
        if btc_price > ema34 and ema34 > ema89:
            btc_trend = BTCTrend.BULLISH
        elif btc_price < ema89:
            btc_trend = BTCTrend.BEARISH
        else:
            btc_trend = BTCTrend.SIDEWAYS
        
        # Determine Market Regime based on EMA200
        ema200_buffer = ema200 * 0.01  # 1% buffer zone
        if btc_price > ema200 + ema200_buffer:
            market_regime = "BULLISH"
        elif btc_price < ema200 - ema200_buffer:
            market_regime = "BEARISH"
        else:
            market_regime = "NEUTRAL"
        
        # Determine market state
        market_state = self._determine_market_state(
            btc_price, ema34, ema89, btc_change_1h
        )
        
        # Calculate EMA89 distance for monitoring
        ema89_distance_pct = ((btc_price - ema89) / ema89) * 100 if ema89 > 0 else 0
        
        # Determine permissions based on market state
        # CRASH: Block everything
        # DANGER: Block LONG, allow SHORT Diamond only
        # WARNING: Block LONG, allow SHORT Diamond only
        # CAUTION/FAVORABLE: Allow all
        
        if market_state == MarketState.CRASH:
            alerts_allowed = False
            long_allowed = False
            short_allowed = False
            short_diamond_allowed = False
        elif market_state in [MarketState.DANGER, MarketState.WARNING]:
            alerts_allowed = True  # Allow alerts but with restrictions
            long_allowed = False
            short_allowed = False  # Block Gold SHORT
            short_diamond_allowed = True  # Allow Diamond SHORT
        else:
            alerts_allowed = True
            long_allowed = True
            short_allowed = True
            short_diamond_allowed = True
        
        # Additional LONG restrictions
        if btc_trend == BTCTrend.BEARISH:
            long_allowed = False
        if btc_change_1h < -1.5:  # BTC dumping > 1.5%
            long_allowed = False
        
        # Create context
        self._current_context = MarketContext(
            timestamp=int(datetime.now().timestamp() * 1000),
            btc_price=btc_price,
            btc_ema34_h4=ema34,
            btc_ema89_h4=ema89,
            btc_ema200_h4=ema200,
            btc_change_1h_pct=btc_change_1h,
            btc_change_4h_pct=0.0,  # TODO: Calculate from klines
            btc_trend=btc_trend.value,
            market_regime=market_regime,
            market_state=market_state.value,
            alerts_allowed=alerts_allowed,
            long_allowed=long_allowed,
            short_allowed=short_allowed,
            short_diamond_allowed=short_diamond_allowed,
            ema89_distance_pct=ema89_distance_pct,
            is_fresh=True
        )
        
        # Save to Redis
        await self.redis.save_market_context(self._current_context.to_dict())
        
        # Enhanced logging with EMA89 distance
        distance_emoji = "⚠️" if ema89_distance_pct < -1 else "✅" if ema89_distance_pct > 1 else "🔄"
        logger.info(
            f"📊 Context: BTC ${btc_price:,.0f} | "
            f"Trend: {btc_trend.value} | "
            f"Regime: {market_regime} | "
            f"State: {market_state.value} | "
            f"EMA89: {distance_emoji}{ema89_distance_pct:+.1f}%"
        )
    
    def _determine_market_state(
        self,
        price: float,
        ema34: float,
        ema89: float,
        change_1h: float
    ) -> MarketState:
        """
        Determine overall market state based on BTC conditions.
        
        REFINED STATES (v2.0):
        - CRASH: BTC drop > 5%/1h (emergency stop all)
        - DANGER: BTC drop > 3%/1h OR (drop > 2% + below EMA89)
        - WARNING: BTC below EMA89 * 0.98 (allow SHORT Diamond)
        - CAUTION: BTC weak/sideways
        - FAVORABLE: BTC strong uptrend
        """
        # CRASH: Emergency stop - BTC flash crash
        if change_1h < -5.0:
            return MarketState.CRASH
        
        # DANGER: BTC dumping hard
        if change_1h < -3.0:
            return MarketState.DANGER
        
        # DANGER: Below EMA89 + moderate dump (combined)
        if price < ema89 and change_1h < -2.0:
            return MarketState.DANGER
        
        # WARNING: Below EMA89 with 2% buffer
        if price < ema89 * 0.98:
            return MarketState.WARNING
        
        # WARNING: Moderate dump (between -1.5% and -2%)
        if change_1h < -1.5:
            return MarketState.WARNING
        
        # FAVORABLE conditions
        if price > ema34 and ema34 > ema89:
            if change_1h >= 0:
                return MarketState.FAVORABLE
        
        # Default: CAUTION
        return MarketState.CAUTION
    
    async def get_current(self) -> Optional[MarketContext]:
        """Get current market context."""
        if self._current_context:
            return self._current_context
        
        # Try to load from Redis
        data = await self.redis.get_market_context()
        if data:
            return MarketContext.from_dict(data)
        return None
    
    @property
    def is_running(self) -> bool:
        return self._running


# ==================== Test ====================

async def test_context_manager():
    """Test context manager."""
    print("=" * 50)
    print("Testing Context Manager...")
    print("=" * 50)
    
    redis = RedisClient()
    await redis.connect()
    
    cm = ContextManager(redis)
    
    # Do one update manually
    await cm.rest_client.connect()
    await cm._update_context()
    
    ctx = await cm.get_current()
    if ctx:
        print(f"\n✅ Context fetched:")
        print(f"   BTC Price: ${ctx.btc_price:,.2f}")
        print(f"   BTC Trend: {ctx.btc_trend}")
        print(f"   EMA34: ${ctx.btc_ema34_h4:,.2f}")
        print(f"   EMA89: ${ctx.btc_ema89_h4:,.2f}")
        print(f"   Market State: {ctx.market_state}")
        print(f"   Alerts Allowed: {ctx.alerts_allowed}")
        print("\n✅ Test passed!")
    else:
        print("❌ Failed to get context")
    
    await cm.rest_client.disconnect()
    await redis.disconnect()


if __name__ == "__main__":
    asyncio.run(test_context_manager())
