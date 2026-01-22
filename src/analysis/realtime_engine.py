"""
Real-time Signal Engine - WebSocket-based trading signal detection.
Replaces polling with instant price feed analysis.

Features:
- Real-time price streaming for all coins
- Instant signal detection on price/volume spikes
- Smart batching to avoid API overload
- Fallback to polling if WebSocket fails
"""

import asyncio
import logging
from typing import Callable, Awaitable, Optional, Dict, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = logging.getLogger(__name__)


@dataclass
class PriceCandle:
    """Real-time price candle data."""
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timestamp: datetime
    
    @property
    def body_size(self) -> float:
        """Candle body size as percentage."""
        if self.open == 0:
            return 0
        return abs(self.close - self.open) / self.open * 100
    
    @property
    def upper_wick(self) -> float:
        """Upper wick as percentage of range."""
        range_size = self.high - self.low
        if range_size == 0:
            return 0
        body_top = max(self.open, self.close)
        return (self.high - body_top) / range_size * 100
    
    @property
    def is_shooting_star(self) -> bool:
        """Quick check for shooting star pattern."""
        return self.upper_wick >= 50 and self.body_size < 30


@dataclass
class SignalTrigger:
    """Real-time signal trigger event."""
    symbol: str
    trigger_type: str  # 'VOLUME_SPIKE', 'PRICE_BREAKOUT', 'PATTERN'
    price: float
    volume_ratio: float
    timestamp: datetime
    metadata: Dict = field(default_factory=dict)


class RealTimeSignalEngine:
    """
    Real-time signal detection engine.
    
    Instead of polling every 90s, this engine:
    1. Streams real-time prices via WebSocket
    2. Detects instant triggers (volume spike, breakout)
    3. Only fetches full candle data when trigger detected
    4. Runs full analysis only on triggered coins
    """
    
    # Trigger thresholds
    VOLUME_SPIKE_RATIO = 2.0      # 2x normal volume
    PRICE_MOVE_THRESHOLD = 0.5    # 0.5% price move
    RAPID_MOVE_WINDOW = 60        # 1 minute window
    
    def __init__(
        self,
        on_signal_trigger: Optional[Callable[[SignalTrigger], Awaitable[None]]] = None,
    ):
        self.on_signal_trigger = on_signal_trigger
        
        # Price history per symbol (last 5 minutes of ticks)
        self._price_history: Dict[str, List[Dict]] = defaultdict(list)
        
        # Volume baseline per symbol (avg volume)
        self._volume_baseline: Dict[str, float] = {}
        
        # Recent triggers (cooldown tracking)
        self._recent_triggers: Dict[str, datetime] = {}
        self.trigger_cooldown = 60  # 60s cooldown per symbol
        
        # Stats
        self._stats = {
            "ticks_processed": 0,
            "triggers_fired": 0,
            "volume_spikes": 0,
            "price_breakouts": 0,
            "patterns_detected": 0,
        }
        
        # Running state
        self._running = False
    
    async def process_tick(self, ticker_data: Dict):
        """
        Process incoming real-time ticker.
        Called for every price update from WebSocket.
        """
        try:
            symbol = ticker_data.get("s", "")
            if not symbol:
                return
            
            price = float(ticker_data.get("c", 0))  # Close/current price
            volume = float(ticker_data.get("v", 0))  # Volume
            high = float(ticker_data.get("h", price))
            low = float(ticker_data.get("l", price))
            open_price = float(ticker_data.get("o", price))
            
            now = datetime.now()
            
            # Store tick
            tick = {
                "price": price,
                "volume": volume,
                "high": high,
                "low": low,
                "open": open_price,
                "time": now
            }
            
            self._price_history[symbol].append(tick)
            self._stats["ticks_processed"] += 1
            
            # Keep only last 5 minutes
            cutoff = now - timedelta(minutes=5)
            self._price_history[symbol] = [
                t for t in self._price_history[symbol] 
                if t["time"] > cutoff
            ]
            
            # Check for triggers
            await self._check_triggers(symbol, tick)
            
        except Exception as e:
            logger.debug(f"Tick processing error: {e}")
    
    async def _check_triggers(self, symbol: str, tick: Dict):
        """Check if current tick triggers a signal."""
        # Cooldown check
        if symbol in self._recent_triggers:
            elapsed = (datetime.now() - self._recent_triggers[symbol]).seconds
            if elapsed < self.trigger_cooldown:
                return
        
        history = self._price_history[symbol]
        if len(history) < 10:  # Need some history
            return
        
        # === 1. Volume Spike Detection ===
        volume_trigger = await self._check_volume_spike(symbol, tick)
        if volume_trigger:
            await self._fire_trigger(volume_trigger)
            return
        
        # === 2. Rapid Price Move Detection ===
        price_trigger = await self._check_price_breakout(symbol, tick)
        if price_trigger:
            await self._fire_trigger(price_trigger)
            return
        
        # === 3. Pattern Detection (Shooting Star) ===
        pattern_trigger = await self._check_pattern(symbol, tick)
        if pattern_trigger:
            await self._fire_trigger(pattern_trigger)
            return
    
    async def _check_volume_spike(self, symbol: str, tick: Dict) -> Optional[SignalTrigger]:
        """Detect volume spike."""
        history = self._price_history[symbol]
        
        # Calculate average volume from history
        volumes = [t["volume"] for t in history[:-1] if t["volume"] > 0]
        if not volumes:
            return None
        
        avg_volume = sum(volumes) / len(volumes)
        current_volume = tick["volume"]
        
        if avg_volume > 0:
            ratio = current_volume / avg_volume
            
            if ratio >= self.VOLUME_SPIKE_RATIO:
                self._stats["volume_spikes"] += 1
                return SignalTrigger(
                    symbol=symbol,
                    trigger_type="VOLUME_SPIKE",
                    price=tick["price"],
                    volume_ratio=ratio,
                    timestamp=datetime.now(),
                    metadata={"avg_volume": avg_volume, "current_volume": current_volume}
                )
        
        return None
    
    async def _check_price_breakout(self, symbol: str, tick: Dict) -> Optional[SignalTrigger]:
        """Detect rapid price movement."""
        history = self._price_history[symbol]
        
        # Get price 1 minute ago
        one_min_ago = datetime.now() - timedelta(minutes=1)
        old_ticks = [t for t in history if t["time"] < one_min_ago]
        
        if not old_ticks:
            return None
        
        old_price = old_ticks[-1]["price"]
        current_price = tick["price"]
        
        if old_price > 0:
            price_change = abs((current_price - old_price) / old_price) * 100
            
            if price_change >= self.PRICE_MOVE_THRESHOLD:
                self._stats["price_breakouts"] += 1
                direction = "UP" if current_price > old_price else "DOWN"
                return SignalTrigger(
                    symbol=symbol,
                    trigger_type="PRICE_BREAKOUT",
                    price=current_price,
                    volume_ratio=1.0,
                    timestamp=datetime.now(),
                    metadata={
                        "old_price": old_price,
                        "change_pct": price_change,
                        "direction": direction
                    }
                )
        
        return None
    
    async def _check_pattern(self, symbol: str, tick: Dict) -> Optional[SignalTrigger]:
        """Detect candlestick patterns in real-time."""
        # Create candle from tick
        candle = PriceCandle(
            symbol=symbol,
            open=tick["open"],
            high=tick["high"],
            low=tick["low"],
            close=tick["price"],
            volume=tick["volume"],
            timestamp=datetime.now()
        )
        
        # Check shooting star
        if candle.is_shooting_star:
            self._stats["patterns_detected"] += 1
            return SignalTrigger(
                symbol=symbol,
                trigger_type="PATTERN",
                price=tick["price"],
                volume_ratio=1.0,
                timestamp=datetime.now(),
                metadata={
                    "pattern": "SHOOTING_STAR",
                    "upper_wick": candle.upper_wick,
                    "body_size": candle.body_size
                }
            )
        
        return None
    
    async def _fire_trigger(self, trigger: SignalTrigger):
        """Fire signal trigger event."""
        self._recent_triggers[trigger.symbol] = datetime.now()
        self._stats["triggers_fired"] += 1
        
        logger.info(f"⚡ TRIGGER: {trigger.symbol} - {trigger.trigger_type} @ ${trigger.price:.4f}")
        
        if self.on_signal_trigger:
            await self.on_signal_trigger(trigger)
    
    def get_stats(self) -> Dict:
        """Get engine statistics."""
        return {
            **self._stats,
            "symbols_tracked": len(self._price_history),
            "active_triggers": len(self._recent_triggers)
        }
    
    def reset_cooldown(self, symbol: str):
        """Reset cooldown for a symbol."""
        if symbol in self._recent_triggers:
            del self._recent_triggers[symbol]
