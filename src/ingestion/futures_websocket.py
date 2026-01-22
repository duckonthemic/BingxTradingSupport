"""
BingX Futures WebSocket Client for Real-time Market Data
Connects to wss://open-api-cswap-ws.bingx.com/market
Features:
- Auto-reconnection with exponential backoff
- Circuit breaker pattern
- Real-time ticker streaming
- BTC monitoring for mood check
"""

import asyncio
import json
import logging
import gzip
from typing import Callable, Awaitable, Optional, Dict, Set, List
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

import aiohttp

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"      # Normal operation
    OPEN = "OPEN"          # Blocking requests
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


@dataclass
class CircuitBreaker:
    """Circuit breaker for API protection."""
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    half_open_max_calls: int = 3
    
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure_time: Optional[datetime] = None
    half_open_calls: int = 0
    
    def record_success(self):
        """Record successful call."""
        self.failure_count = 0
        self.half_open_calls = 0
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.CLOSED
            logger.info("🟢 Circuit breaker CLOSED - recovered")
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            self.state = CircuitState.OPEN
            logger.warning("🔴 Circuit breaker OPEN - failed during recovery")
        elif self.failure_count >= self.failure_threshold:
            self.state = CircuitState.OPEN
            logger.warning(f"🔴 Circuit breaker OPEN - {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if request can be made."""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check if recovery timeout passed
            if self.last_failure_time:
                elapsed = (datetime.now() - self.last_failure_time).total_seconds()
                if elapsed >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.half_open_calls = 0
                    logger.info("🟡 Circuit breaker HALF_OPEN - testing recovery")
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False
        
        return False


class FuturesWebSocketClient:
    """
    Real-time WebSocket client for BingX Futures.
    Streams ticker data and monitors BTC for mood check.
    """
    
    # Public market WebSocket (no auth required)
    WS_URL = "wss://open-api-ws.bingx.com/market"
    
    def __init__(
        self,
        on_ticker: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_btc_update: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        self.on_ticker = on_ticker
        self.on_btc_update = on_btc_update
        
        self._ws = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._running = False
        self._reconnect_delay = 1
        self._max_reconnect_delay = 60
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker()
        
        # Stats
        self._message_count = 0
        self._last_message_time: Optional[datetime] = None
        self._connected_at: Optional[datetime] = None
        
        # BTC tracking for mood check
        self._btc_prices: list = []  # Last 15 minutes of prices
        self._btc_last_update: Optional[datetime] = None
        
        # Subscribed symbols
        self._subscribed: Set[str] = set()
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect_once()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()
    
    async def connect_once(self):
        """Connect to WebSocket once (for context manager use)."""
        self._running = True
        self._session = aiohttp.ClientSession()
        
        try:
            self._ws = await self._session.ws_connect(
                self.WS_URL,
                heartbeat=20,
                timeout=aiohttp.ClientTimeout(total=30)
            )
            self._connected_at = datetime.now()
            logger.info("✅ Futures WebSocket connected (context manager)")
            return self
        except Exception as e:
            logger.error(f"❌ WebSocket connect failed: {e}")
            raise
    
    async def receive(self):
        """Async generator to receive messages."""
        if not self._ws:
            return
            
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    yield json.loads(msg.data)
                except:
                    pass
            elif msg.type == aiohttp.WSMsgType.BINARY:
                try:
                    data = gzip.decompress(msg.data).decode('utf-8')
                    yield json.loads(data)
                except:
                    pass
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f"WebSocket error: {self._ws.exception()}")
                break
    
    async def disconnect(self):
        """Disconnect WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
            self._ws = None
        if self._session:
            await self._session.close()
            self._session = None
        logger.info("🔌 WebSocket disconnected")
    
    async def connect(self):
        """Connect to WebSocket and start streaming."""
        self._running = True
        self._session = aiohttp.ClientSession()
        
        logger.info(f"🔌 Connecting to BingX Futures WebSocket...")
        
        while self._running:
            if not self.circuit_breaker.can_execute():
                logger.warning("⏸️ Circuit breaker OPEN - waiting...")
                await asyncio.sleep(5)
                continue
            
            try:
                async with self._session.ws_connect(
                    self.WS_URL,
                    heartbeat=20,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as ws:
                    self._ws = ws
                    self._connected_at = datetime.now()
                    self._reconnect_delay = 1
                    self.circuit_breaker.record_success()
                    
                    logger.info("✅ Futures WebSocket connected")
                    
                    # Subscribe to BTC for mood check
                    await self._subscribe_btc()
                    
                    # Message loop
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            await self._handle_message(msg.data)
                        elif msg.type == aiohttp.WSMsgType.BINARY:
                            # Decompress gzip
                            try:
                                data = gzip.decompress(msg.data).decode('utf-8')
                                await self._handle_message(data)
                            except:
                                pass
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"WebSocket error: {ws.exception()}")
                            break
                    
            except aiohttp.ClientError as e:
                logger.error(f"❌ WebSocket connection error: {e}")
                self.circuit_breaker.record_failure()
                await self._reconnect()
            except Exception as e:
                logger.error(f"❌ Unexpected WebSocket error: {e}")
                self.circuit_breaker.record_failure()
                await self._reconnect()
    
    async def _subscribe_btc(self):
        """Subscribe to BTC-USDT ticker for mood monitoring."""
        subscribe_msg = {
            "id": "btc_ticker",
            "reqType": "sub",
            "dataType": "BTC-USDT@ticker"
        }
        if self._ws:
            await self._ws.send_str(json.dumps(subscribe_msg))
            self._subscribed.add("BTC-USDT")
            logger.info("📡 Subscribed to BTC-USDT ticker")
    
    async def subscribe_symbol(self, symbol: str):
        """Subscribe to a specific symbol's ticker."""
        if symbol in self._subscribed:
            return
        
        subscribe_msg = {
            "id": f"{symbol}_ticker",
            "reqType": "sub",
            "dataType": f"{symbol}@ticker"
        }
        if self._ws:
            await self._ws.send_str(json.dumps(subscribe_msg))
            self._subscribed.add(symbol)
            logger.debug(f"📡 Subscribed to {symbol}")
    
    async def _handle_message(self, raw_message: str):
        """Handle incoming WebSocket message."""
        try:
            data = json.loads(raw_message)
            
            # Handle ping/pong
            if "ping" in data:
                pong_msg = {"pong": data["ping"]}
                await self._ws.send_str(json.dumps(pong_msg))
                return
            
            # Handle ticker data
            if data.get("dataType", "").endswith("@ticker"):
                ticker_data = data.get("data", {})
                symbol = ticker_data.get("s", "")
                
                self._message_count += 1
                self._last_message_time = datetime.now()
                
                # BTC special handling for mood check
                if symbol == "BTC-USDT":
                    await self._update_btc_price(ticker_data)
                    if self.on_btc_update:
                        await self.on_btc_update(ticker_data)
                
                # General ticker callback
                if self.on_ticker:
                    await self.on_ticker(ticker_data)
                    
        except json.JSONDecodeError:
            pass
        except Exception as e:
            logger.debug(f"Message handling error: {e}")
    
    async def _update_btc_price(self, ticker: dict):
        """Track BTC price for mood check."""
        try:
            price = float(ticker.get("c", 0))  # Current price
            now = datetime.now()
            
            self._btc_prices.append({
                "price": price,
                "time": now
            })
            
            # Keep only last 15 minutes
            cutoff = now - timedelta(minutes=15)
            self._btc_prices = [p for p in self._btc_prices if p["time"] > cutoff]
            self._btc_last_update = now
            
        except Exception as e:
            logger.debug(f"BTC price update error: {e}")
    
    def get_btc_change_15m(self) -> float:
        """Get BTC price change in last 15 minutes (%)."""
        if len(self._btc_prices) < 2:
            return 0.0
        
        oldest = self._btc_prices[0]["price"]
        newest = self._btc_prices[-1]["price"]
        
        if oldest > 0:
            return ((newest - oldest) / oldest) * 100
        return 0.0
    
    def is_btc_dumping(self, threshold: float = -0.5) -> bool:
        """Check if BTC is dumping (default: >0.5% drop in 15m)."""
        change = self.get_btc_change_15m()
        return change < threshold
    
    async def _reconnect(self):
        """Reconnect with exponential backoff."""
        if not self._running:
            return
        
        logger.info(f"🔄 Reconnecting in {self._reconnect_delay}s...")
        await asyncio.sleep(self._reconnect_delay)
        
        self._reconnect_delay = min(
            self._reconnect_delay * 2,
            self._max_reconnect_delay
        )
    
    async def disconnect(self):
        """Disconnect from WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        logger.info("WebSocket disconnected")
    
    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._ws is not None and not self._ws.closed
    
    @property
    def stats(self) -> dict:
        """Get connection statistics."""
        uptime = None
        if self._connected_at:
            uptime = str(datetime.now() - self._connected_at)
        
        return {
            "connected": self.is_connected,
            "uptime": uptime,
            "message_count": self._message_count,
            "last_message": self._last_message_time.isoformat() if self._last_message_time else None,
            "circuit_state": self.circuit_breaker.state.value,
            "btc_change_15m": f"{self.get_btc_change_15m():+.2f}%",
            "subscribed_count": len(self._subscribed)
        }
    
    async def subscribe_symbols(self, symbols: List[str]):
        """
        Subscribe to multiple symbols at once.
        
        Args:
            symbols: List of symbol pairs (e.g., ["ETH-USDT", "SOL-USDT"])
        """
        if not self._ws:
            logger.warning("WebSocket not connected - cannot subscribe")
            return
        
        for symbol in symbols:
            if symbol not in self._subscribed:
                await self.subscribe_symbol(symbol)
                await asyncio.sleep(0.1)  # Small delay to avoid rate limit
        
        logger.info(f"📡 Subscribed to {len(self._subscribed)} symbols")
    
    async def unsubscribe_symbol(self, symbol: str):
        """Unsubscribe from a symbol's ticker."""
        if symbol not in self._subscribed:
            return
        
        unsubscribe_msg = {
            "id": f"{symbol}_unsub",
            "reqType": "unsub",
            "dataType": f"{symbol}@ticker"
        }
        
        if self._ws:
            await self._ws.send_str(json.dumps(unsubscribe_msg))
            self._subscribed.discard(symbol)
            logger.debug(f"📴 Unsubscribed from {symbol}")
    
    def get_subscribed_symbols(self) -> Set[str]:
        """Get currently subscribed symbols."""
        return self._subscribed.copy()


class RetryHandler:
    """
    Retry handler with exponential backoff for API calls.
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.exponential_base = exponential_base
    
    async def execute(
        self,
        func: Callable[[], Awaitable],
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """
        Execute function with retry logic.
        
        Args:
            func: Async function to execute
            circuit_breaker: Optional circuit breaker
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                raise Exception("Circuit breaker is OPEN")
            
            try:
                result = await func()
                if circuit_breaker:
                    circuit_breaker.record_success()
                return result
                
            except Exception as e:
                last_exception = e
                if circuit_breaker:
                    circuit_breaker.record_failure()
                
                if attempt < self.max_retries:
                    delay = min(
                        self.base_delay * (self.exponential_base ** attempt),
                        self.max_delay
                    )
                    logger.warning(f"Retry {attempt + 1}/{self.max_retries} after {delay:.1f}s: {e}")
                    await asyncio.sleep(delay)
        
        raise last_exception


# ==================== Test ====================

async def test_futures_ws():
    """Test Futures WebSocket."""
    print("=" * 60)
    print("Testing BingX Futures WebSocket")
    print("=" * 60)
    
    message_count = 0
    
    async def on_ticker(data: dict):
        nonlocal message_count
        message_count += 1
        symbol = data.get("s", "")
        price = data.get("c", "")
        print(f"  [{message_count}] {symbol}: ${price}")
    
    async def on_btc(data: dict):
        price = data.get("c", "")
        print(f"  📊 BTC Update: ${price}")
    
    client = FuturesWebSocketClient(
        on_ticker=on_ticker,
        on_btc_update=on_btc
    )
    
    async def auto_stop():
        await asyncio.sleep(15)
        print(f"\n⏱️ Test complete - stopping...")
        print(f"📈 Stats: {client.stats}")
        await client.disconnect()
    
    asyncio.create_task(auto_stop())
    
    try:
        await client.connect()
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(test_futures_ws())
