"""
Redis client for BingX Zone Alert Bot.
Handles connection, state management, and cooldown tracking.
"""

import json
import logging
from typing import Optional, Any
import redis.asyncio as redis

from ..config import config

logger = logging.getLogger(__name__)


class RedisClient:
    """Async Redis client with JSON support and cooldown management."""
    
    def __init__(self):
        self.host = config.redis.host
        self.port = config.redis.port
        self.db = config.redis.db
        self.password = config.redis.password or None
        self._client: Optional[redis.Redis] = None
        self._connected = False
    
    async def connect(self) -> "RedisClient":
        """
        Establish connection to Redis server.
        Returns self for method chaining.
        """
        try:
            self._client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
            # Test connection
            await self._client.ping()
            self._connected = True
            logger.info(f"✅ Redis connected: {self.host}:{self.port}/{self.db}")
            return self
        except redis.ConnectionError as e:
            logger.error(f"❌ Redis connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close Redis connection."""
        if self._client:
            await self._client.aclose()
            self._connected = False
            logger.info("Redis disconnected")
    
    @property
    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._connected
    
    # ==================== JSON Operations ====================
    
    async def set_json(self, key: str, value: Any, expire: Optional[int] = None) -> bool:
        """
        Store a value as JSON.
        
        Args:
            key: Redis key
            value: Any JSON-serializable value
            expire: Optional TTL in seconds
            
        Returns:
            True if successful
        """
        try:
            json_data = json.dumps(value, default=str)
            if expire:
                await self._client.setex(key, expire, json_data)
            else:
                await self._client.set(key, json_data)
            return True
        except Exception as e:
            logger.error(f"Redis set_json error: {e}")
            return False
    
    async def get_json(self, key: str) -> Optional[Any]:
        """
        Retrieve and parse JSON value.
        
        Args:
            key: Redis key
            
        Returns:
            Parsed JSON value or None if not found
        """
        try:
            data = await self._client.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get_json error: {e}")
            return None
    
    async def delete(self, key: str) -> bool:
        """Delete a key."""
        try:
            await self._client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    # ==================== Cooldown Management ====================
    
    async def set_cooldown(self, coin: str, zone_type: str, seconds: int) -> bool:
        """
        Set a cooldown for a specific coin and zone type.
        This prevents sending duplicate alerts.
        
        Args:
            coin: Coin symbol (e.g., "PEPE-USDT")
            zone_type: Zone type (e.g., "BREAKOUT_ATTEMPT")
            seconds: Cooldown duration in seconds
            
        Returns:
            True if successful
        """
        key = f"cooldown:{coin}:{zone_type}"
        try:
            await self._client.setex(key, seconds, "1")
            logger.debug(f"Cooldown set: {key} for {seconds}s")
            return True
        except Exception as e:
            logger.error(f"Redis set_cooldown error: {e}")
            return False
    
    async def check_cooldown(self, coin: str, zone_type: str) -> bool:
        """
        Check if a cooldown is active.
        
        Args:
            coin: Coin symbol
            zone_type: Zone type
            
        Returns:
            True if cooldown is active (should NOT send alert)
        """
        key = f"cooldown:{coin}:{zone_type}"
        try:
            exists = await self._client.exists(key)
            return exists > 0
        except Exception as e:
            logger.error(f"Redis check_cooldown error: {e}")
            return False  # If error, allow alert
    
    async def get_cooldown_ttl(self, coin: str, zone_type: str) -> int:
        """
        Get remaining cooldown time.
        
        Returns:
            Remaining seconds, or 0 if no cooldown
        """
        key = f"cooldown:{coin}:{zone_type}"
        try:
            ttl = await self._client.ttl(key)
            return max(0, ttl)
        except Exception as e:
            logger.error(f"Redis get_cooldown_ttl error: {e}")
            return 0
    
    # ==================== Market Context ====================
    
    async def save_market_context(self, context: dict) -> bool:
        """Save current market context."""
        return await self.set_json("market_context", context, expire=60)
    
    async def get_market_context(self) -> Optional[dict]:
        """Get current market context."""
        return await self.get_json("market_context")
    
    # ==================== Coin Data Cache ====================
    
    async def cache_coin_data(self, symbol: str, data: dict, expire: int = 30) -> bool:
        """Cache coin indicator data."""
        key = f"coin:{symbol}"
        return await self.set_json(key, data, expire=expire)
    
    async def get_coin_data(self, symbol: str) -> Optional[dict]:
        """Get cached coin data."""
        key = f"coin:{symbol}"
        return await self.get_json(key)
    
# ==================== Rate Limiter (Sliding Window) ====================
    
    async def record_alert_sent(self, channel_id: str = "default") -> bool:
        """
        Record an alert was sent for rate limiting.
        Uses sorted set with timestamps for sliding window.
        
        Args:
            channel_id: Telegram channel ID (for multi-channel support)
            
        Returns:
            True if successful
        """
        import time
        key = f"ratelimit:alerts:{channel_id}"
        now = time.time()
        
        try:
            # Add current timestamp to sorted set
            await self._client.zadd(key, {str(now): now})
            
            # Remove entries older than 1 hour (sliding window)
            one_hour_ago = now - 3600
            await self._client.zremrangebyscore(key, "-inf", one_hour_ago)
            
            # Set TTL for auto-cleanup (2 hours)
            await self._client.expire(key, 7200)
            
            logger.debug(f"Rate limit: recorded alert for channel {channel_id}")
            return True
        except Exception as e:
            logger.error(f"Redis record_alert_sent error: {e}")
            return False
    
    async def get_alerts_last_hour(self, channel_id: str = "default") -> int:
        """
        Get number of alerts sent in the last hour.
        
        Args:
            channel_id: Telegram channel ID
            
        Returns:
            Number of alerts in last 60 minutes
        """
        import time
        key = f"ratelimit:alerts:{channel_id}"
        now = time.time()
        one_hour_ago = now - 3600
        
        try:
            # Clean old entries first
            await self._client.zremrangebyscore(key, "-inf", one_hour_ago)
            
            # Count remaining entries
            count = await self._client.zcard(key)
            return count
        except Exception as e:
            logger.error(f"Redis get_alerts_last_hour error: {e}")
            return 0
    
    async def can_send_alert(
        self, 
        score: int, 
        channel_id: str = "default",
        max_alerts: int = 8,      # Reduced from 10 for quality (v2.1)
        tight_threshold: int = 4,  # Reduced from 5
        sniper_threshold: int = 6  # Reduced from 8
    ) -> tuple[bool, str, int]:
        """
        Check if we can send an alert based on rate limit and score.
        
        Rate Limit Modes (v2.1 - stricter for winrate):
        - OPEN (< 4 alerts): Send Gold+ (score >= 60)
        - TIGHT (4-5 alerts): Send Diamond only (score >= 80)
        - SNIPER (6-7 alerts): Send Super only (score >= 90)
        - CLOSED (>= 8 alerts): No alerts
        
        Args:
            score: Signal confidence score (0-100)
            channel_id: Telegram channel ID
            max_alerts: Maximum alerts per hour
            tight_threshold: Alerts count to enter TIGHT mode
            sniper_threshold: Alerts count to enter SNIPER mode
            
        Returns:
            (can_send, mode, alerts_count)
        """
        alerts_count = await self.get_alerts_last_hour(channel_id)
        
        # Determine mode - STRICTER for high winrate (v2.1)
        if alerts_count >= max_alerts:
            mode = "CLOSED"
            can_send = False
        elif alerts_count >= sniper_threshold:
            mode = "SNIPER"
            can_send = score >= 90  # Super Diamond only (was 75)
        elif alerts_count >= tight_threshold:
            mode = "TIGHT"
            can_send = score >= 80  # Diamond only (was 65)
        else:
            mode = "OPEN"
            can_send = score >= 60  # Match new GOLD threshold (was 55)
        
        logger.debug(f"Rate limit: {mode} mode, {alerts_count}/{max_alerts} alerts, score={score}, can_send={can_send}")
        
        return can_send, mode, alerts_count
    
    async def get_rate_limit_status(self, channel_id: str = "default") -> dict:
        """
        Get detailed rate limit status.
        
        Returns:
            Dict with mode, alerts_count, quota info
        """
        alerts_count = await self.get_alerts_last_hour(channel_id)
        
        if alerts_count >= 10:
            mode = "CLOSED"
            remaining = 0
            status_icon = "🔴"
        elif alerts_count >= 8:
            mode = "SNIPER"
            remaining = 10 - alerts_count
            status_icon = "🟠"
        elif alerts_count >= 5:
            mode = "TIGHT"
            remaining = 10 - alerts_count
            status_icon = "🟡"
        else:
            mode = "OPEN"
            remaining = 10 - alerts_count
            status_icon = "🟢"
        
        return {
            "mode": mode,
            "icon": status_icon,
            "alerts_count": alerts_count,
            "remaining": remaining,
            "can_send_diamond": alerts_count < 10,
            "can_send_gold": alerts_count < 5,
            "message": f"{status_icon} {mode}: {alerts_count}/10 alerts/hour"
        }
    
    # ==================== Health Check ====================
    
    async def health_check(self) -> bool:
        """Check if Redis is healthy."""
        try:
            await self._client.ping()
            return True
        except Exception:
            return False


# ==================== Test Functions ====================

async def test_redis_connection():
    """Test Redis connection and basic operations."""
    print("=" * 50)
    print("Testing Redis Connection...")
    print("=" * 50)
    
    client = RedisClient()
    
    try:
        # Test connect
        await client.connect()
        print("✅ Connection successful")
        
        # Test set/get JSON
        test_data = {"price": 100.5, "volume": 1000000, "symbol": "TEST-USDT"}
        await client.set_json("test:coin", test_data, expire=60)
        result = await client.get_json("test:coin")
        
        if result == test_data:
            print("✅ JSON set/get successful")
        else:
            print("❌ JSON mismatch")
        
        # Test cooldown
        await client.set_cooldown("PEPE-USDT", "BREAKOUT_ATTEMPT", 10)
        is_cooling = await client.check_cooldown("PEPE-USDT", "BREAKOUT_ATTEMPT")
        ttl = await client.get_cooldown_ttl("PEPE-USDT", "BREAKOUT_ATTEMPT")
        
        if is_cooling:
            print(f"✅ Cooldown working, TTL: {ttl}s")
        else:
            print("❌ Cooldown not set")
        
        # Test health check
        healthy = await client.health_check()
        if healthy:
            print("✅ Health check passed")
        
        # Cleanup
        await client.delete("test:coin")
        await client.disconnect()
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_redis_connection())
