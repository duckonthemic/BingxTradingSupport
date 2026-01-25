"""
Daily Bias Manager for IE Trade

Manages:
- Daily bias state (LONG/SHORT/NONE)
- 7AM reminder to set bias
- Bias expiry after 24h
- Commands to set/check bias
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
from enum import Enum
import asyncio

from .config import IETradeConfig, DEFAULT_CONFIG


logger = logging.getLogger(__name__)


class DailyBias(Enum):
    """Daily trading bias"""
    NONE = "NONE"      # No bias set - don't scan
    LONG = "LONG"      # Bullish bias - look for longs
    SHORT = "SHORT"    # Bearish bias - look for shorts


@dataclass
class BiasState:
    """Current bias state"""
    bias: DailyBias
    set_at: Optional[datetime]
    set_by: str  # User who set the bias
    expires_at: Optional[datetime]
    
    @property
    def is_active(self) -> bool:
        """Check if bias is active and not expired"""
        if self.bias == DailyBias.NONE:
            return False
        if self.expires_at and datetime.utcnow() > self.expires_at:
            return False
        return True
    
    @property
    def hours_remaining(self) -> float:
        """Hours until bias expires"""
        if not self.expires_at:
            return 0
        remaining = (self.expires_at - datetime.utcnow()).total_seconds() / 3600
        return max(0, remaining)
    
    def __str__(self) -> str:
        if not self.is_active:
            return "No bias set"
        return f"{self.bias.value} (expires in {self.hours_remaining:.1f}h)"


class BiasManager:
    """
    Manages daily trading bias for IE Trade module
    
    The bias determines whether we look for LONG or SHORT setups.
    Must be confirmed by user before scanning starts.
    
    Now with Redis persistence - bias survives bot restart!
    """
    
    # Redis key for bias persistence
    REDIS_KEY = "ie_trade:daily_bias"
    
    def __init__(self, config: IETradeConfig = DEFAULT_CONFIG, redis_client=None):
        self.config = config
        self.redis = redis_client  # Optional Redis for persistence
        self._state = BiasState(
            bias=DailyBias.NONE,
            set_at=None,
            set_by="",
            expires_at=None
        )
        self._reminder_sent_today = False
        self._last_reminder_date: Optional[datetime] = None
    
    async def load_from_redis(self) -> bool:
        """Load bias from Redis on startup. Returns True if loaded."""
        if not self.redis:
            return False
        
        try:
            # Use get_json from our RedisClient wrapper
            bias_data = await self.redis.get_json(self.REDIS_KEY)
            if not bias_data:
                logger.info("🎯 IE Bias: No saved bias found in Redis")
                return False
            
            # Parse expires_at
            expires_at = datetime.fromisoformat(bias_data["expires_at"]) if bias_data.get("expires_at") else None
            
            # Check if expired
            if expires_at and datetime.utcnow() > expires_at:
                logger.info("🎯 IE Bias: Saved bias has expired")
                await self.redis.delete(self.REDIS_KEY)
                return False
            
            # Restore state
            self._state = BiasState(
                bias=DailyBias(bias_data["bias"]),
                set_at=datetime.fromisoformat(bias_data["set_at"]) if bias_data.get("set_at") else None,
                set_by=bias_data.get("set_by", "restored"),
                expires_at=expires_at
            )
            
            remaining = self._state.hours_remaining
            logger.info(f"🎯 IE Bias RESTORED from Redis: {self._state.bias.value}, expires in {remaining:.1f}h")
            return True
            
        except Exception as e:
            logger.error(f"🎯 IE Bias: Failed to load from Redis: {e}")
            return False
    
    async def _save_to_redis(self) -> None:
        """Save current bias to Redis for persistence."""
        if not self.redis:
            return
        
        try:
            if not self._state.is_active:
                # Delete key if no active bias
                await self.redis.delete(self.REDIS_KEY)
                return
            
            bias_data = {
                "bias": self._state.bias.value,
                "set_at": self._state.set_at.isoformat() if self._state.set_at else None,
                "set_by": self._state.set_by,
                "expires_at": self._state.expires_at.isoformat() if self._state.expires_at else None
            }
            
            # Set with TTL = bias expiry time + 1 hour buffer
            ttl = int(self._state.hours_remaining * 3600) + 3600
            await self.redis.set_json(self.REDIS_KEY, bias_data, expire=ttl)
            logger.debug(f"🎯 IE Bias saved to Redis, TTL={ttl}s")
            
        except Exception as e:
            logger.error(f"🎯 IE Bias: Failed to save to Redis: {e}")
    
    @property
    def current_bias(self) -> DailyBias:
        """Get current active bias"""
        if self._state.is_active:
            return self._state.bias
        return DailyBias.NONE
    
    @property
    def is_bias_set(self) -> bool:
        """Check if a valid bias is set"""
        return self._state.is_active
    
    @property
    def state(self) -> BiasState:
        """Get full bias state"""
        return self._state
    
    def set_bias(self, bias: DailyBias, set_by: str = "user") -> BiasState:
        """
        Set daily bias
        
        Args:
            bias: LONG or SHORT
            set_by: Username who set the bias
            
        Returns:
            Updated bias state
        """
        now = datetime.utcnow()
        expires_at = now + timedelta(hours=self.config.BIAS_EXPIRY_HOURS)
        
        self._state = BiasState(
            bias=bias,
            set_at=now,
            set_by=set_by,
            expires_at=expires_at
        )
        
        logger.info(f"🎯 IE Bias set to {bias.value} by {set_by}, expires at {expires_at}")
        
        # Save to Redis for persistence (async in background)
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._save_to_redis())
        except RuntimeError:
            # No running loop - will save on next async operation
            pass
        
        return self._state
    
    def clear_bias(self) -> None:
        """Clear current bias"""
        self._state = BiasState(
            bias=DailyBias.NONE,
            set_at=None,
            set_by="",
            expires_at=None
        )
        logger.info("🎯 IE Bias cleared")
    
    def parse_bias_command(self, command: str) -> Optional[DailyBias]:
        """
        Parse bias command from user input
        
        Supported formats:
        - /dbias B or /dbias b -> LONG
        - /dbias S or /dbias s -> SHORT
        - /dbias LONG or /dbias long -> LONG
        - /dbias SHORT or /dbias short -> SHORT
        
        Returns:
            DailyBias or None if invalid
        """
        parts = command.strip().lower().split()
        
        if len(parts) < 2:
            return None
        
        bias_str = parts[1].upper()
        
        if bias_str in ('B', 'BUY', 'LONG', 'L'):
            return DailyBias.LONG
        elif bias_str in ('S', 'SELL', 'SHORT'):
            return DailyBias.SHORT
        
        return None
    
    def should_send_reminder(self, current_hour: int) -> bool:
        """
        Check if we should send a reminder to set bias
        
        Reminder is sent at 7AM Vietnam time if:
        - No reminder sent today
        - No active bias set
        """
        # Check if it's reminder hour (7AM VN)
        if current_hour != self.config.BIAS_REMINDER_HOUR:
            return False
        
        # Check if already sent today
        today = datetime.utcnow().date()
        if self._last_reminder_date == today:
            return False
        
        # Check if bias already set
        if self.is_bias_set:
            return False
        
        return True
    
    def mark_reminder_sent(self) -> None:
        """Mark that reminder was sent today"""
        self._last_reminder_date = datetime.utcnow().date()
        self._reminder_sent_today = True
        logger.info("🎯 IE Bias reminder sent")
    
    def reset_daily(self) -> None:
        """Reset for new day (called at midnight or 7AM)"""
        self._reminder_sent_today = False
        # Optionally clear bias at new day
        # self.clear_bias()
    
    def get_reminder_message(self) -> str:
        """Get the reminder message to send"""
        return """
🌅 **Good Morning! IE Trade Reminder**

Đã đến lúc xác nhận Daily Bias cho hôm nay!

📊 **Hãy phân tích khung Daily và xác định xu hướng:**

• Cấu trúc đang **GIẢM** (Lower Highs/Lows)?
  → Dùng lệnh: `/dbias S`

• Cấu trúc đang **TĂNG** (Higher Highs/Lows)?
  → Dùng lệnh: `/dbias B`

⚠️ **Nếu không xác nhận bias, IE Trade sẽ không quét signal!**

---
_Bias sẽ tự động hết hạn sau 24h_
"""
    
    def get_status_message(self) -> str:
        """Get current status message"""
        if not self.is_bias_set:
            return """
🎯 **IE Trade Status**

📊 Daily Bias: ❌ **Chưa set**
⏰ Trạng thái: **Không quét**

💡 Dùng `/dbias B` (Long) hoặc `/dbias S` (Short) để bắt đầu.
"""
        
        return f"""
🎯 **IE Trade Status**

📊 Daily Bias: {self._state.bias.value}
⏰ Set at: {self._state.set_at.strftime('%H:%M %d/%m')} UTC
⏱️ Expires in: {self._state.hours_remaining:.1f} hours
👤 Set by: {self._state.set_by}

✅ **Đang quét signal theo bias {self._state.bias.value}**
"""
    
    def get_bias_confirmed_message(self, bias: DailyBias) -> str:
        """Get confirmation message after setting bias"""
        direction = "🟢 LONG (Bullish)" if bias == DailyBias.LONG else "🔴 SHORT (Bearish)"
        
        return f"""
✅ **Daily Bias Confirmed!**

📊 Bias: {direction}
⏰ Hết hạn sau: {self.config.BIAS_EXPIRY_HOURS}h
🔍 Trạng thái: **Đang quét signal**

**IE Trade sẽ:**
• Tìm H1 FVG trong vùng {'Premium' if bias == DailyBias.SHORT else 'Discount'}
• Chờ M5 MSS xác nhận
• Alert khi vào Kill Zone (London/NY)

_Dùng `/iestatus` để xem trạng thái_
"""


class BiasScheduler:
    """
    Scheduler for bias reminders and reset
    
    Runs in background to:
    - Send 7AM reminder if no bias set
    - Reset reminder flag at midnight
    """
    
    def __init__(
        self, 
        bias_manager: BiasManager,
        send_callback,  # async function to send Telegram message
        config: IETradeConfig = DEFAULT_CONFIG
    ):
        self.bias_manager = bias_manager
        self.send_callback = send_callback
        self.config = config
        self._running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self) -> None:
        """Start the scheduler"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("🎯 IE Bias scheduler started")
    
    async def stop(self) -> None:
        """Stop the scheduler"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("🎯 IE Bias scheduler stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            try:
                # Get current hour (Vietnam time = UTC+7)
                utc_now = datetime.utcnow()
                vn_hour = (utc_now.hour + 7) % 24
                
                # Check if reminder needed
                if self.bias_manager.should_send_reminder(vn_hour):
                    message = self.bias_manager.get_reminder_message()
                    await self.send_callback(message)
                    self.bias_manager.mark_reminder_sent()
                
                # Sleep for 5 minutes before next check
                await asyncio.sleep(300)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in bias scheduler: {e}")
                await asyncio.sleep(60)
