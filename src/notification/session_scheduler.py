"""
Trading Session Scheduler.
Sends automatic notifications when trading sessions open.

Sessions (UTC):
- Asia: 00:00 UTC (Tokyo, Hong Kong, Singapore open)
- Europe: 07:00 UTC (London open)  
- US: 13:30 UTC (New York open)
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable
import pytz

logger = logging.getLogger(__name__)


class TradingSession:
    """Trading session definition."""
    
    def __init__(self, name: str, emoji: str, utc_hour: int, utc_minute: int = 0,
                 description: str = "", tips: list = None):
        self.name = name
        self.emoji = emoji
        self.utc_hour = utc_hour
        self.utc_minute = utc_minute
        self.description = description
        self.tips = tips or []
    
    def get_next_open(self, now: datetime = None) -> datetime:
        """Get next session open time."""
        if now is None:
            now = datetime.now(pytz.utc)
        
        # Make sure now is UTC
        if now.tzinfo is None:
            now = pytz.utc.localize(now)
        else:
            now = now.astimezone(pytz.utc)
        
        # Today's session time
        today_session = now.replace(
            hour=self.utc_hour, 
            minute=self.utc_minute, 
            second=0, 
            microsecond=0
        )
        
        # If session already passed today, get tomorrow's
        if today_session <= now:
            today_session += timedelta(days=1)
        
        return today_session


# Session definitions
TRADING_SESSIONS = [
    TradingSession(
        name="ASIA",
        emoji="🌏",
        utc_hour=0,
        utc_minute=0,
        description="Tokyo, Hong Kong, Singapore mở cửa",
        tips=[
            "• Volume thường thấp hơn EU/US",
            "• Altcoins châu Á có thể pump (SUI, APT, TIA)",
            "• BTC thường sideway chờ EU",
        ]
    ),
    TradingSession(
        name="EUROPE", 
        emoji="🌍",
        utc_hour=7,
        utc_minute=0,
        description="London mở cửa",
        tips=[
            "• Volume tăng mạnh, breakout thường xảy ra",
            "• Cẩn thận với fake breakout đầu phiên",
            "• BTC/ETH thường có move lớn",
        ]
    ),
    TradingSession(
        name="US",
        emoji="🌎",
        utc_hour=13,
        utc_minute=30,
        description="New York mở cửa",
        tips=[
            "• Volume CAO NHẤT trong ngày",
            "• Tin kinh tế Mỹ thường ra lúc này",
            "• BTC correlation với S&P500 cao",
            "• Major moves thường xảy ra 14:30-16:00 UTC",
        ]
    ),
]


class SessionScheduler:
    """
    Schedules and sends trading session notifications.
    """
    
    def __init__(self, send_message_fn: Callable[[str], Awaitable[bool]] = None):
        """
        Initialize scheduler.
        
        Args:
            send_message_fn: Async function to send Telegram messages
        """
        self.send_message = send_message_fn
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sent: dict = {}  # Track last sent times
        
    async def start(self):
        """Start the scheduler."""
        if self._running:
            return
            
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("⏰ Session Scheduler started")
        
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⏰ Session Scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop."""
        while self._running:
            try:
                now = datetime.now(pytz.utc)
                
                for session in TRADING_SESSIONS:
                    # Check if it's time to send (within 1 minute of session open)
                    next_open = session.get_next_open(now)
                    time_until = (next_open - now).total_seconds()
                    
                    # If session opens in less than 30 seconds
                    if 0 <= time_until <= 30:
                        # Check if we already sent for this session today
                        session_key = f"{session.name}_{next_open.strftime('%Y%m%d')}"
                        if session_key not in self._last_sent:
                            await self._send_session_notification(session, now)
                            self._last_sent[session_key] = now
                            
                            # Clean old entries (keep only last 10)
                            if len(self._last_sent) > 10:
                                oldest = min(self._last_sent.keys())
                                del self._last_sent[oldest]
                
                # Sleep for 10 seconds before checking again
                await asyncio.sleep(10)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session scheduler error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _send_session_notification(self, session: TradingSession, now: datetime):
        """Send session open notification."""
        if not self.send_message:
            logger.warning("No send_message function configured")
            return
        
        # Format tips
        tips_text = "\n".join(session.tips) if session.tips else ""
        
        # Vietnam timezone for local time display
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        vn_time = now.astimezone(vn_tz).strftime('%H:%M')
        
        message = f"""
{session.emoji} <b>PHIÊN {session.name} MỞ CỬA</b>
━━━━━━━━━━━━━━━━━━━━━━━━

🕐 <b>Giờ hiện tại:</b> {vn_time} (VN)
📍 <b>{session.description}</b>

<b>💡 Lưu ý:</b>
{tips_text}

━━━━━━━━━━━━━━━━━━━━━━━━
🤖 <i>Auto Session Alert</i>
"""
        
        try:
            success = await self.send_message(message.strip())
            if success:
                logger.info(f"✅ Sent {session.name} session notification")
            else:
                logger.warning(f"❌ Failed to send {session.name} session notification")
        except Exception as e:
            logger.error(f"Error sending session notification: {e}")
    
    async def send_manual_session_status(self) -> str:
        """
        Generate current session status message.
        Returns formatted message string.
        """
        now = datetime.now(pytz.utc)
        vn_tz = pytz.timezone('Asia/Ho_Chi_Minh')
        vn_time = now.astimezone(vn_tz).strftime('%H:%M')
        
        lines = [
            "🌐 <b>TRADING SESSIONS STATUS</b>",
            "━━━━━━━━━━━━━━━━━━━━━━━━",
            f"🕐 Giờ hiện tại: {vn_time} (VN) | {now.strftime('%H:%M')} (UTC)",
            "",
        ]
        
        # Determine current session
        current_hour = now.hour
        if 0 <= current_hour < 7:
            current_session = "ASIA"
        elif 7 <= current_hour < 13:
            current_session = "EUROPE"
        elif 13 <= current_hour < 21:
            current_session = "US"
        else:
            current_session = "ASIA (Night)"
        
        lines.append(f"📍 <b>Phiên hiện tại:</b> {current_session}")
        lines.append("")
        lines.append("<b>⏰ Lịch mở cửa (UTC):</b>")
        
        for session in TRADING_SESSIONS:
            next_open = session.get_next_open(now)
            time_until = next_open - now
            hours = int(time_until.total_seconds() // 3600)
            mins = int((time_until.total_seconds() % 3600) // 60)
            
            is_active = session.name in current_session
            status = "🟢 ACTIVE" if is_active else f"⏳ {hours}h {mins}m"
            
            lines.append(f"  {session.emoji} {session.name}: {session.utc_hour:02d}:{session.utc_minute:02d} - {status}")
        
        lines.append("")
        lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
        
        return "\n".join(lines)
