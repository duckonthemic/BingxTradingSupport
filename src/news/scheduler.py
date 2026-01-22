"""
Session Scheduler - Handle trading session alerts and weekly reports.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, List

import pytz

from .models import SessionType, SessionInfo, TRADING_SESSIONS, NewsEvent

logger = logging.getLogger(__name__)


class SessionScheduler:
    """
    Scheduler for trading session open alerts.
    
    Sends alerts when major trading sessions open:
    - Asia Session: 00:00 UTC (Tokyo open)
    - London Session: 07:00/08:00 UTC (depends on DST)
    - US Session: 13:30/14:30 UTC (depends on DST)
    - Market Close: 20:00/21:00 UTC
    """
    
    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._alert_callback: Optional[Callable] = None
        self._last_alerts: dict = {}  # Track which sessions were alerted today
    
    def set_alert_callback(self, callback: Callable):
        """Set callback for sending session alerts."""
        self._alert_callback = callback
    
    async def start(self):
        """Start the session scheduler."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("📅 Session scheduler started")
    
    async def stop(self):
        """Stop the session scheduler."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("📅 Session scheduler stopped")
    
    async def _scheduler_loop(self):
        """Main scheduler loop - check every minute for session opens."""
        while self.running:
            try:
                await self._check_sessions()
                # Sleep until next minute
                now = datetime.now(pytz.utc)
                seconds_to_next_minute = 60 - now.second
                await asyncio.sleep(seconds_to_next_minute)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session scheduler error: {e}")
                await asyncio.sleep(60)
    
    async def _check_sessions(self):
        """Check if any session is opening now."""
        now = datetime.now(pytz.utc)
        current_hour = now.hour
        current_minute = now.minute
        today = now.strftime("%Y-%m-%d")
        
        # Only alert at the top of the hour (within first minute)
        if current_minute > 1:
            return
        
        for session_type, session_info in TRADING_SESSIONS.items():
            # Check both winter and summer hours
            session_hours = [session_info.hour_utc_winter, session_info.hour_utc]
            
            for hour in session_hours:
                if current_hour == hour:
                    # Check if already alerted today for this session
                    alert_key = f"{today}_{session_type.value}"
                    if alert_key in self._last_alerts:
                        continue
                    
                    # Mark as alerted
                    self._last_alerts[alert_key] = now
                    
                    # Clean old alerts (keep only today)
                    self._clean_old_alerts(today)
                    
                    # Send alert
                    await self._send_session_alert(session_type, session_info)
                    break
    
    def _clean_old_alerts(self, today: str):
        """Remove alerts from previous days."""
        to_remove = [k for k in self._last_alerts if not k.startswith(today)]
        for k in to_remove:
            del self._last_alerts[k]
    
    async def _send_session_alert(self, session_type: SessionType, session_info: SessionInfo):
        """Send a session open alert."""
        if not self._alert_callback:
            logger.warning("No alert callback set for session scheduler")
            return
        
        message = self._format_session_alert(session_type, session_info)
        
        try:
            await self._alert_callback(message)
            logger.info(f"📢 Sent session alert: {session_type.value}")
        except Exception as e:
            logger.error(f"Failed to send session alert: {e}")
    
    def _format_session_alert(self, session_type: SessionType, session_info: SessionInfo) -> str:
        """Format a session open alert message."""
        now = datetime.now(pytz.utc)
        
        return f"""
🌍 <b>SESSION OPEN: {session_info.display_name}</b>

{session_info.emoji} <b>{session_type.value.upper()} SESSION</b>

⏰ Time: {now.strftime("%H:%M")} UTC
📅 Date: {now.strftime("%A, %B %d, %Y")}

{self._get_session_notes(session_type)}

━━━━━━━━━━━━━━━━━━━━━━
"""
    
    def _get_session_notes(self, session_type: SessionType) -> str:
        """Get trading notes for each session."""
        notes = {
            SessionType.ASIA: (
                "📌 <b>Asia Session Notes:</b>\n"
                "• Usually lower volatility\n"
                "• JPY pairs most active\n"
                "• Watch for Asian news (BOJ, China data)"
            ),
            SessionType.LONDON: (
                "📌 <b>London Session Notes:</b>\n"
                "• Highest volatility period\n"
                "• EUR/GBP pairs most active\n"
                "• Often sets daily range"
            ),
            SessionType.US: (
                "📌 <b>US Session Notes:</b>\n"
                "• High volatility first 2 hours\n"
                "• USD pairs most active\n"
                "• Watch for US economic data"
            ),
            SessionType.CLOSE: (
                "📌 <b>Market Close Notes:</b>\n"
                "• Spreads may widen\n"
                "• Less liquidity\n"
                "• Consider closing positions"
            )
        }
        return notes.get(session_type, "")
    
    def is_session_active(self, session_type: SessionType) -> bool:
        """Check if a session is currently active."""
        now = datetime.now(pytz.utc)
        hour = now.hour
        
        # Simplified session hours (using summer time as baseline)
        session_ranges = {
            SessionType.ASIA: (0, 9),      # 00:00 - 09:00 UTC
            SessionType.LONDON: (7, 16),   # 07:00 - 16:00 UTC
            SessionType.US: (13, 21),      # 13:30 - 21:00 UTC
            SessionType.CLOSE: (20, 23)    # 20:00 - 23:00 UTC
        }
        
        if session_type in session_ranges:
            start, end = session_ranges[session_type]
            return start <= hour <= end
        
        return False
    
    def get_current_session(self) -> Optional[SessionType]:
        """Get the currently most active session."""
        now = datetime.now(pytz.utc)
        hour = now.hour
        
        # Overlapping sessions - return the most significant
        if 13 <= hour < 16:
            return SessionType.US  # London/US overlap
        elif 7 <= hour < 13:
            return SessionType.LONDON
        elif 13 <= hour < 21:
            return SessionType.US
        elif 0 <= hour < 9:
            return SessionType.ASIA
        else:
            return SessionType.CLOSE


class WeeklyReporter:
    """
    Generate and send weekly economic calendar reports.
    Sent every Sunday at 20:00 UTC.
    """
    
    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        self._report_callback: Optional[Callable] = None
        self._get_events_callback: Optional[Callable] = None
    
    def set_callbacks(self, report_callback: Callable, get_events_callback: Callable):
        """
        Set callbacks for the reporter.
        
        Args:
            report_callback: Function to send the report message
            get_events_callback: Function to get weekly events
        """
        self._report_callback = report_callback
        self._get_events_callback = get_events_callback
    
    async def start(self):
        """Start the weekly reporter."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._reporter_loop())
        logger.info("📊 Weekly reporter started")
    
    async def stop(self):
        """Stop the weekly reporter."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("📊 Weekly reporter stopped")
    
    async def _reporter_loop(self):
        """Check every hour if it's Sunday 20:00 UTC."""
        while self.running:
            try:
                now = datetime.now(pytz.utc)
                
                # Sunday = 6 (weekday), at 20:00 UTC
                if now.weekday() == 6 and now.hour == 20 and now.minute < 5:
                    await self._send_weekly_report()
                    # Wait until next hour to avoid duplicate
                    await asyncio.sleep(3600)
                else:
                    # Check every 5 minutes
                    await asyncio.sleep(300)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Weekly reporter error: {e}")
                await asyncio.sleep(300)
    
    async def _send_weekly_report(self):
        """Generate and send the weekly report."""
        if not self._report_callback or not self._get_events_callback:
            logger.warning("Weekly reporter callbacks not set")
            return
        
        try:
            # Get events for the week
            events = await self._get_events_callback()
            
            # Generate report
            report = self._format_weekly_report(events)
            
            # Send report
            await self._report_callback(report, pin=True)
            logger.info("📊 Weekly report sent and pinned")
        except Exception as e:
            logger.error(f"Failed to send weekly report: {e}")
    
    async def send_report_now(self) -> str:
        """Manually trigger a weekly report. Returns the report text."""
        if not self._get_events_callback:
            return "❌ Events callback not configured"
        
        try:
            events = await self._get_events_callback()
            return self._format_weekly_report(events)
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            return f"❌ Error generating report: {e}"
    
    def _format_weekly_report(self, events: List[NewsEvent]) -> str:
        """Format the weekly economic calendar report."""
        now = datetime.now(pytz.utc)
        week_start = now - timedelta(days=now.weekday())
        week_end = week_start + timedelta(days=6)
        
        # Group events by day
        by_day = {}
        for event in events:
            day_name = event.date.strftime("%A")
            if day_name not in by_day:
                by_day[day_name] = []
            by_day[day_name].append(event)
        
        # Build report
        lines = [
            f"📅 <b>WEEKLY ECONOMIC CALENDAR</b>",
            f"📆 {week_start.strftime('%B %d')} - {week_end.strftime('%B %d, %Y')}",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━",
            ""
        ]
        
        # Days of the week in order
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        high_impact_count = 0
        
        for day in day_order:
            day_events = by_day.get(day, [])
            
            if not day_events:
                continue
            
            lines.append(f"📍 <b>{day.upper()}</b>")
            lines.append("")
            
            for event in sorted(day_events, key=lambda e: e.date):
                impact_emoji = self._get_impact_emoji(event.impact)
                time_str = event.date.strftime("%H:%M")
                
                if event.impact.value == "high":
                    high_impact_count += 1
                    lines.append(f"  {impact_emoji} <b>{time_str} - {event.title}</b>")
                else:
                    lines.append(f"  {impact_emoji} {time_str} - {event.title}")
            
            lines.append("")
        
        # Summary
        lines.extend([
            "━━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"📊 <b>Summary:</b>",
            f"• Total tracked events: {len(events)}",
            f"• High impact events: {high_impact_count}",
            "",
            "⚠️ Bot will PAUSE trading 5 mins before high-impact news",
            "📈 Straddle alerts will be sent 3 mins before",
            "",
            "━━━━━━━━━━━━━━━━━━━━━━"
        ])
        
        return "\n".join(lines)
    
    def _get_impact_emoji(self, impact) -> str:
        """Get emoji for impact level."""
        from .models import NewsImpact
        return {
            NewsImpact.HIGH: "🔥",
            NewsImpact.MEDIUM: "⚡",
            NewsImpact.LOW: "📊",
            NewsImpact.HOLIDAY: "🏖️"
        }.get(impact, "📰")
