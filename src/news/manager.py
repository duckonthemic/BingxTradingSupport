"""
News Manager - Main orchestrator for all news-related functionality.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, List

import pytz

from .models import NewsEvent, PollingMode, NewsWorkflowState
from .client import NewsClient
from .filter import NewsFilter
from .scheduler import SessionScheduler, WeeklyReporter
from .poller import AdaptivePoller
from .trader import NewsTrader

logger = logging.getLogger(__name__)


class NewsManager:
    """
    Main orchestrator for all news-related functionality.
    
    Combines:
    - NewsClient: Fetch economic calendar
    - NewsFilter: Filter important events
    - SessionScheduler: Session open alerts
    - WeeklyReporter: Weekly calendar reports
    - AdaptivePoller: Dynamic polling
    - NewsTrader: Trading workflow around news
    """
    
    def __init__(self):
        # Components
        self.client = NewsClient()
        self.filter = NewsFilter()
        self.scheduler = SessionScheduler()
        self.reporter = WeeklyReporter()
        self.poller = AdaptivePoller()
        self.trader = NewsTrader()
        
        # State
        self.running = False
        self._events: List[NewsEvent] = []
        self._last_fetch: Optional[datetime] = None
        
        # Callbacks (set by AlertManager)
        self._send_message: Optional[callable] = None
        self._send_pinned_message: Optional[callable] = None
        self._pause_trading: Optional[callable] = None
        self._resume_trading: Optional[callable] = None
    
    def configure(
        self,
        send_message: callable,
        send_pinned_message: callable,
        pause_trading: callable,
        resume_trading: callable
    ):
        """
        Configure callbacks for the news manager.
        
        Args:
            send_message: Async function to send a Telegram message
            send_pinned_message: Async function to send and pin a message
            pause_trading: Async function to pause the trading bot
            resume_trading: Async function to resume the trading bot
        """
        self._send_message = send_message
        self._send_pinned_message = send_pinned_message
        self._pause_trading = pause_trading
        self._resume_trading = resume_trading
        
        # Configure session scheduler
        self.scheduler.set_alert_callback(send_message)
        
        # Configure weekly reporter
        self.reporter.set_callbacks(
            report_callback=self._send_weekly_report,
            get_events_callback=self._get_filtered_events
        )
        
        # Configure adaptive poller
        self.poller.set_callbacks(
            poll_callback=self._poll_events,
            mode_change_callback=self._on_mode_change
        )
        
        # Configure news trader
        self.trader.set_callbacks(
            alert_callback=send_message,
            pause_callback=pause_trading,
            resume_callback=resume_trading,
            straddle_callback=self._send_straddle_alert
        )
        
        logger.info("📰 News manager configured")
    
    async def start(self):
        """Start all news manager components."""
        if self.running:
            return
        
        self.running = True
        
        # Initial fetch
        await self._poll_events()
        
        # Start components
        await self.scheduler.start()
        await self.reporter.start()
        await self.poller.start()
        await self.trader.start()
        
        logger.info("📰 News manager started")
    
    async def stop(self):
        """Stop all news manager components."""
        self.running = False
        
        # Stop in reverse order
        await self.trader.stop()
        await self.poller.stop()
        await self.reporter.stop()
        await self.scheduler.stop()
        
        logger.info("📰 News manager stopped")
    
    async def _poll_events(self) -> List[NewsEvent]:
        """Fetch and filter events."""
        try:
            # Fetch from client
            all_events = await self.client.fetch_weekly_calendar()
            
            # Filter important events
            self._events = self.filter.filter_events(all_events)
            self._last_fetch = datetime.now(pytz.utc)
            
            # Update trader with events
            await self.trader.process_events(self._events)
            
            logger.debug(f"📰 Polled {len(self._events)} important events")
            return self._events
        except Exception as e:
            logger.error(f"Failed to poll events: {e}")
            return self._events  # Return cached events
    
    async def _get_filtered_events(self) -> List[NewsEvent]:
        """Get filtered events (for weekly reporter)."""
        if not self._events or self._should_refresh():
            await self._poll_events()
        return self._events
    
    def _should_refresh(self) -> bool:
        """Check if events cache should be refreshed."""
        if not self._last_fetch:
            return True
        
        # Refresh if older than 5 minutes
        age = datetime.now(pytz.utc) - self._last_fetch
        return age > timedelta(minutes=5)
    
    async def _send_weekly_report(self, message: str, pin: bool = False):
        """Send weekly report (optionally pinned)."""
        if pin and self._send_pinned_message:
            await self._send_pinned_message(message)
        elif self._send_message:
            await self._send_message(message)
    
    async def _send_straddle_alert(self, setup):
        """Send straddle setup alert."""
        if not self._send_message:
            return
        
        message = f"""
📈 <b>STRADDLE SETUP - NEWS TRAP</b>

🔥 Event: <b>{setup.event.title}</b>
📊 Symbol: {setup.symbol}

<b>Strategy:</b>
• LONG Entry: +{setup.entry_buffer_pct}% from current price
• SHORT Entry: -{setup.entry_buffer_pct}% from current price
• TP: {setup.tp_pct}%
• SL: {setup.sl_pct}%

⚡ First trigger cancels the other!
⏰ News in: 3 MINUTES

━━━━━━━━━━━━━━━━━━━━━━
"""
        await self._send_message(message)
    
    async def _on_mode_change(self, old_mode: PollingMode, new_mode: PollingMode):
        """Handle polling mode changes."""
        # Update client cache TTL based on mode
        ttl_map = {
            PollingMode.IDLE: 300,
            PollingMode.STANDBY: 60,
            PollingMode.BATTLE: 5
        }
        self.client.set_cache_ttl(ttl_map.get(new_mode, 300))
        
        # Log mode change
        logger.info(f"📰 Polling mode: {old_mode.value} → {new_mode.value}")
    
    # ==================== Public API ====================
    
    async def get_upcoming_events(self, hours: int = 24) -> List[NewsEvent]:
        """Get important events in the next N hours."""
        if self._should_refresh():
            await self._poll_events()
        
        now = datetime.now(pytz.utc)
        cutoff = now + timedelta(hours=hours)
        
        def get_aware_date(event):
            """Ensure event date is timezone-aware."""
            if event.date.tzinfo is None:
                return pytz.utc.localize(event.date)
            return event.date.astimezone(pytz.utc)
        
        return [
            e for e in self._events
            if now <= get_aware_date(e) <= cutoff
        ]
    
    async def get_weekly_report(self) -> str:
        """Get the weekly report text."""
        return await self.reporter.send_report_now()
    
    async def send_weekly_report_now(self):
        """Manually send the weekly report."""
        report = await self.get_weekly_report()
        if self._send_pinned_message:
            await self._send_pinned_message(report)
        elif self._send_message:
            await self._send_message(report)
    
    def get_status(self) -> dict:
        """Get comprehensive status of the news manager."""
        return {
            "running": self.running,
            "events_count": len(self._events),
            "last_fetch": self._last_fetch.isoformat() if self._last_fetch else None,
            "poller": self.poller.get_status(),
            "trader": self.trader.get_status(),
            "current_session": self.scheduler.get_current_session().value if self.scheduler.get_current_session() else None
        }
    
    @property
    def is_trading_paused(self) -> bool:
        """Check if trading is currently paused due to news."""
        return self.trader.is_trading_paused
    
    def force_resume(self):
        """Force resume trading (emergency override)."""
        self.trader.force_resume()
    
    async def check_trading_allowed(self) -> tuple:
        """
        Check if trading is allowed right now.
        
        Returns:
            Tuple of (allowed: bool, reason: str)
        """
        if self.trader.is_trading_paused:
            event = self.trader._paused_for
            reason = f"Paused for: {event.title}" if event else "Paused for news"
            return (False, reason)
        
        return (True, "Trading allowed")


# Singleton instance
_news_manager: Optional[NewsManager] = None


def get_news_manager() -> NewsManager:
    """Get or create the singleton NewsManager instance."""
    global _news_manager
    if _news_manager is None:
        _news_manager = NewsManager()
    return _news_manager
