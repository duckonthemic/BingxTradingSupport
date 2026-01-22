"""
News Trader - Workflow for trading around news events.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, List, Dict

import pytz

from .models import (
    NewsEvent, 
    NewsWorkflowState,
    StraddleSetup,
    NewsImpact
)

logger = logging.getLogger(__name__)


class NewsTrader:
    """
    Manages trading workflow around high-impact news events.
    
    Workflow Timeline:
    - T-30min: PRE_ALERT - Send upcoming news notification
    - T-5min: DEFENSE - PAUSE bot trading
    - T-3min: STRADDLE - Send straddle setup alert
    - T+0: MONITORING - Wait for news result
    - T+15min: RESUME - Resume bot trading
    """
    
    # Workflow timing thresholds (in minutes)
    THRESHOLDS = {
        NewsWorkflowState.PRE_ALERT: 30,   # 30 mins before
        NewsWorkflowState.DEFENSE: 5,       # 5 mins before - PAUSE
        NewsWorkflowState.STRADDLE: 3,      # 3 mins before - Send straddle
        NewsWorkflowState.MONITORING: 0,    # At news time
        NewsWorkflowState.COOLDOWN: -15     # 15 mins after - RESUME
    }
    
    def __init__(self):
        self.running = False
        self._task: Optional[asyncio.Task] = None
        
        # Callbacks
        self._alert_callback: Optional[Callable] = None      # Send telegram message
        self._pause_callback: Optional[Callable] = None      # Pause trading
        self._resume_callback: Optional[Callable] = None     # Resume trading
        self._straddle_callback: Optional[Callable] = None   # Send straddle alert
        
        # State tracking
        self._active_events: Dict[str, NewsWorkflowState] = {}  # event_id -> state
        self._paused_for: Optional[NewsEvent] = None
        self.is_trading_paused = False
    
    def set_callbacks(
        self,
        alert_callback: Callable,
        pause_callback: Callable,
        resume_callback: Callable,
        straddle_callback: Optional[Callable] = None
    ):
        """Set workflow callbacks."""
        self._alert_callback = alert_callback
        self._pause_callback = pause_callback
        self._resume_callback = resume_callback
        self._straddle_callback = straddle_callback
    
    async def start(self):
        """Start the news trader workflow."""
        if self.running:
            return
        
        self.running = True
        logger.info("📈 News trader workflow started")
    
    async def stop(self):
        """Stop the news trader workflow."""
        self.running = False
        # Resume trading if paused
        if self.is_trading_paused:
            await self._do_resume("Shutdown")
        logger.info("📈 News trader workflow stopped")
    
    async def process_events(self, events: List[NewsEvent]):
        """
        Process events and trigger workflow actions.
        
        Called by the poller when new events are fetched.
        """
        now = datetime.now(pytz.utc)
        
        for event in events:
            # Only process high impact USD events
            if not event.is_high_impact_usd:
                continue
            
            # Calculate time until event
            time_until = event.time_until
            if time_until is None:
                continue
            
            minutes_until = time_until.total_seconds() / 60
            event_id = f"{event.date.isoformat()}_{event.title[:20]}"
            
            # Get current workflow state for this event
            current_state = self._active_events.get(event_id, NewsWorkflowState.IDLE)
            
            # Determine new state based on time
            new_state = self._determine_state(minutes_until)
            
            # Trigger actions on state transitions
            if new_state != current_state:
                await self._handle_state_transition(event, event_id, current_state, new_state)
                self._active_events[event_id] = new_state
        
        # Clean up old events
        self._cleanup_old_events()
    
    def _determine_state(self, minutes_until: float) -> NewsWorkflowState:
        """Determine workflow state based on time until event."""
        if minutes_until > self.THRESHOLDS[NewsWorkflowState.PRE_ALERT]:
            return NewsWorkflowState.IDLE
        elif minutes_until > self.THRESHOLDS[NewsWorkflowState.DEFENSE]:
            return NewsWorkflowState.PRE_ALERT
        elif minutes_until > self.THRESHOLDS[NewsWorkflowState.STRADDLE]:
            return NewsWorkflowState.DEFENSE
        elif minutes_until > self.THRESHOLDS[NewsWorkflowState.MONITORING]:
            return NewsWorkflowState.STRADDLE
        elif minutes_until > self.THRESHOLDS[NewsWorkflowState.COOLDOWN]:
            return NewsWorkflowState.MONITORING
        else:
            return NewsWorkflowState.COOLDOWN
    
    async def _handle_state_transition(
        self, 
        event: NewsEvent, 
        event_id: str,
        old_state: NewsWorkflowState, 
        new_state: NewsWorkflowState
    ):
        """Handle workflow state transitions."""
        logger.info(f"📈 {event.title}: {old_state.value} → {new_state.value}")
        
        if new_state == NewsWorkflowState.PRE_ALERT:
            await self._send_pre_alert(event)
        
        elif new_state == NewsWorkflowState.DEFENSE:
            await self._do_pause(event)
        
        elif new_state == NewsWorkflowState.STRADDLE:
            await self._send_straddle(event)
        
        elif new_state == NewsWorkflowState.MONITORING:
            await self._start_monitoring(event)
        
        elif new_state == NewsWorkflowState.COOLDOWN:
            await self._do_resume(event.title)
    
    async def _send_pre_alert(self, event: NewsEvent):
        """Send T-30 pre-alert notification."""
        if not self._alert_callback:
            return
        
        message = f"""
⚠️ <b>NEWS ALERT - 30 MINUTES</b>

🔥 <b>{event.title}</b>
🕐 Time: {event.date.strftime("%H:%M")} UTC
💰 Currency: {event.currency}
📊 Impact: HIGH

⏰ Bot will PAUSE trading at T-5 min
📈 Straddle alert at T-3 min

━━━━━━━━━━━━━━━━━━━━━━
"""
        try:
            await self._alert_callback(message)
        except Exception as e:
            logger.error(f"Failed to send pre-alert: {e}")
    
    async def _do_pause(self, event: NewsEvent):
        """Pause trading at T-5."""
        if self.is_trading_paused:
            logger.info(f"Already paused, skipping pause for {event.title}")
            return
        
        self.is_trading_paused = True
        self._paused_for = event
        
        # Call pause callback
        if self._pause_callback:
            try:
                await self._pause_callback()
            except Exception as e:
                logger.error(f"Pause callback error: {e}")
        
        # Send pause notification
        if self._alert_callback:
            message = f"""
🛑 <b>TRADING PAUSED</b>

🔥 <b>{event.title}</b>
🕐 Releasing at: {(event.date + timedelta(minutes=15)).strftime("%H:%M")} UTC

⚠️ No new positions will be opened
📊 Existing positions remain active

━━━━━━━━━━━━━━━━━━━━━━
"""
            try:
                await self._alert_callback(message)
            except Exception as e:
                logger.error(f"Failed to send pause notification: {e}")
    
    async def _send_straddle(self, event: NewsEvent):
        """Send straddle setup at T-3."""
        if not self._straddle_callback:
            # Use regular alert callback if no straddle callback
            if self._alert_callback:
                message = self._format_straddle_message(event)
                try:
                    await self._alert_callback(message)
                except Exception as e:
                    logger.error(f"Failed to send straddle alert: {e}")
            return
        
        # Create straddle setup
        setup = StraddleSetup(
            event=event,
            symbol="BTCUSDT",  # Default to BTC
            entry_buffer_pct=0.15,  # 0.15% from current price
            tp_pct=0.5,
            sl_pct=0.3,
            created_at=datetime.now(pytz.utc)
        )
        
        try:
            await self._straddle_callback(setup)
        except Exception as e:
            logger.error(f"Straddle callback error: {e}")
    
    def _format_straddle_message(self, event: NewsEvent) -> str:
        """Format a basic straddle alert message."""
        return f"""
📈 <b>STRADDLE SETUP - NEWS TRAP</b>

🔥 Event: <b>{event.title}</b>
🕐 In: 3 MINUTES

<b>Strategy:</b>
• Place LONG order above current price (+0.15%)
• Place SHORT order below current price (-0.15%)
• First order hit cancels the other
• TP: 0.5% | SL: 0.3%

⚡ React fast when news drops!

━━━━━━━━━━━━━━━━━━━━━━
"""
    
    async def _start_monitoring(self, event: NewsEvent):
        """Start monitoring for news result at T+0."""
        if self._alert_callback:
            message = f"""
👀 <b>NEWS RELEASED - MONITORING</b>

🔥 <b>{event.title}</b>
🕐 Released at: {event.date.strftime("%H:%M")} UTC

📊 Watching for result...
⏳ Trading resumes in ~15 minutes

━━━━━━━━━━━━━━━━━━━━━━
"""
            try:
                await self._alert_callback(message)
            except Exception as e:
                logger.error(f"Failed to send monitoring notification: {e}")
    
    async def _do_resume(self, reason: str):
        """Resume trading after cooldown."""
        if not self.is_trading_paused:
            return
        
        self.is_trading_paused = False
        event = self._paused_for
        self._paused_for = None
        
        # Call resume callback
        if self._resume_callback:
            try:
                await self._resume_callback()
            except Exception as e:
                logger.error(f"Resume callback error: {e}")
        
        # Send resume notification
        if self._alert_callback:
            message = f"""
✅ <b>TRADING RESUMED</b>

📊 Reason: {reason}
🕐 Resumed at: {datetime.now(pytz.utc).strftime("%H:%M")} UTC

🚀 Bot is now scanning for setups

━━━━━━━━━━━━━━━━━━━━━━
"""
            try:
                await self._alert_callback(message)
            except Exception as e:
                logger.error(f"Failed to send resume notification: {e}")
    
    def _cleanup_old_events(self):
        """Remove events that are past cooldown."""
        to_remove = []
        
        for event_id, state in self._active_events.items():
            if state == NewsWorkflowState.COOLDOWN:
                # Event is done, schedule removal
                to_remove.append(event_id)
        
        for event_id in to_remove:
            del self._active_events[event_id]
    
    def force_resume(self):
        """Force resume trading (emergency override)."""
        asyncio.create_task(self._do_resume("Manual override"))
    
    def get_status(self) -> dict:
        """Get current workflow status."""
        return {
            "is_paused": self.is_trading_paused,
            "paused_for": self._paused_for.title if self._paused_for else None,
            "active_events": len(self._active_events),
            "event_states": {
                k: v.value for k, v in self._active_events.items()
            }
        }
