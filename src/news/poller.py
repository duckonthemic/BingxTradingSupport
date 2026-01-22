"""
Adaptive Poller - Dynamic polling based on time-to-event.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Callable, Optional, List

import pytz

from .models import NewsEvent, PollingMode

logger = logging.getLogger(__name__)


class AdaptivePoller:
    """
    Adaptive polling system for news events.
    
    Polling Modes:
    - IDLE: No significant events in next 2 hours → Poll every 5 minutes
    - STANDBY: Event within 30-120 minutes → Poll every 1 minute
    - BATTLE: Event within 30 minutes → Poll every 2 seconds
    """
    
    # Polling intervals in seconds
    INTERVALS = {
        PollingMode.IDLE: 300,      # 5 minutes
        PollingMode.STANDBY: 60,    # 1 minute
        PollingMode.BATTLE: 2       # 2 seconds
    }
    
    # Time thresholds in minutes
    THRESHOLDS = {
        PollingMode.BATTLE: 30,     # Enter BATTLE when event < 30 mins away
        PollingMode.STANDBY: 120,   # Enter STANDBY when event < 2 hours away
    }
    
    def __init__(self):
        self.running = False
        self.mode = PollingMode.IDLE
        self._task: Optional[asyncio.Task] = None
        self._poll_callback: Optional[Callable] = None  # Fetch new events
        self._mode_change_callback: Optional[Callable] = None  # On mode change
        self._events: List[NewsEvent] = []
        self._last_mode_change = datetime.now(pytz.utc)
    
    def set_callbacks(
        self, 
        poll_callback: Callable,
        mode_change_callback: Optional[Callable] = None
    ):
        """
        Set callbacks for the poller.
        
        Args:
            poll_callback: Async function to fetch events
            mode_change_callback: Optional callback when mode changes
        """
        self._poll_callback = poll_callback
        self._mode_change_callback = mode_change_callback
    
    async def start(self):
        """Start the adaptive poller."""
        if self.running:
            return
        
        self.running = True
        self._task = asyncio.create_task(self._poller_loop())
        logger.info(f"⚡ Adaptive poller started in {self.mode.value} mode")
    
    async def stop(self):
        """Stop the adaptive poller."""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("⚡ Adaptive poller stopped")
    
    async def _poller_loop(self):
        """Main polling loop with adaptive intervals."""
        while self.running:
            try:
                # Fetch events
                if self._poll_callback:
                    self._events = await self._poll_callback()
                
                # Update polling mode based on events
                await self._update_mode()
                
                # Get current interval
                interval = self.INTERVALS[self.mode]
                
                # Sleep for interval
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poller error: {e}")
                await asyncio.sleep(60)
    
    async def _update_mode(self):
        """Update polling mode based on time to next event."""
        if not self._events:
            if self.mode != PollingMode.IDLE:
                await self._set_mode(PollingMode.IDLE)
            return
        
        # Find time to next event
        now = datetime.now(pytz.utc)
        min_minutes = float('inf')
        
        for event in self._events:
            time_diff = event.time_until
            if time_diff is not None and time_diff.total_seconds() > 0:
                minutes = time_diff.total_seconds() / 60
                min_minutes = min(min_minutes, minutes)
        
        # Determine appropriate mode
        if min_minutes <= self.THRESHOLDS[PollingMode.BATTLE]:
            new_mode = PollingMode.BATTLE
        elif min_minutes <= self.THRESHOLDS[PollingMode.STANDBY]:
            new_mode = PollingMode.STANDBY
        else:
            new_mode = PollingMode.IDLE
        
        if new_mode != self.mode:
            await self._set_mode(new_mode)
    
    async def _set_mode(self, mode: PollingMode):
        """Change polling mode."""
        old_mode = self.mode
        self.mode = mode
        self._last_mode_change = datetime.now(pytz.utc)
        
        logger.info(f"⚡ Polling mode: {old_mode.value} → {mode.value}")
        
        if self._mode_change_callback:
            try:
                await self._mode_change_callback(old_mode, mode)
            except Exception as e:
                logger.error(f"Mode change callback error: {e}")
    
    def get_status(self) -> dict:
        """Get current poller status."""
        now = datetime.now(pytz.utc)
        
        # Find next event
        next_event = None
        min_time = timedelta.max
        
        for event in self._events:
            if event.time_until and event.time_until > timedelta(0):
                if event.time_until < min_time:
                    min_time = event.time_until
                    next_event = event
        
        return {
            "mode": self.mode.value,
            "interval_seconds": self.INTERVALS[self.mode],
            "events_count": len(self._events),
            "next_event": next_event.title if next_event else None,
            "next_event_in": str(min_time) if next_event else None,
            "mode_duration": str(now - self._last_mode_change)
        }
    
    def force_mode(self, mode: PollingMode):
        """Force a specific polling mode (for testing/override)."""
        self.mode = mode
        self._last_mode_change = datetime.now(pytz.utc)
        logger.warning(f"⚡ Polling mode FORCED to: {mode.value}")
    
    @property
    def current_interval(self) -> int:
        """Get current polling interval in seconds."""
        return self.INTERVALS[self.mode]
    
    @property
    def events(self) -> List[NewsEvent]:
        """Get currently tracked events."""
        return self._events
