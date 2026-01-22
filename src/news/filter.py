"""
News Filter - Filter important economic events.
"""

import logging
from typing import List

from .models import NewsEvent, NewsImpact, IMPORTANT_KEYWORDS

logger = logging.getLogger(__name__)


class NewsFilter:
    """Filter for identifying important economic events."""
    
    def __init__(self):
        self.keywords = IMPORTANT_KEYWORDS
        self.tracked_currencies = ["USD"]  # Can expand to EUR, GBP, etc.
    
    def should_track(self, event: NewsEvent) -> bool:
        """
        Determine if an event should be tracked.
        
        Logic:
        - Track if: USD + High Impact
        - Track if: USD + Medium Impact + Contains important keyword
        
        Args:
            event: NewsEvent to evaluate
            
        Returns:
            True if event should be tracked
        """
        # Must be in tracked currencies
        if event.currency not in self.tracked_currencies:
            return False
        
        # High impact USD events - always track
        if event.impact == NewsImpact.HIGH:
            return True
        
        # Medium impact with important keywords
        if event.impact == NewsImpact.MEDIUM:
            title_upper = event.title.upper()
            for keyword in self.keywords:
                if keyword.upper() in title_upper:
                    logger.debug(f"Tracking medium impact: {event.title} (keyword: {keyword})")
                    return True
        
        return False
    
    def filter_events(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """
        Filter a list of events to only include trackable ones.
        
        Args:
            events: List of NewsEvent objects
            
        Returns:
            Filtered list of important events
        """
        tracked = []
        
        for event in events:
            if self.should_track(event):
                event.is_tracked = True
                tracked.append(event)
        
        logger.info(f"📋 Filtered {len(tracked)}/{len(events)} events as important")
        return tracked
    
    def get_high_impact_only(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """Get only high impact events."""
        return [
            e for e in events 
            if e.currency in self.tracked_currencies and e.impact == NewsImpact.HIGH
        ]
    
    def categorize_by_day(self, events: List[NewsEvent]) -> dict:
        """
        Categorize events by day of week.
        
        Returns:
            Dict with day names as keys and list of events as values
        """
        by_day = {}
        
        for event in events:
            day_name = event.date.strftime("%A")  # Monday, Tuesday, etc.
            if day_name not in by_day:
                by_day[day_name] = []
            by_day[day_name].append(event)
        
        return by_day
    
    def get_impact_emoji(self, impact: NewsImpact) -> str:
        """Get emoji for impact level."""
        return {
            NewsImpact.HIGH: "🔥",
            NewsImpact.MEDIUM: "⚡",
            NewsImpact.LOW: "📊",
            NewsImpact.HOLIDAY: "🏖️"
        }.get(impact, "📰")
    
    def get_event_keywords(self, event: NewsEvent) -> List[str]:
        """Get matching keywords for an event."""
        title_upper = event.title.upper()
        matching = []
        
        for keyword in self.keywords:
            if keyword.upper() in title_upper:
                matching.append(keyword)
        
        return matching
