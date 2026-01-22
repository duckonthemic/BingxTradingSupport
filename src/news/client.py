"""
News Client - Fetch economic calendar data from ForexFactory and fallback sources.
"""

import logging
import aiohttp
import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup
import hashlib

from .models import NewsEvent, NewsImpact

logger = logging.getLogger(__name__)


class NewsClient:
    """Client for fetching economic calendar data."""
    
    # Primary source: ForexFactory JSON (via FairEconomy mirror)
    PRIMARY_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
    
    # Alternative: Different mirror
    ALT_URL = "https://cdn.jsdelivr.net/gh/fx-library/calendar@main/week.json"
    
    # Fallback: Investing.com (requires scraping - often blocked)
    FALLBACK_URL = "https://www.investing.com/economic-calendar/"
    
    # Request settings
    TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAY = 5
    
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: List[NewsEvent] = []
        self._cache_time: Optional[datetime] = None
        self._cache_ttl = 60  # Cache for 60 seconds in IDLE mode
        self.consecutive_failures = 0
    
    async def connect(self):
        """Initialize HTTP session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
            )
        logger.info("📰 NewsClient connected")
    
    async def close(self):
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
    
    async def fetch_weekly_calendar(self, use_cache: bool = True) -> List[NewsEvent]:
        """
        Fetch economic calendar for this week.
        
        Args:
            use_cache: Whether to use cached data if available
            
        Returns:
            List of NewsEvent objects
        """
        # Check cache
        if use_cache and self._cache and self._cache_time:
            cache_age = (datetime.now(timezone.utc) - self._cache_time).total_seconds()
            if cache_age < self._cache_ttl:
                return self._cache
        
        # Try primary source
        events = await self._fetch_from_forexfactory()
        
        if events:
            self._cache = events
            self._cache_time = datetime.now(timezone.utc)
            self.consecutive_failures = 0
            logger.info(f"📰 Fetched {len(events)} events from ForexFactory")
            return events
        
        # Try alternative source
        logger.warning("Primary source failed, trying alternative...")
        events = await self._fetch_from_alternative()
        
        if events:
            self._cache = events
            self._cache_time = datetime.now(timezone.utc)
            self.consecutive_failures = 0
            logger.info(f"📰 Fetched {len(events)} events from alternative")
            return events
        
        # All sources failed
        self.consecutive_failures += 1
        logger.error(f"All news sources failed. Consecutive failures: {self.consecutive_failures}")
        
        # Return cached data if available
        if self._cache:
            logger.warning("Using stale cache data")
            return self._cache
        
        return []
    
    async def _fetch_from_alternative(self) -> List[NewsEvent]:
        """Fetch from alternative JSON source."""
        try:
            if not self._session:
                await self.connect()
            
            # Try different calendar APIs
            alt_urls = [
                "https://raw.githubusercontent.com/fx-library/calendar/main/week.json",
                "https://economiccalendar.fxstreet.com/eventdatejson?d=1&c=&f=&fcs=&v=",
            ]
            
            for url in alt_urls:
                try:
                    async with self._session.get(
                        url,
                        timeout=aiohttp.ClientTimeout(total=self.TIMEOUT)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            events = self._parse_forexfactory_data(data)
                            if events:
                                return events
                except Exception as e:
                    logger.debug(f"Alt source {url} failed: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Alternative source error: {e}")
        
        return []
    
    async def _fetch_from_forexfactory(self) -> List[NewsEvent]:
        """Fetch from ForexFactory JSON API."""
        try:
            if not self._session:
                await self.connect()
            
            async with self._session.get(
                self.PRIMARY_URL,
                timeout=aiohttp.ClientTimeout(total=self.TIMEOUT)
            ) as response:
                if response.status == 429:
                    logger.warning("ForexFactory rate limited, waiting 60s...")
                    await asyncio.sleep(60)
                    return []
                elif response.status != 200:
                    logger.error(f"ForexFactory returned {response.status}")
                    return []
                
                data = await response.json()
                return self._parse_forexfactory_data(data)
                
        except asyncio.TimeoutError:
            logger.error("ForexFactory request timeout")
        except aiohttp.ClientError as e:
            logger.error(f"ForexFactory client error: {e}")
        except Exception as e:
            logger.error(f"ForexFactory error: {e}")
        
        return []
    
    def _parse_forexfactory_data(self, data: List[Dict[str, Any]]) -> List[NewsEvent]:
        """Parse ForexFactory JSON response."""
        events = []
        
        for item in data:
            try:
                # Parse date
                date_str = item.get("date", "")
                if not date_str:
                    continue
                
                # ForexFactory format: "2026-01-20T13:30:00-05:00"
                try:
                    import pytz
                    event_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                    # Convert to UTC and keep timezone-aware
                    event_date = event_date.astimezone(pytz.utc)
                except:
                    continue
                
                # Parse impact
                impact_str = item.get("impact", "Low")
                impact = {
                    "High": NewsImpact.HIGH,
                    "Medium": NewsImpact.MEDIUM,
                    "Low": NewsImpact.LOW,
                    "Holiday": NewsImpact.HOLIDAY
                }.get(impact_str, NewsImpact.LOW)
                
                # Generate ID
                event_id = hashlib.md5(
                    f"{item.get('title', '')}{date_str}{item.get('country', '')}".encode()
                ).hexdigest()[:12]
                
                event = NewsEvent(
                    id=event_id,
                    title=item.get("title", "Unknown"),
                    currency=item.get("country", ""),
                    impact=impact,
                    date=event_date,
                    forecast=item.get("forecast"),
                    previous=item.get("previous"),
                    actual=item.get("actual")
                )
                
                events.append(event)
                
            except Exception as e:
                logger.debug(f"Failed to parse event: {e}")
                continue
        
        logger.info(f"📰 Fetched {len(events)} events from ForexFactory")
        return events
    
    async def _fetch_from_fallback(self) -> List[NewsEvent]:
        """Fetch from Investing.com as fallback (scraping)."""
        try:
            if not self._session:
                await self.connect()
            
            async with self._session.get(
                self.FALLBACK_URL,
                timeout=aiohttp.ClientTimeout(total=self.TIMEOUT)
            ) as response:
                if response.status != 200:
                    logger.error(f"Investing.com returned {response.status}")
                    return []
                
                html = await response.text()
                return self._parse_investing_html(html)
                
        except Exception as e:
            logger.error(f"Fallback source error: {e}")
        
        return []
    
    def _parse_investing_html(self, html: str) -> List[NewsEvent]:
        """Parse Investing.com HTML (simplified)."""
        events = []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            rows = soup.select('tr.js-event-item')
            
            for row in rows:
                try:
                    # Extract data from row
                    currency = row.get('data-country', '')
                    
                    # Get impact from bull icons
                    bulls = row.select('.sentiment')
                    impact_level = len([b for b in bulls if 'active' in b.get('class', [])])
                    impact = {
                        3: NewsImpact.HIGH,
                        2: NewsImpact.MEDIUM,
                        1: NewsImpact.LOW
                    }.get(impact_level, NewsImpact.LOW)
                    
                    title_elem = row.select_one('.event')
                    title = title_elem.text.strip() if title_elem else "Unknown"
                    
                    time_elem = row.select_one('.time')
                    time_str = time_elem.text.strip() if time_elem else ""
                    
                    # Parse time (simplified - would need date context)
                    # This is a simplified implementation
                    
                    event_id = hashlib.md5(
                        f"{title}{time_str}{currency}".encode()
                    ).hexdigest()[:12]
                    
                    # Simplified - actual implementation needs better date parsing
                    event = NewsEvent(
                        id=event_id,
                        title=title,
                        currency=currency.upper(),
                        impact=impact,
                        date=datetime.utcnow(),  # Placeholder
                        forecast=None,
                        previous=None,
                        actual=None
                    )
                    
                    events.append(event)
                    
                except Exception as e:
                    continue
            
            logger.info(f"📰 Fetched {len(events)} events from Investing.com")
            
        except Exception as e:
            logger.error(f"Failed to parse Investing.com: {e}")
        
        return events
    
    async def get_upcoming_events(
        self,
        hours: int = 24,
        currency: Optional[str] = None
    ) -> List[NewsEvent]:
        """
        Get events within the next X hours.
        
        Args:
            hours: Number of hours to look ahead
            currency: Filter by currency (e.g., "USD")
            
        Returns:
            List of upcoming NewsEvent objects sorted by date
        """
        events = await self.fetch_weekly_calendar()
        
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours)
        
        upcoming = []
        for event in events:
            if event.date > now and event.date <= cutoff:
                if currency is None or event.currency == currency:
                    upcoming.append(event)
        
        # Sort by date
        upcoming.sort(key=lambda e: e.date)
        
        return upcoming
    
    async def get_event_by_id(self, event_id: str) -> Optional[NewsEvent]:
        """Get a specific event by ID (to check for Actual updates)."""
        events = await self.fetch_weekly_calendar(use_cache=False)
        
        for event in events:
            if event.id == event_id:
                return event
        
        return None
    
    def set_cache_ttl(self, seconds: int):
        """Set cache TTL (for adaptive polling)."""
        self._cache_ttl = seconds
