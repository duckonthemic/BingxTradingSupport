"""
News Manager Models - Data structures for economic calendar events.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, List
from enum import Enum


class NewsImpact(Enum):
    """Impact level of news event."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    HOLIDAY = "Holiday"


class PollingMode(Enum):
    """Adaptive polling mode based on time-to-event."""
    IDLE = "IDLE"          # > 30 min, poll every 5 min
    STANDBY = "STANDBY"    # 2-30 min, poll every 1 min
    BATTLE = "BATTLE"      # < 2 min, poll every 2 sec


class SessionType(Enum):
    """Trading session types."""
    ASIA = "ASIA"
    LONDON = "LONDON"
    US = "US"
    CLOSE = "CLOSE"


class NewsWorkflowState(Enum):
    """State of news trading workflow."""
    IDLE = "IDLE"
    PRE_ALERT = "PRE_ALERT"       # T-30 min
    DEFENSE = "DEFENSE"            # T-5 min (PAUSED)
    STRADDLE = "STRADDLE"          # T-3 min
    MONITORING = "MONITORING"      # T+0 to T+15
    COOLDOWN = "COOLDOWN"          # T+15 - Resume trading


@dataclass
class NewsEvent:
    """Represents an economic calendar event."""
    title: str
    currency: str
    impact: NewsImpact
    date: datetime
    id: str = ""  # Generated from title + date if not provided
    forecast: Optional[str] = None
    previous: Optional[str] = None
    actual: Optional[str] = None
    
    # Tracking
    is_tracked: bool = False
    pre_alert_sent: bool = False
    defense_triggered: bool = False
    straddle_sent: bool = False
    result_sent: bool = False
    
    def __post_init__(self):
        """Generate id if not provided."""
        if not self.id:
            date_str = self.date.strftime("%Y%m%d_%H%M")
            self.id = f"{date_str}_{self.title[:30].replace(' ', '_')}"
    
    @property
    def is_high_impact_usd(self) -> bool:
        """Check if this is a high impact USD event."""
        return self.currency == "USD" and self.impact == NewsImpact.HIGH
    
    @property
    def time_until(self) -> Optional[timedelta]:
        """Time until event (negative if past)."""
        import pytz
        now = datetime.now(pytz.utc)
        # Ensure date is timezone-aware
        if self.date.tzinfo is None:
            event_time = pytz.utc.localize(self.date)
        else:
            event_time = self.date.astimezone(pytz.utc)
        return event_time - now
    
    @property
    def is_upcoming(self) -> bool:
        """Check if event is in the future."""
        td = self.time_until
        return td is not None and td.total_seconds() > 0
    
    @property
    def has_result(self) -> bool:
        """Check if actual result is available."""
        return self.actual is not None and self.actual != ""
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, NewsEvent):
            return self.id == other.id
        return False


@dataclass
class StraddleSetup:
    """News straddle trade setup."""
    symbol: str
    event: NewsEvent
    timestamp: datetime
    
    # Price levels from M5 candle
    reference_high: float
    reference_low: float
    
    # Entry points
    buy_stop: float    # High + 0.1%
    sell_stop: float   # Low - 0.1%
    
    # Status
    is_active: bool = True
    triggered_direction: Optional[str] = None  # "LONG" or "SHORT"
    entry_price: Optional[float] = None


@dataclass
class SessionInfo:
    """Trading session information."""
    session_type: SessionType
    hour_utc: int
    hour_utc_winter: int  # DST adjustment
    emoji: str
    flag: str
    description: str
    alert_message: str


# Session definitions
TRADING_SESSIONS = {
    SessionType.ASIA: SessionInfo(
        session_type=SessionType.ASIA,
        hour_utc=0,
        hour_utc_winter=0,
        emoji="🌏",
        flag="🇯🇵",
        description="Tokyo/Hong Kong",
        alert_message=(
            "🌏 *ASIA SESSION OPEN!* 🇯🇵\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "📊 Market usually consolidates here\n"
            "🎯 Look for Range Setup\n"
            "⏰ Duration: ~7 hours"
        )
    ),
    SessionType.LONDON: SessionInfo(
        session_type=SessionType.LONDON,
        hour_utc=7,  # Summer
        hour_utc_winter=8,  # Winter
        emoji="☕",
        flag="🇬🇧",
        description="London",
        alert_message=(
            "☕ *LONDON SESSION OPEN!* 🇬🇧\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "📈 Volume spiking!\n"
            "⚠️ Watch out for Judas Swing\n"
            "🎯 Fakeouts of Asia High/Low common"
        )
    ),
    SessionType.US: SessionInfo(
        session_type=SessionType.US,
        hour_utc=13,  # Summer (13:30)
        hour_utc_winter=14,  # Winter (14:30)
        emoji="🔔",
        flag="🇺🇸",
        description="New York",
        alert_message=(
            "🔔 *US SESSION OPEN!* 🇺🇸\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "🏦 Institutions entering\n"
            "📊 Trend acceleration expected\n"
            "⚡ High volatility period"
        )
    ),
    SessionType.CLOSE: SessionInfo(
        session_type=SessionType.CLOSE,
        hour_utc=20,  # Summer
        hour_utc_winter=21,  # Winter
        emoji="💤",
        flag="",
        description="Market Close",
        alert_message=(
            "💤 *US SESSION CLOSE*\n"
            "━━━━━━━━━━━━━━━━━━━━━\n"
            "📉 Volume decreasing\n"
            "⚠️ Be careful with low-liquidity wicks\n"
            "🌙 Market entering drift mode"
        )
    )
}


# Important keywords for Medium impact filtering
IMPORTANT_KEYWORDS = [
    'FOMC', 'Powell', 'Fed', 'Minutes',
    'CPI', 'PPI', 'NFP', 'Non-Farm',
    'GDP', 'Federal Funds Rate',
    'Unemployment', 'Retail Sales',
    'ISM', 'PCE', 'Core'
]


@dataclass
class PollingStatus:
    """Current polling status."""
    mode: PollingMode
    interval_seconds: int
    next_event: Optional[NewsEvent]
    time_to_event: int  # seconds
    events_today: int
    events_this_week: int
    
    @property
    def mode_emoji(self) -> str:
        return {
            PollingMode.IDLE: "🟢",
            PollingMode.STANDBY: "🟡",
            PollingMode.BATTLE: "🔴"
        }.get(self.mode, "⚪")
