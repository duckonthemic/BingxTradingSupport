"""
News Module - Economic calendar and news-aware trading.

This module provides:
- Economic calendar fetching (ForexFactory, Investing.com fallback)
- Session open alerts (Asia, London, US, Close)
- Adaptive polling based on time-to-event
- Trading pause/resume workflow around high-impact news
- Straddle setup alerts for news trading

Usage:
    from src.news import get_news_manager
    
    news_manager = get_news_manager()
    news_manager.configure(
        send_message=telegram_bot.send_message,
        send_pinned_message=telegram_bot.send_pinned_message,
        pause_trading=alert_manager.pause,
        resume_trading=alert_manager.resume
    )
    await news_manager.start()
"""

from .models import (
    NewsEvent,
    NewsImpact,
    PollingMode,
    SessionType,
    SessionInfo,
    NewsWorkflowState,
    StraddleSetup,
    TRADING_SESSIONS,
    IMPORTANT_KEYWORDS
)
from .client import NewsClient
from .filter import NewsFilter
from .scheduler import SessionScheduler, WeeklyReporter
from .poller import AdaptivePoller
from .trader import NewsTrader
from .manager import NewsManager, get_news_manager

__all__ = [
    # Models
    "NewsEvent",
    "NewsImpact",
    "PollingMode",
    "SessionType",
    "SessionInfo",
    "NewsWorkflowState",
    "StraddleSetup",
    "TRADING_SESSIONS",
    "IMPORTANT_KEYWORDS",
    
    # Components
    "NewsClient",
    "NewsFilter",
    "SessionScheduler",
    "WeeklyReporter",
    "AdaptivePoller",
    "NewsTrader",
    
    # Manager
    "NewsManager",
    "get_news_manager"
]
