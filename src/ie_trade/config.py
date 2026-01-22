"""
IE Trade Configuration

All settings for the IE Trade module including:
- Kill Zones (London, New York sessions)
- Top coins to scan
- FVG age limits
- Premium/Discount thresholds
- Position limits
"""

from dataclasses import dataclass, field
from typing import List, Tuple
from datetime import timedelta


@dataclass
class KillZone:
    """Trading session kill zone definition"""
    name: str
    start_hour: int  # Vietnam time (UTC+7)
    end_hour: int
    
    def is_active(self, hour: int) -> bool:
        """Check if current hour is within this kill zone"""
        return self.start_hour <= hour < self.end_hour


@dataclass 
class IETradeConfig:
    """Main configuration for IE Trade module"""
    
    # ==================== KILL ZONES ====================
    # Trading sessions when SMC setups are most effective
    # Times in Vietnam timezone (UTC+7)
    
    LONDON_OPEN: KillZone = field(default_factory=lambda: KillZone(
        name="London",
        start_hour=14,  # 14:00 VN = 07:00 UTC
        end_hour=17     # 17:00 VN = 10:00 UTC
    ))
    
    NEW_YORK_OPEN: KillZone = field(default_factory=lambda: KillZone(
        name="New York",
        start_hour=19,  # 19:00 VN = 12:00 UTC  
        end_hour=23     # 23:00 VN = 16:00 UTC
    ))
    
    # ==================== TOP COINS ====================
    # Only scan high-volume coins for better liquidity & cleaner setups
    
    TOP_COINS: List[str] = field(default_factory=lambda: [
        "BTC-USDT",   # King - always prioritize
        "ETH-USDT",   # Queen
        "SOL-USDT",
        "XRP-USDT",
        "BNB-USDT",
        "DOGE-USDT",
        "ADA-USDT",
        "LINK-USDT",
        "AVAX-USDT",
        "DOT-USDT",
        "MATIC-USDT",
        "SHIB-USDT",
        "LTC-USDT",
        "UNI-USDT",
        "ATOM-USDT",
    ])
    
    # Coins to always prioritize (correlation leaders)
    PRIORITY_COINS: List[str] = field(default_factory=lambda: [
        "BTC-USDT",  # King
        "ETH-USDT",  # Queen
    ])
    
    # ==================== FVG SETTINGS ====================
    
    # H1 FVG maximum age (hours) - older FVGs are considered stale
    FVG_MAX_AGE_HOURS: int = 24
    
    # Preferred: FVG from current or previous session
    FVG_PREFER_SAME_SESSION: bool = True
    
    # Minimum FVG size as percentage of price
    FVG_MIN_SIZE_PCT: float = 0.1  # 0.1% minimum gap
    
    # Maximum FVG size (too large = suspicious)
    FVG_MAX_SIZE_PCT: float = 2.0  # 2% maximum gap
    
    # ==================== PREMIUM/DISCOUNT ====================
    
    # Fibonacci levels for zone classification
    # SHORT: Price must be in Premium (above 50%)
    # LONG: Price must be in Discount (below 50%)
    
    PREMIUM_THRESHOLD: float = 0.618  # 61.8% - Golden Pocket (strict)
    DISCOUNT_THRESHOLD: float = 0.382  # 38.2% - Golden Pocket (strict)
    
    # Alternative: Use 50% for more signals (less strict)
    # PREMIUM_THRESHOLD: float = 0.50
    # DISCOUNT_THRESHOLD: float = 0.50
    
    # ==================== MSS SETTINGS ====================
    
    # Minimum body ratio for displacement candle
    MSS_MIN_BODY_RATIO: float = 0.70  # 70% body = strong displacement
    
    # Swing point lookback for MSS detection
    MSS_SWING_LOOKBACK: int = 5  # candles
    
    # Maximum candles to look back for MSS
    MSS_MAX_LOOKBACK: int = 20  # candles
    
    # ==================== ENTRY SETTINGS ====================
    
    # Entry at 50% of FVG (Optimal Trade Entry)
    ENTRY_FVG_LEVEL: float = 0.5
    
    # Minimum Risk:Reward ratio to TP1
    MIN_RR_RATIO: float = 2.0  # 1:2 minimum
    
    # ==================== POSITION MANAGEMENT ====================
    
    # Maximum concurrent positions from IE Trade
    MAX_OPEN_POSITIONS: int = 1
    
    # If multiple setups, prioritize:
    # 1. BTC (King)
    # 2. ETH (Queen)  
    # 3. Best R:R
    PRIORITIZE_BTC: bool = True
    
    # ==================== DAILY BIAS ====================
    
    # Hour to send daily bias reminder (Vietnam time)
    BIAS_REMINDER_HOUR: int = 7  # 7:00 AM VN
    
    # Bias expires after this many hours if not confirmed
    BIAS_EXPIRY_HOURS: int = 24
    
    # ==================== SCANNING ====================
    
    # Scan interval in seconds
    SCAN_INTERVAL_SECONDS: int = 30
    
    # Only scan during kill zones (recommended)
    SCAN_ONLY_KILL_ZONES: bool = False  # Scan 24/7 but alert only in KZ
    
    # Alert only during kill zones
    ALERT_ONLY_KILL_ZONES: bool = True
    
    # ==================== TIMEZONE ====================
    
    # Timezone for time calculations
    TIMEZONE: str = "Asia/Ho_Chi_Minh"
    
    # Alias for backwards compatibility
    @property
    def timezone(self) -> str:
        return self.TIMEZONE
    
    # ==================== LOGGING ====================
    
    # Log prefix for IE Trade
    LOG_PREFIX: str = "🎯 IE"
    
    # Sheet note identifier
    SHEET_NOTE: str = "IE trade"
    
    # ==================== HELPER METHODS ====================
    
    def is_kill_zone(self, hour: int) -> Tuple[bool, str]:
        """
        Check if current hour is within any kill zone
        
        Args:
            hour: Current hour in Vietnam time (0-23)
            
        Returns:
            Tuple of (is_active, zone_name)
        """
        if self.LONDON_OPEN.is_active(hour):
            return True, self.LONDON_OPEN.name
        if self.NEW_YORK_OPEN.is_active(hour):
            return True, self.NEW_YORK_OPEN.name
        return False, ""
    
    def get_fvg_max_age(self) -> timedelta:
        """Get FVG maximum age as timedelta"""
        return timedelta(hours=self.FVG_MAX_AGE_HOURS)
    
    def is_priority_coin(self, symbol: str) -> bool:
        """Check if coin is in priority list (BTC, ETH)"""
        return symbol in self.PRIORITY_COINS
    
    def get_coin_priority(self, symbol: str) -> int:
        """
        Get priority score for coin (lower = higher priority)
        BTC = 0, ETH = 1, others = 2
        """
        if symbol == "BTC-USDT":
            return 0
        elif symbol == "ETH-USDT":
            return 1
        else:
            return 2


# Default configuration instance
DEFAULT_CONFIG = IETradeConfig()
