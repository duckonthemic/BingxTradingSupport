"""
Configuration management for BingX Zone Alert Bot.
Loads settings from environment variables with sensible defaults.
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env file
load_dotenv()


@dataclass
class RedisConfig:
    """Redis connection configuration."""
    host: str = os.getenv("REDIS_HOST", "localhost")
    port: int = int(os.getenv("REDIS_PORT", 6379))
    db: int = int(os.getenv("REDIS_DB", 0))
    password: str = os.getenv("REDIS_PASSWORD", "")


@dataclass
class TelegramConfig:
    """Telegram bot configuration."""
    bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


@dataclass
class BingXConfig:
    """BingX API configuration."""
    ws_url: str = os.getenv("BINGX_WS_URL", "wss://open-api-ws.bingx.com/market")
    rest_url: str = os.getenv("BINGX_REST_URL", "https://open-api.bingx.com")


@dataclass
class FilterConfig:
    """Coin filtering configuration."""
    # Volume filter - INCREASED to $10M minimum for quality
    min_volume_24h: float = float(os.getenv("MIN_VOLUME_24H", 10_000_000))  # $10M min
    max_volume_24h: float = float(os.getenv("MAX_VOLUME_24H", 500_000_000))  # $500M max
    mid_cap_threshold: float = float(os.getenv("MID_CAP_THRESHOLD", 50_000_000))  # $50M


@dataclass
class ZoneDetectionConfig:
    """Zone detection thresholds."""
    resistance_proximity: float = float(os.getenv("RESISTANCE_PROXIMITY", 0.01))
    support_proximity: float = float(os.getenv("SUPPORT_PROXIMITY", 0.01))
    bb_squeeze_threshold: float = float(os.getenv("BB_SQUEEZE_THRESHOLD", 0.05))
    volume_spike_multiplier: float = float(os.getenv("VOLUME_SPIKE_MULTIPLIER", 4.0))
    volume_surge_multiplier: float = float(os.getenv("VOLUME_SURGE_MULTIPLIER", 2.0))
    adx_trend_threshold: float = float(os.getenv("ADX_TREND_THRESHOLD", 25))
    rsi_overbought: float = float(os.getenv("RSI_OVERBOUGHT", 75))
    rsi_oversold: float = float(os.getenv("RSI_OVERSOLD", 40))


@dataclass
class IndicatorConfig:
    """Technical indicator settings."""
    ema_fast: int = int(os.getenv("EMA_FAST", 34))
    ema_slow: int = int(os.getenv("EMA_SLOW", 89))
    rsi_period: int = int(os.getenv("RSI_PERIOD", 14))
    bb_period: int = int(os.getenv("BB_PERIOD", 20))
    bb_std: float = float(os.getenv("BB_STD", 2.0))
    adx_period: int = int(os.getenv("ADX_PERIOD", 14))
    atr_period: int = int(os.getenv("ATR_PERIOD", 14))
    volume_lookback: int = int(os.getenv("VOLUME_LOOKBACK", 20))


@dataclass
class TimingConfig:
    """Timing and cooldown settings."""
    scan_interval: int = int(os.getenv("SCAN_INTERVAL", 90))  # 1.5 min for scalping
    context_update_interval: int = int(os.getenv("CONTEXT_UPDATE_INTERVAL", 30))
    alert_cooldown: int = int(os.getenv("ALERT_COOLDOWN", 900))  # 15 min per coin+zone
    priority_1_cooldown: int = int(os.getenv("PRIORITY_1_COOLDOWN", 900))
    priority_2_batch_interval: int = int(os.getenv("PRIORITY_2_BATCH_INTERVAL", 300))
    different_zone_cooldown: int = int(os.getenv("DIFFERENT_ZONE_COOLDOWN", 300))


@dataclass
class RiskConfig:
    """Risk management settings."""
    account_balance: float = float(os.getenv("ACCOUNT_BALANCE", 25.0))  # Tổng vốn mặc định $25
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))  # Tối đa $1/lệnh
    risk_per_trade_pct: float = 0.0  # Deprecated, dùng max_risk_per_trade
    max_position_pct: float = 4.0  # 4% (tương đương $1/25$)
    daily_max_loss_pct: float = float(os.getenv("DAILY_MAX_LOSS_PCT", 5.0))  # 5%
    weekly_max_loss_pct: float = float(os.getenv("WEEKLY_MAX_LOSS_PCT", 15.0))  # 15%
    btc_dump_threshold: float = float(os.getenv("BTC_DUMP_THRESHOLD", -1.0))  # -1%


@dataclass 
class TradingConfig:
    """Trading size and leverage settings."""
    fixed_position_usd: float = float(os.getenv("FIXED_POSITION_USD", 1.0))  # $1 per trade (max)
    
    # Leverage rules by type
    min_leverage_trap: int = 300  # Vàng/bạc/chứng chỉ/quỹ bẫy: tối thiểu x300
    min_leverage_top_trap: int = 100  # Top coin trap: tối thiểu x100
    default_leverage: int = 15  # Default for normal trades
    max_leverage: int = 50  # Max for altcoins
    # Volume thresholds for leverage tiers
    small_cap_max_volume: float = float(os.getenv("SMALL_CAP_MAX_VOL", 50_000_000))  # $50M
    large_cap_min_volume: float = float(os.getenv("LARGE_CAP_MIN_VOL", 200_000_000))  # $200M


@dataclass
class AppConfig:
    """Main application configuration."""
    redis: RedisConfig
    telegram: TelegramConfig
    bingx: BingXConfig
    filter: FilterConfig
    zone_detection: ZoneDetectionConfig
    indicator: IndicatorConfig
    timing: TimingConfig
    risk: "RiskConfig"
    trading: "TradingConfig"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"


def load_config() -> AppConfig:
    """Load and return the application configuration."""
    return AppConfig(
        redis=RedisConfig(),
        telegram=TelegramConfig(),
        bingx=BingXConfig(),
        filter=FilterConfig(),
        zone_detection=ZoneDetectionConfig(),
        indicator=IndicatorConfig(),
        timing=TimingConfig(),
        risk=RiskConfig(),
        trading=TradingConfig()
    )


# Global config instance
config = load_config()


if __name__ == "__main__":
    # Print config for debugging
    import json
    from dataclasses import asdict
    
    cfg = load_config()
    print("=== BingX Zone Alert Bot Configuration ===")
    print(json.dumps(asdict(cfg), indent=2, default=str))
