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
    account_balance: float = float(os.getenv("ACCOUNT_BALANCE", 25.0))  # Default $25
    max_risk_per_trade: float = float(os.getenv("MAX_RISK_PER_TRADE", 1.0))  # Max $1/trade
    risk_per_trade_pct: float = 0.0  # Deprecated, use max_risk_per_trade
    max_position_pct: float = 4.0  # 4% ($1 of $25)
    daily_max_loss_pct: float = float(os.getenv("DAILY_MAX_LOSS_PCT", 5.0))  # 5%
    weekly_max_loss_pct: float = float(os.getenv("WEEKLY_MAX_LOSS_PCT", 15.0))  # 15%
    btc_dump_threshold: float = float(os.getenv("BTC_DUMP_THRESHOLD", -1.0))  # -1%


@dataclass 
class TradingConfig:
    """
    Trading size and leverage settings - RISK OPTIMIZED v3.0
    
    Based on TradeHistory2 analysis:
    - x500 leverage: -4463.7% PnL (DISASTER) => ELIMINATED
    - x15 leverage: +2591.6% PnL => KEEP AS DEFAULT
    - x100 leverage: +1410.7% PnL => CAP
    """
    fixed_position_usd: float = float(os.getenv("FIXED_POSITION_USD", 1.0))  # $1 per trade (max)
    
    # === LEVERAGE CAPS (CRITICAL SAFETY RULES) ===
    # ABSOLUTE MAX: x100 (NO x500 EVER - destroyed -4463% PnL)
    absolute_max_leverage: int = 100
    
    # Default leverage by asset type
    default_leverage: int = 15         # Altcoins (safest, +2591.6% PnL)
    major_coin_leverage: int = 75      # BTC, ETH, SOL (reduced from 100)
    gold_leverage: int = 100           # XAUT, PAXG (reduced from 500!)
    index_leverage: int = 50           # Indices (NASDAQ, S&P500, etc.)
    
    # DEPRECATED - kept for compatibility
    min_leverage_trap: int = 100       # Reduced from 300
    min_leverage_top_trap: int = 75    # Reduced from 100
    max_leverage: int = 100            # Hard cap (was 50 for altcoins only)
    
    # Volume thresholds for leverage tiers
    small_cap_max_volume: float = float(os.getenv("SMALL_CAP_MAX_VOL", 50_000_000))  # $50M
    large_cap_min_volume: float = float(os.getenv("LARGE_CAP_MIN_VOL", 200_000_000))  # $200M
    
    # === CHECKLIST REQUIREMENTS ===
    # Based on analysis: 3/3 checklist = +1455.6% PnL, 2/3 = -3471.4% PnL
    long_min_checklist: int = 3        # LONG requires 3/3 checklist
    short_min_checklist: int = 2       # SHORT works with 2/3+
    
    # === DIRECTION PREFERENCE ===
    # LONG: 32.6% WR, -4339.4% PnL => RESTRICT
    # SHORT: 57.9% WR, +2433.5% PnL => PREFER
    prefer_short: bool = True
    block_sfp_long: bool = True        # SFP LONG: -3194.3% PnL
    
    # === ICT LONG EXCEPTIONS ===
    # Allow LONG for super-setups with 3+ ICT confluences
    allow_silver_bullet_long_super: bool = True   # Silver Bullet super-setup LONG
    allow_ict_reversal_long: bool = True           # ICT reversal LONG with dream setup


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
