"""
Precious Metals & Large Cap Leverage Configuration.
Special handling for Gold (PAXG), Silver, and high-volume assets.

Rationale:
- Precious metals have lower volatility than crypto
- Higher leverage is safer with proper risk management
- Different ATR multipliers for SL/TP
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    """Asset classification for leverage rules."""
    PRECIOUS_METAL = "PRECIOUS_METAL"  # PAXG, etc.
    LARGE_CAP = "LARGE_CAP"            # BTC, ETH, BNB
    MID_CAP = "MID_CAP"                # Top 20-50 coins
    SMALL_CAP = "SMALL_CAP"            # Altcoins
    MEME = "MEME"                       # High volatility meme coins


@dataclass
class LeverageProfile:
    """Leverage profile for an asset class."""
    min_leverage: int
    max_leverage: int
    default_leverage: int
    sl_atr_multiplier: float
    tp_atr_multiplier: float
    max_position_pct: float  # Max position as % of account


# Precious Metals Configuration
PRECIOUS_METALS_CONFIG = {
    "PAXG-USDT": {
        "name": "Gold (PAXG)",
        "class": AssetClass.PRECIOUS_METAL,
        "leverage_profile": LeverageProfile(
            min_leverage=100,
            max_leverage=500,
            default_leverage=200,
            sl_atr_multiplier=0.5,   # Tighter SL (lower volatility)
            tp_atr_multiplier=1.0,   # Reasonable TP
            max_position_pct=5.0     # 5% max position
        ),
        "volatility_factor": 0.3,    # 30% of crypto volatility
        "correlation": {
            "BTC": 0.2,  # Low correlation with BTC
            "USD": -0.8  # Inverse correlation with USD strength
        }
    },
    # Can add more precious metals when available
    # "XAGUSD": {...},  # Silver
    # "XPTUSD": {...},  # Platinum
}


# Large Cap Crypto Configuration
LARGE_CAP_CONFIG = {
    "BTC-USDT": {
        "name": "Bitcoin",
        "class": AssetClass.LARGE_CAP,
        "leverage_profile": LeverageProfile(
            min_leverage=50,
            max_leverage=125,
            default_leverage=100,
            sl_atr_multiplier=1.0,
            tp_atr_multiplier=2.0,
            max_position_pct=10.0
        ),
        "min_volume_24h": 500_000_000
    },
    "ETH-USDT": {
        "name": "Ethereum",
        "class": AssetClass.LARGE_CAP,
        "leverage_profile": LeverageProfile(
            min_leverage=50,
            max_leverage=100,
            default_leverage=75,
            sl_atr_multiplier=1.0,
            tp_atr_multiplier=2.0,
            max_position_pct=10.0
        ),
        "min_volume_24h": 200_000_000
    },
    "BNB-USDT": {
        "name": "BNB",
        "class": AssetClass.LARGE_CAP,
        "leverage_profile": LeverageProfile(
            min_leverage=25,
            max_leverage=75,
            default_leverage=50,
            sl_atr_multiplier=1.2,
            tp_atr_multiplier=2.0,
            max_position_pct=8.0
        ),
        "min_volume_24h": 100_000_000
    },
    "SOL-USDT": {
        "name": "Solana",
        "class": AssetClass.LARGE_CAP,
        "leverage_profile": LeverageProfile(
            min_leverage=20,
            max_leverage=75,
            default_leverage=50,
            sl_atr_multiplier=1.2,
            tp_atr_multiplier=2.5,
            max_position_pct=8.0
        ),
        "min_volume_24h": 80_000_000
    },
}


# Meme coins - extra caution
MEME_COINS = {
    "DOGE-USDT", "SHIB-USDT", "PEPE-USDT", "WIF-USDT", 
    "BONK-USDT", "FLOKI-USDT", "MEME-USDT", "BRETT-USDT"
}


class LeverageManager:
    """
    Dynamic leverage manager based on asset class and market conditions.
    
    Features:
    - Asset class detection
    - Dynamic leverage based on volatility
    - Confidence-based leverage adjustment
    - Special handling for precious metals
    """
    
    def __init__(self):
        self.precious_metals = PRECIOUS_METALS_CONFIG
        self.large_caps = LARGE_CAP_CONFIG
        self.meme_coins = MEME_COINS
    
    def get_asset_class(self, symbol: str) -> AssetClass:
        """Determine asset class for a symbol."""
        if symbol in self.precious_metals:
            return AssetClass.PRECIOUS_METAL
        if symbol in self.large_caps:
            return AssetClass.LARGE_CAP
        if symbol in self.meme_coins:
            return AssetClass.MEME
        
        # Default classification based on naming
        if "PAXG" in symbol or "XAU" in symbol:
            return AssetClass.PRECIOUS_METAL
        
        return AssetClass.SMALL_CAP
    
    def get_leverage(
        self,
        symbol: str,
        confidence_score: int,
        volume_24h: float = 0,
        atr_percent: float = 0
    ) -> Tuple[int, str]:
        """
        Calculate optimal leverage for a trade.
        
        Args:
            symbol: Trading pair symbol
            confidence_score: Signal confidence (0-100)
            volume_24h: 24h trading volume in USD
            atr_percent: ATR as percentage of price
        
        Returns:
            Tuple of (leverage, reasoning)
        """
        asset_class = self.get_asset_class(symbol)
        
        # Get base profile
        if symbol in self.precious_metals:
            config = self.precious_metals[symbol]
            profile = config["leverage_profile"]
            base_leverage = profile.default_leverage
            
            # Precious metals: Higher leverage for high confidence
            if confidence_score >= 75:  # DIAMOND
                leverage = profile.max_leverage  # x500
                reason = f"💎 DIAMOND precious metal setup - MAX leverage x{leverage}"
            elif confidence_score >= 55:  # GOLD
                leverage = profile.default_leverage  # x200
                reason = f"🥇 GOLD precious metal setup - x{leverage}"
            else:
                leverage = profile.min_leverage  # x100
                reason = f"Standard precious metal - x{leverage}"
            
            return leverage, reason
        
        elif symbol in self.large_caps:
            config = self.large_caps[symbol]
            profile = config["leverage_profile"]
            
            if confidence_score >= 75:
                leverage = profile.max_leverage
                reason = f"💎 DIAMOND large cap - x{leverage}"
            elif confidence_score >= 55:
                leverage = profile.default_leverage
                reason = f"🥇 GOLD large cap - x{leverage}"
            else:
                leverage = profile.min_leverage
                reason = f"Standard large cap - x{leverage}"
            
            return leverage, reason
        
        elif asset_class == AssetClass.MEME:
            # Meme coins: Lower leverage due to high volatility
            if confidence_score >= 75:
                leverage = 20
                reason = f"⚠️ MEME coin DIAMOND - capped at x{leverage}"
            else:
                leverage = 10
                reason = f"⚠️ MEME coin - low leverage x{leverage}"
            
            return leverage, reason
        
        else:
            # Small/Mid cap: Volume-based leverage
            if volume_24h >= 200_000_000:  # >$200M
                base = 100
            elif volume_24h >= 50_000_000:  # >$50M
                base = 50
            else:
                base = 15
            
            # Adjust by confidence
            if confidence_score >= 75:
                leverage = base
                reason = f"💎 DIAMOND setup - x{leverage}"
            elif confidence_score >= 55:
                leverage = int(base * 0.7)
                reason = f"🥇 GOLD setup - x{leverage}"
            else:
                leverage = int(base * 0.5)
                reason = f"Standard setup - x{leverage}"
            
            return max(5, leverage), reason
    
    def get_sl_tp_multipliers(self, symbol: str) -> Tuple[float, float]:
        """
        Get SL/TP ATR multipliers for asset.
        
        Returns:
            Tuple of (sl_multiplier, tp_multiplier)
        """
        if symbol in self.precious_metals:
            profile = self.precious_metals[symbol]["leverage_profile"]
            return profile.sl_atr_multiplier, profile.tp_atr_multiplier
        
        if symbol in self.large_caps:
            profile = self.large_caps[symbol]["leverage_profile"]
            return profile.sl_atr_multiplier, profile.tp_atr_multiplier
        
        # Default for other assets
        return 1.5, 2.0
    
    def is_precious_metal(self, symbol: str) -> bool:
        """Check if symbol is a precious metal."""
        return symbol in self.precious_metals or "PAXG" in symbol or "XAU" in symbol
    
    def get_profile_info(self, symbol: str) -> Dict:
        """Get full profile info for logging."""
        asset_class = self.get_asset_class(symbol)
        
        info = {
            "symbol": symbol,
            "asset_class": asset_class.value,
            "is_precious_metal": self.is_precious_metal(symbol),
            "is_meme": symbol in self.meme_coins,
        }
        
        if symbol in self.precious_metals:
            config = self.precious_metals[symbol]
            info.update({
                "name": config["name"],
                "min_leverage": config["leverage_profile"].min_leverage,
                "max_leverage": config["leverage_profile"].max_leverage,
                "volatility_factor": config.get("volatility_factor", 1.0)
            })
        elif symbol in self.large_caps:
            config = self.large_caps[symbol]
            info.update({
                "name": config["name"],
                "min_leverage": config["leverage_profile"].min_leverage,
                "max_leverage": config["leverage_profile"].max_leverage,
            })
        
        return info


# Singleton instance
_leverage_manager: Optional[LeverageManager] = None


def get_leverage_manager() -> LeverageManager:
    """Get or create leverage manager singleton."""
    global _leverage_manager
    if _leverage_manager is None:
        _leverage_manager = LeverageManager()
    return _leverage_manager
