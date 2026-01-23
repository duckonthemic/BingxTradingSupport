"""
Trade Filter Module
Implements:
- BTC Mood Check: Cancel Long altcoin if BTC dumping
- MTF Trend Filter: Only trade with higher timeframe trend
- Leverage-based Entry/TP/SL optimization
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple, List
from enum import Enum

logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class FilterResult(Enum):
    PASS = "PASS"
    REJECT_BTC_DUMP = "REJECT_BTC_DUMP"
    REJECT_MTF_TREND = "REJECT_MTF_TREND"
    REJECT_LOW_RR = "REJECT_LOW_RR"


@dataclass
class LeverageConfig:
    """Leverage configuration per asset class."""
    # Altcoins: 10-20x
    altcoin_leverage: int = 15
    # Major coins (BTC, ETH, SOL): 100x
    major_leverage: int = 100
    # Gold: 500x
    gold_leverage: int = 500
    
    # Position size - Grade-based (Phase 1)
    position_size_grade_a: float = 2.0  # $2 for Grade A/Diamond (full)
    position_size_grade_b: float = 1.0  # $1 for Grade B/Gold (half)
    position_size_usd: float = 2.0      # Legacy default
    
    # Major coins list
    major_coins: tuple = ("BTC-USDT", "ETH-USDT", "SOL-USDT")
    gold_symbols: tuple = ("XAUT-USDT", "PAXG-USDT")


@dataclass 
class OptimizedLevels:
    """Optimized Entry/SL/TP levels based on leverage."""
    entry: float
    stop_loss: float
    take_profit_1: float  # 2R
    take_profit_2: float  # 4R
    take_profit_3: float  # Target
    risk_usd: float
    reward_usd: float
    risk_reward: float
    leverage: int
    position_size: float  # Actual position with leverage
    liquidation_price: float


class TradeFilter:
    """
    Filters trades based on:
    1. BTC Mood Check - reject Long if BTC dumping
    2. MTF Trend Filter - align with H1 trend
    3. Range Market Detection - allow mean-reversion in sideways
    """
    
    def __init__(self):
        self.btc_dump_threshold = -0.5  # -0.5% in 15 minutes
        self.leverage_config = LeverageConfig()
        self.allow_range_trading = True  # Allow trades in sideways market
        self.adx_range_threshold = 20  # ADX below this = ranging
        
        # Track BTC state
        self._btc_price_15m_ago: Optional[float] = None
        self._btc_current_price: Optional[float] = None
        self._btc_change_pct: float = 0.0
        self._btc_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
        self._btc_ema_distance_pct: float = 0.0  # % from EMA89
    
    def update_btc_trend(self, trend: str, ema_distance_pct: float = 0.0):
        """Update BTC overall trend from context manager."""
        self._btc_trend = trend
        self._btc_ema_distance_pct = ema_distance_pct
    
    def update_btc_state(
        self, 
        current_price: float, 
        price_15m_ago: Optional[float] = None,
        change_pct: Optional[float] = None
    ):
        """Update BTC price state."""
        self._btc_current_price = current_price
        if price_15m_ago:
            self._btc_price_15m_ago = price_15m_ago
            self._btc_change_pct = ((current_price - price_15m_ago) / price_15m_ago) * 100
        elif change_pct is not None:
            self._btc_change_pct = change_pct
    
    def is_btc_dumping(self) -> bool:
        """Check if BTC is in dump mode."""
        return self._btc_change_pct < self.btc_dump_threshold
    
    def is_ranging_market(self, adx: float) -> bool:
        """Check if market is ranging (sideways)."""
        return adx < self.adx_range_threshold
    
    def check_btc_mood(self, direction: TradeDirection, symbol: str) -> Tuple[bool, str]:
        """
        Check BTC mood before allowing trade.
        
        Rules:
        - If BTC dumping (>0.5% drop in 15m) -> BLOCK ALL LONG on altcoins
        - BTC/ETH can still Long (they follow their own trend)
        """
        # Major coins can trade their own direction
        if symbol in self.leverage_config.major_coins:
            return True, "Major coin - own trend"
        
        # Altcoins: Block Long if BTC dumping
        if direction == TradeDirection.LONG and self.is_btc_dumping():
            return False, f"BTC dumping {self._btc_change_pct:.2f}% - LONG blocked"
        
        return True, "BTC mood OK"
    
    def check_mtf_trend(
        self,
        direction: TradeDirection,
        current_price: float,
        ema89_h1: float,
        ema89_h4: Optional[float] = None,
        adx: Optional[float] = None,
        is_range_trade: bool = False
    ) -> Tuple[bool, str]:
        """
        Check Multi-Timeframe trend alignment.
        
        Rules:
        - LONG: Price must be ABOVE EMA89 on H1
        - SHORT: Price must be BELOW EMA89 on H1
        - Range trades: Allowed if ADX < 20 (sideways market)
        """
        # Allow range trades in sideways market
        if is_range_trade and adx and self.is_ranging_market(adx):
            return True, f"Range trade allowed (ADX={adx:.1f} < {self.adx_range_threshold})"
        
        if direction == TradeDirection.LONG:
            if current_price > ema89_h1:
                return True, f"Price ${current_price:.4f} > EMA89 H1 ${ema89_h1:.4f}"
            else:
                # Allow if within 5% of EMA (relaxed for sideways/scalping)
                distance_pct = ((ema89_h1 - current_price) / current_price) * 100
                if distance_pct < 5.0 and self.allow_range_trading:
                    return True, f"Scalp zone - {distance_pct:.2f}% from EMA89"
                return False, f"Price ${current_price:.4f} < EMA89 H1 ${ema89_h1:.4f} - LONG blocked"
        
        elif direction == TradeDirection.SHORT:
            if current_price < ema89_h1:
                return True, f"Price ${current_price:.4f} < EMA89 H1 ${ema89_h1:.4f}"
            else:
                # Allow if within 10% of EMA (relaxed for bearish market)
                distance_pct = ((current_price - ema89_h1) / current_price) * 100
                if distance_pct < 10.0:  # Increased from 5% to 10%
                    return True, f"Near EMA zone - {distance_pct:.2f}% from EMA89 - SHORT allowed"
                # Also allow SHORT when BTC trend is bearish
                if self._btc_trend == "BEARISH":
                    return True, f"BTC BEARISH trend ({self._btc_ema_distance_pct:.1f}% from EMA89) - SHORT allowed"
                return False, f"Price ${current_price:.4f} > EMA89 H1 ${ema89_h1:.4f} - SHORT blocked"
        
        return True, "No filter"
    
    def get_leverage(self, symbol: str) -> int:
        """Get appropriate leverage for symbol."""
        if symbol in self.leverage_config.gold_symbols:
            return self.leverage_config.gold_leverage
        elif symbol in self.leverage_config.major_coins:
            return self.leverage_config.major_leverage
        else:
            return self.leverage_config.altcoin_leverage
    
    def get_position_size(self, tier: str = "DIAMOND") -> float:
        """
        Get position size based on signal tier/grade.
        
        Phase 1 Implementation:
        - DIAMOND / Grade A: Full position ($2)
        - GOLD / Grade B: Half position ($1)
        - SILVER / below: No trade
        """
        if tier.upper() in ["DIAMOND", "A", "A_SNIPER"]:
            return self.leverage_config.position_size_grade_a  # $2
        elif tier.upper() in ["GOLD", "B", "B_SCALP"]:
            return self.leverage_config.position_size_grade_b  # $1
        else:
            return 0.0  # No trade for Silver/Reject
    
    def calculate_optimized_levels(
        self,
        symbol: str,
        direction: TradeDirection,
        entry_price: float,
        atr: float,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None
    ) -> OptimizedLevels:
        """
        Calculate optimized Entry/SL/TP based on leverage.
        
        Strategy:
        - High leverage = tighter SL (smaller % move = same $ risk)
        - Focus on high R:R (minimum 1:4)
        - Use ATR for SL distance
        """
        leverage = self.get_leverage(symbol)
        position_size_usd = self.leverage_config.position_size_usd
        
        # Actual position value with leverage
        position_value = position_size_usd * leverage
        
        # Calculate SL distance based on leverage
        # Higher leverage = tighter stop (in % terms)
        # Formula: max_loss_% = (position_size / position_value) * 100
        # With $2 and 15x, max 5% move before losing all
        # We want to risk 50% of position ($1), so SL at 2.5% for 15x
        
        if leverage >= 100:
            # High leverage (BTC, ETH, SOL, Gold): moderate stop
            sl_multiplier = 1.5  # 1.5 ATR - wider for volatility
        elif leverage >= 50:
            sl_multiplier = 1.8  # 1.8 ATR
        else:
            sl_multiplier = 2.0  # 2.0 ATR for altcoins - need more buffer
        
        sl_distance = atr * sl_multiplier
        
        # Ensure SL is not too tight (min 0.5% for high leverage, 2% for altcoins)
        min_sl_pct = 0.005 if leverage >= 100 else 0.02
        min_sl_distance = entry_price * min_sl_pct
        sl_distance = max(sl_distance, min_sl_distance)
        
        if direction == TradeDirection.LONG:
            stop_loss = entry_price - sl_distance
            risk = entry_price - stop_loss
            
            # TPs with high R:R
            tp1 = entry_price + (risk * 2)   # 2R
            tp2 = entry_price + (risk * 4)   # 4R  
            tp3 = entry_price + (risk * 6)   # 6R or swing target
            
            # Use swing high as alternative TP3 if closer
            if swing_high and swing_high > entry_price:
                tp3 = max(tp3, swing_high)
            
            # Calculate liquidation price (simplified)
            # Liquidation when loss = position value
            # loss_at_liq = position_value, so price_drop = position_value / leverage
            liq_distance = entry_price * (1 / leverage) * 0.9  # 90% of theoretical
            liquidation_price = entry_price - liq_distance
            
        else:  # SHORT
            stop_loss = entry_price + sl_distance
            risk = stop_loss - entry_price
            
            tp1 = entry_price - (risk * 2)   # 2R
            tp2 = entry_price - (risk * 4)   # 4R
            tp3 = entry_price - (risk * 6)   # 6R or swing target
            
            if swing_low and swing_low < entry_price:
                tp3 = min(tp3, swing_low)
            
            liq_distance = entry_price * (1 / leverage) * 0.9
            liquidation_price = entry_price + liq_distance
        
        # Calculate risk/reward in USD
        # Risk: if hit SL, lose (sl_distance / entry_price) * position_value
        risk_pct = sl_distance / entry_price
        risk_usd = risk_pct * position_value
        
        # Reward at TP2 (4R)
        reward_pct = (abs(tp2 - entry_price) / entry_price)
        reward_usd = reward_pct * position_value
        
        risk_reward = reward_usd / risk_usd if risk_usd > 0 else 0
        
        return OptimizedLevels(
            entry=round(entry_price, 8),
            stop_loss=round(stop_loss, 8),
            take_profit_1=round(tp1, 8),
            take_profit_2=round(tp2, 8),
            take_profit_3=round(tp3, 8),
            risk_usd=round(risk_usd, 2),
            reward_usd=round(reward_usd, 2),
            risk_reward=round(risk_reward, 1),
            leverage=leverage,
            position_size=round(position_value, 2),
            liquidation_price=round(liquidation_price, 8)
        )
    
    def filter_trade(
        self,
        symbol: str,
        direction: TradeDirection,
        current_price: float,
        ema89_h1: float,
        atr: float,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        is_mean_reversion: bool = False
    ) -> Tuple[FilterResult, str, Optional[OptimizedLevels]]:
        """
        Full trade filter with all checks.
        
        Args:
            is_mean_reversion: If True, skip MTF trend filter (for BB_BOUNCE, etc.)
        
        Returns:
            (result, reason, levels or None)
        """
        # 1. BTC Mood Check
        btc_ok, btc_reason = self.check_btc_mood(direction, symbol)
        if not btc_ok:
            return FilterResult.REJECT_BTC_DUMP, btc_reason, None
        
        # 2. MTF Trend Filter - SKIP for mean reversion strategies
        if not is_mean_reversion:
            mtf_ok, mtf_reason = self.check_mtf_trend(direction, current_price, ema89_h1)
            if not mtf_ok:
                return FilterResult.REJECT_MTF_TREND, mtf_reason, None
        
        # 3. Calculate optimized levels
        levels = self.calculate_optimized_levels(
            symbol=symbol,
            direction=direction,
            entry_price=current_price,
            atr=atr,
            swing_high=swing_high,
            swing_low=swing_low
        )
        
        # 4. Check minimum R:R (lowered for scalping)
        if levels.risk_reward < 1.2:
            return FilterResult.REJECT_LOW_RR, f"R:R {levels.risk_reward:.1f} < 1.2", None
        
        return FilterResult.PASS, "All filters passed", levels


class SimplifiedIndicators:
    """
    Simplified indicator set - focus on Price Action + WaveTrend.
    Removed: CCI, Parabolic SAR, TD Sequential
    """
    
    @staticmethod
    def get_required_indicators() -> List[str]:
        """Get list of required indicators."""
        return [
            "price",
            "ema34_h1",
            "ema89_h1", 
            "ema89_h4",
            "rsi_h1",
            "mfi",
            "volume_ratio",
            "atr",
            "wt1",
            "wt2",
            "wt_signal",
            "swing_high_20",
            "swing_low_20",
            "macd_trend",
            "adx",
            "bb_upper",
            "bb_lower",
        ]
    
    @staticmethod
    def get_removed_indicators() -> List[str]:
        """Indicators that were removed."""
        return [
            "cci",           # Too noisy
            "psar",          # Lagging
            "psar_direction",
            "td_count",      # Too complex
            "td_direction",
            "stoch_rsi_k",   # Redundant with RSI
            "stoch_rsi_d",
        ]


# ==================== Test ====================

def test_trade_filter():
    """Test trade filter."""
    print("=" * 60)
    print("Testing Trade Filter")
    print("=" * 60)
    
    tf = TradeFilter()
    
    # Simulate BTC state
    tf.update_btc_state(current_price=92000, price_15m_ago=92500)
    print(f"\n📊 BTC Change: {tf._btc_change_pct:.2f}%")
    print(f"   Is Dumping: {tf.is_btc_dumping()}")
    
    # Test filter for altcoin LONG (should be blocked)
    result, reason, levels = tf.filter_trade(
        symbol="DOGE-USDT",
        direction=TradeDirection.LONG,
        current_price=0.127,
        ema89_h1=0.125,
        atr=0.005
    )
    print(f"\n🐕 DOGE LONG:")
    print(f"   Result: {result.value}")
    print(f"   Reason: {reason}")
    
    # Test filter for BTC LONG (should pass, major coin)
    tf.update_btc_state(current_price=92500, price_15m_ago=92000)  # BTC recovering
    result, reason, levels = tf.filter_trade(
        symbol="BTC-USDT",
        direction=TradeDirection.LONG,
        current_price=92500,
        ema89_h1=91000,
        atr=1500,
        swing_high=95000
    )
    print(f"\n₿ BTC LONG:")
    print(f"   Result: {result.value}")
    print(f"   Reason: {reason}")
    if levels:
        print(f"   Entry: ${levels.entry:,.2f}")
        print(f"   SL: ${levels.stop_loss:,.2f}")
        print(f"   TP1: ${levels.take_profit_1:,.2f} (2R)")
        print(f"   TP2: ${levels.take_profit_2:,.2f} (4R)")
        print(f"   Leverage: {levels.leverage}x")
        print(f"   Position: ${levels.position_size:,.2f}")
        print(f"   Risk: ${levels.risk_usd:.2f}")
        print(f"   Reward: ${levels.reward_usd:.2f}")
        print(f"   R:R: 1:{levels.risk_reward}")
        print(f"   Liquidation: ${levels.liquidation_price:,.2f}")
    
    # Test SHORT rejected by MTF
    result, reason, levels = tf.filter_trade(
        symbol="ETH-USDT",
        direction=TradeDirection.SHORT,
        current_price=3200,
        ema89_h1=3100,  # Price above EMA = no short
        atr=100
    )
    print(f"\n🔷 ETH SHORT:")
    print(f"   Result: {result.value}")
    print(f"   Reason: {reason}")
    
    print("\n✅ Trade Filter test complete!")


if __name__ == "__main__":
    test_trade_filter()
