"""
Entry Calculator for IE Trade

Calculates:
- Entry price (50% of M5 FVG)
- Stop Loss (above/below MSS swing point)
- Take Profit 1 (Internal Liquidity - recent swing)
- Take Profit 2 (External Liquidity - H1/Daily swing)
- Risk:Reward ratio
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

from .config import IETradeConfig, DEFAULT_CONFIG
from .fvg_detector import FVG, FVGType, Candle
from .mss_detector import MSS, MSSType


logger = logging.getLogger(__name__)


@dataclass
class TradeSetup:
    """Complete trade setup with all parameters"""
    
    # Identity
    symbol: str
    direction: str  # "LONG" or "SHORT"
    timestamp: datetime
    
    # Entry
    entry_price: float
    
    # Risk Management
    stop_loss: float
    take_profit_1: float  # Internal Liquidity
    take_profit_2: float  # External Liquidity
    
    # R:R Analysis
    risk_pct: float      # Risk as % of entry
    reward_1_pct: float  # Reward to TP1 as % of entry
    reward_2_pct: float  # Reward to TP2 as % of entry
    rr_ratio_1: float    # Risk:Reward to TP1
    rr_ratio_2: float    # Risk:Reward to TP2
    
    # Entry type (with default)
    entry_type: str = "LIMIT"  # LIMIT or MARKET
    
    # Source data
    h1_fvg: Optional[FVG] = None
    m5_fvg: Optional[FVG] = None
    mss: Optional[MSS] = None
    
    # Session info
    kill_zone: str = ""
    daily_bias: str = ""
    
    # Validation
    is_valid: bool = True
    invalidation_reason: str = ""
    
    # Priority (for multi-setup filtering)
    priority_score: float = 0.0
    
    def __str__(self) -> str:
        return (f"Setup({self.direction} {self.symbol} "
                f"Entry={self.entry_price:.4f} SL={self.stop_loss:.4f} "
                f"TP1={self.take_profit_1:.4f} R:R={self.rr_ratio_1:.2f})")
    
    def to_alert_message(self) -> str:
        """Format setup for Telegram alert"""
        emoji = "🔴" if self.direction == "SHORT" else "🟢"
        
        msg = f"""
{emoji} **IE TRADE ALERT** {emoji}

📊 **{self.symbol}** - {self.direction}
⏰ Kill Zone: {self.kill_zone}
📈 Daily Bias: {self.daily_bias}

💰 **Entry:** ${self.entry_price:.4f} (Limit)
🛡 **Stop Loss:** ${self.stop_loss:.4f} ({self.risk_pct:.2f}%)
🎯 **TP1:** ${self.take_profit_1:.4f} (R:R {self.rr_ratio_1:.1f})
🎯 **TP2:** ${self.take_profit_2:.4f} (R:R {self.rr_ratio_2:.1f})

📐 **Analysis:**
• H1 FVG: ${self.h1_fvg.bottom:.4f} - ${self.h1_fvg.top:.4f}
• M5 MSS: {self.mss.displacement_body_ratio:.0%} displacement
• Zone: {self.h1_fvg.zone_type.value if self.h1_fvg.zone_type else 'N/A'}

⚠️ *Alert only - Manual entry required*
"""
        return msg.strip()
    
    def to_sheet_row(self) -> dict:
        """Format setup for Google Sheet logging"""
        return {
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'symbol': self.symbol,
            'direction': self.direction,
            'entry': self.entry_price,
            'sl': self.stop_loss,
            'tp1': self.take_profit_1,
            'tp2': self.take_profit_2,
            'rr': self.rr_ratio_1,
            'kill_zone': self.kill_zone,
            'daily_bias': self.daily_bias,
            'note': 'IE trade'
        }


class EntryCalculator:
    """
    Calculates optimal entry, SL, and TP levels
    
    Entry Strategy:
    - Entry at 50% of M5 FVG (Optimal Trade Entry)
    - SL above/below MSS swing point with buffer
    - TP1 at Internal Liquidity (recent swing)
    - TP2 at External Liquidity (H1/Daily swing)
    """
    
    def __init__(self, config: IETradeConfig = DEFAULT_CONFIG):
        self.config = config
    
    def calculate_setup(
        self,
        symbol: str,
        direction: str,
        current_price: float,
        h1_fvg: FVG,
        m5_fvg: Optional[FVG],
        mss: MSS,
        h1_candles: List[Candle],
        m5_candles: List[Candle],
        kill_zone: str = "",
        daily_bias: str = ""
    ) -> Optional[TradeSetup]:
        """
        Calculate complete trade setup
        
        Args:
            symbol: Trading pair
            direction: "LONG" or "SHORT"
            current_price: Current market price
            h1_fvg: H1 Fair Value Gap
            m5_fvg: M5 FVG from MSS (optional, use if available)
            mss: Market Structure Shift
            h1_candles: H1 candle data for TP calculation
            m5_candles: M5 candle data
            kill_zone: Current kill zone name
            daily_bias: Daily bias direction
            
        Returns:
            TradeSetup object or None if invalid
        """
        
        # 1. Calculate Entry Price
        if m5_fvg and m5_fvg.is_valid:
            # Use M5 FVG mid point (OTE)
            entry_price = m5_fvg.mid
        else:
            # Fallback: Use H1 FVG mid point
            entry_price = h1_fvg.mid
        
        # 2. Calculate Stop Loss
        stop_loss = self._calculate_stop_loss(mss, direction, m5_candles)
        if stop_loss is None:
            logger.warning(f"Could not calculate SL for {symbol}")
            return None
        
        # 3. Calculate Take Profits
        tp1, tp2 = self._calculate_take_profits(
            direction=direction,
            entry_price=entry_price,
            m5_candles=m5_candles,
            h1_candles=h1_candles
        )
        
        if tp1 is None:
            logger.warning(f"Could not calculate TP for {symbol}")
            return None
        
        # 4. Calculate Risk:Reward
        risk_pct, reward_1_pct, reward_2_pct, rr_1, rr_2 = self._calculate_rr(
            entry_price=entry_price,
            stop_loss=stop_loss,
            tp1=tp1,
            tp2=tp2,
            direction=direction
        )
        
        # 5. Validate minimum R:R
        if rr_1 < self.config.MIN_RR_RATIO:
            logger.info(f"{symbol}: R:R {rr_1:.2f} below minimum {self.config.MIN_RR_RATIO}")
            return TradeSetup(
                symbol=symbol,
                direction=direction,
                timestamp=datetime.utcnow(),
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit_1=tp1,
                take_profit_2=tp2,
                risk_pct=risk_pct,
                reward_1_pct=reward_1_pct,
                reward_2_pct=reward_2_pct,
                rr_ratio_1=rr_1,
                rr_ratio_2=rr_2,
                h1_fvg=h1_fvg,
                m5_fvg=m5_fvg,
                mss=mss,
                kill_zone=kill_zone,
                daily_bias=daily_bias,
                is_valid=False,
                invalidation_reason=f"R:R {rr_1:.2f} < min {self.config.MIN_RR_RATIO}"
            )
        
        # 6. Calculate priority score (for multi-setup filtering)
        priority_score = self._calculate_priority(
            symbol=symbol,
            rr_ratio=rr_1,
            displacement=mss.displacement_body_ratio
        )
        
        return TradeSetup(
            symbol=symbol,
            direction=direction,
            timestamp=datetime.utcnow(),
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_pct=risk_pct,
            reward_1_pct=reward_1_pct,
            reward_2_pct=reward_2_pct,
            rr_ratio_1=rr_1,
            rr_ratio_2=rr_2,
            h1_fvg=h1_fvg,
            m5_fvg=m5_fvg,
            mss=mss,
            kill_zone=kill_zone,
            daily_bias=daily_bias,
            is_valid=True,
            priority_score=priority_score
        )
    
    def _calculate_stop_loss(
        self,
        mss: MSS,
        direction: str,
        m5_candles: List[Candle]
    ) -> Optional[float]:
        """Calculate Stop Loss from MSS swing points"""
        
        buffer_pct = 0.001  # 0.1% buffer
        
        if direction == "SHORT":
            # SL above recent Swing High
            if mss.swing_high:
                buffer = mss.swing_high.price * buffer_pct
                return mss.swing_high.price + buffer
            else:
                # Fallback: highest high in recent candles
                recent = m5_candles[-20:] if len(m5_candles) >= 20 else m5_candles
                max_high = max(c.high for c in recent)
                return max_high * (1 + buffer_pct)
        
        elif direction == "LONG":
            # SL below recent Swing Low
            if mss.swing_low:
                buffer = mss.swing_low.price * buffer_pct
                return mss.swing_low.price - buffer
            else:
                # Fallback: lowest low in recent candles
                recent = m5_candles[-20:] if len(m5_candles) >= 20 else m5_candles
                min_low = min(c.low for c in recent)
                return min_low * (1 - buffer_pct)
        
        return None
    
    def _calculate_take_profits(
        self,
        direction: str,
        entry_price: float,
        m5_candles: List[Candle],
        h1_candles: List[Candle]
    ) -> Tuple[Optional[float], Optional[float]]:
        """
        Calculate Take Profit levels
        
        TP1: Internal Liquidity (recent swing on M5/H1)
        TP2: External Liquidity (larger swing on H1/Daily)
        """
        
        if direction == "SHORT":
            # TP1: Recent low (Internal Liquidity)
            m5_recent = m5_candles[-30:] if len(m5_candles) >= 30 else m5_candles
            tp1 = min(c.low for c in m5_recent)
            
            # TP2: H1 swing low (External Liquidity)
            h1_recent = h1_candles[-50:] if len(h1_candles) >= 50 else h1_candles
            tp2 = min(c.low for c in h1_recent)
            
            # Ensure TP1 < entry and TP2 < TP1
            if tp1 >= entry_price:
                tp1 = entry_price * 0.99  # 1% below entry
            if tp2 >= tp1:
                tp2 = tp1 * 0.98  # 2% below TP1
        
        elif direction == "LONG":
            # TP1: Recent high (Internal Liquidity)
            m5_recent = m5_candles[-30:] if len(m5_candles) >= 30 else m5_candles
            tp1 = max(c.high for c in m5_recent)
            
            # TP2: H1 swing high (External Liquidity)
            h1_recent = h1_candles[-50:] if len(h1_candles) >= 50 else h1_candles
            tp2 = max(c.high for c in h1_recent)
            
            # Ensure TP1 > entry and TP2 > TP1
            if tp1 <= entry_price:
                tp1 = entry_price * 1.01  # 1% above entry
            if tp2 <= tp1:
                tp2 = tp1 * 1.02  # 2% above TP1
        
        else:
            return None, None
        
        return tp1, tp2
    
    def _calculate_rr(
        self,
        entry_price: float,
        stop_loss: float,
        tp1: float,
        tp2: float,
        direction: str
    ) -> Tuple[float, float, float, float, float]:
        """
        Calculate Risk:Reward ratios
        
        Returns:
            Tuple of (risk_pct, reward_1_pct, reward_2_pct, rr_1, rr_2)
        """
        
        if direction == "SHORT":
            risk = stop_loss - entry_price
            reward_1 = entry_price - tp1
            reward_2 = entry_price - tp2
        else:
            risk = entry_price - stop_loss
            reward_1 = tp1 - entry_price
            reward_2 = tp2 - entry_price
        
        risk_pct = (risk / entry_price) * 100
        reward_1_pct = (reward_1 / entry_price) * 100
        reward_2_pct = (reward_2 / entry_price) * 100
        
        rr_1 = reward_1 / risk if risk > 0 else 0
        rr_2 = reward_2 / risk if risk > 0 else 0
        
        return abs(risk_pct), abs(reward_1_pct), abs(reward_2_pct), rr_1, rr_2
    
    def _calculate_priority(
        self,
        symbol: str,
        rr_ratio: float,
        displacement: float
    ) -> float:
        """
        Calculate priority score for multi-setup filtering
        
        Priority = Coin Priority * 100 + R:R * 10 + Displacement * 10
        
        Lower score = Higher priority
        BTC = 0, ETH = 1, others = 2
        """
        coin_priority = self.config.get_coin_priority(symbol)
        
        # Invert R:R and displacement so higher is better
        # But we want lower score = higher priority
        # So: coin_priority (lower better) - rr (higher better) - displacement (higher better)
        
        score = (coin_priority * 1000) - (rr_ratio * 10) - (displacement * 10)
        
        return score
    
    def filter_best_setup(self, setups: List[TradeSetup]) -> Optional[TradeSetup]:
        """
        Filter multiple setups to find the best one
        
        Priority order:
        1. BTC (if has setup)
        2. ETH (if no BTC)
        3. Best R:R among others
        """
        if not setups:
            return None
        
        valid_setups = [s for s in setups if s.is_valid]
        if not valid_setups:
            return None
        
        # Check for BTC
        btc_setup = next((s for s in valid_setups if s.symbol == "BTC-USDT"), None)
        if btc_setup:
            return btc_setup
        
        # Check for ETH
        eth_setup = next((s for s in valid_setups if s.symbol == "ETH-USDT"), None)
        if eth_setup:
            return eth_setup
        
        # Sort by R:R (higher is better)
        valid_setups.sort(key=lambda x: x.rr_ratio_1, reverse=True)
        
        return valid_setups[0]
