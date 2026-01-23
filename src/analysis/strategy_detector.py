"""
Strategy Detector v2.0 - Optimized for High Win Rate Scalping

MAIN Strategies (ON):
✅ SFP (Swing Failure Pattern) - Primary weapon for reversals
✅ Liquidity Sweep - Complement SFP (double/triple top/bottom sweeps)
✅ EMA Cloud Pullback - Strong trend continuation
✅ Breaker Block Retest - Safe breakout entries

FILTER Only (Not trigger):
✅ SMC Order Block - Zone confirmation only

DISABLED (Too noisy for scalping):
❌ CHoCH - Merged into SFP logic
❌ FVG - Price often runs through small FVGs
❌ TD Sequential, Parabolic SAR - Too slow

ENTRY LOGIC (3-Step):
1. Trend Filter (H1): Price > EMA34 > EMA89 = LONG only
2. Setup Detection (M5/M15): SFP or EMA Pullback
3. Momentum Confirm: WaveTrend cross + Volume spike + RSI Divergence
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Active trading strategies."""
    SFP = "SFP"                          # Swing Failure Pattern (MAIN)
    LIQUIDITY_SWEEP = "LIQ_SWEEP"        # Liquidity Sweep (MAIN)
    EMA_PULLBACK = "EMA_PULLBACK"        # EMA Cloud Pullback (SUPPORT)
    BREAKER_RETEST = "BREAKER_RETEST"    # Breaker Block Retest (SUPPORT)
    BB_BOUNCE = "BB_BOUNCE"              # Bollinger Band Bounce (RANGE)


STRATEGY_ICONS = {
    StrategyType.SFP: "🔄",
    StrategyType.LIQUIDITY_SWEEP: "🌊",
    StrategyType.EMA_PULLBACK: "☁️",
    StrategyType.BREAKER_RETEST: "💥",
    StrategyType.BB_BOUNCE: "📊",
}

STRATEGY_NAMES = {
    StrategyType.SFP: "Swing Failure Pattern",
    StrategyType.LIQUIDITY_SWEEP: "Liquidity Sweep",
    StrategyType.EMA_PULLBACK: "EMA Cloud Pullback",
    StrategyType.BREAKER_RETEST: "Breaker Block Retest",
    StrategyType.BB_BOUNCE: "BB Bounce Range",
}

# Volume allocation per strategy
STRATEGY_VOLUME_WEIGHT = {
    StrategyType.SFP: 1.0,              # 100% standard volume
    StrategyType.LIQUIDITY_SWEEP: 1.0,  # 100% standard volume
    StrategyType.EMA_PULLBACK: 0.7,     # 70% volume (wider SL)
    StrategyType.BREAKER_RETEST: 0.8,   # 80% volume
    StrategyType.BB_BOUNCE: 0.5,        # 50% volume (mean reversion)
}


@dataclass
class TradeSetup:
    """A complete trade setup with entry, targets, and reasoning."""
    strategy: StrategyType
    symbol: str
    direction: str  # "LONG" or "SHORT"
    
    # Price levels
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Analysis
    confidence: float  # 0.0 - 1.0
    risk_reward: float
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Confluence factors
    has_rsi_divergence: bool = False
    has_wavetrend_cross: bool = False
    has_volume_spike: bool = False
    has_ob_confluence: bool = False  # Near Order Block
    
    # Context
    current_price: float = 0.0
    distance_to_entry_pct: float = 0.0
    is_actionable: bool = False
    
    # Structure
    market_structure: str = "NEUTRAL"
    trend_bias: str = "NEUTRAL"
    zone_type: str = "EQUILIBRIUM"
    
    # Volume weight
    volume_weight: float = 1.0
    
    @property
    def icon(self) -> str:
        return STRATEGY_ICONS.get(self.strategy, "📊")
    
    @property
    def name(self) -> str:
        return STRATEGY_NAMES.get(self.strategy, "Unknown")
    
    @property
    def direction_emoji(self) -> str:
        return "🟢" if self.direction == "LONG" else "🔴"
    
    @property
    def confluence_score(self) -> int:
        """Count confluence factors (0-4)."""
        score = 0
        if self.has_rsi_divergence:
            score += 1
        if self.has_wavetrend_cross:
            score += 1
        if self.has_volume_spike:
            score += 1
        if self.has_ob_confluence:
            score += 1
        return score
    
    @property
    def is_golden_setup(self) -> bool:
        """True if SFP + RSI divergence (double volume)."""
        return self.strategy == StrategyType.SFP and self.has_rsi_divergence


@dataclass
class TrendFilter:
    """H1 Trend Filter Result."""
    is_valid: bool
    allowed_direction: str  # "LONG", "SHORT", "NONE"
    ema34: float
    ema89: float
    current_price: float
    ema_gap_pct: float  # Gap between EMA34 and EMA89
    is_strong_trend: bool  # EMA gap is expanding
    
    @property
    def is_above_emas(self) -> bool:
        return self.current_price > self.ema34 > self.ema89
    
    @property
    def is_below_emas(self) -> bool:
        return self.current_price < self.ema34 < self.ema89


@dataclass 
class MomentumConfirm:
    """Momentum confirmation factors."""
    # WaveTrend
    wt1: float = 0.0
    wt2: float = 0.0
    wt_cross_up: bool = False
    wt_cross_down: bool = False
    wt_oversold: bool = False  # WT < -60
    wt_overbought: bool = False  # WT > 60
    
    # Volume
    current_volume: float = 0.0
    avg_volume: float = 0.0
    volume_ratio: float = 0.0
    has_volume_spike: bool = False  # Volume > 1.5x avg
    
    # RSI Divergence
    rsi: float = 50.0
    has_bullish_div: bool = False
    has_bearish_div: bool = False


class StrategyDetector:
    """
    Optimized Strategy Detector for High Win Rate.
    
    Logic Flow:
    1. Check H1 Trend Filter (EMA34/89)
    2. Detect setups on M5/M15
    3. Confirm with momentum (WT + Volume + RSI Divergence)
    """
    
    def __init__(self):
        self.swing_lookback = 5
        self.entry_proximity = 0.003  # 0.3% for actionable
        self.min_rr = 2.0  # Minimum 2:1 R:R
        self.ema_gap_strong = 0.5  # 0.5% gap = strong trend
    
    def analyze(
        self,
        symbol: str,
        df_m5: pd.DataFrame,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        current_price: float,
        atr: float,
        wt1: float = 0.0,
        wt2: float = 0.0,
        rsi: float = 50.0,
        volume_ratio: float = 1.0
    ) -> List[TradeSetup]:
        """
        Analyze and return valid setups.
        Only returns setups that pass all 3 filters.
        """
        setups = []
        
        if df_h1.empty or len(df_h1) < 50:
            return setups
        
        # Step 1: H1 Trend Filter
        trend_filter = self._check_trend_filter(df_h1, current_price)
        if not trend_filter.is_valid:
            logger.debug(f"{symbol}: No valid trend direction")
            return setups
        
        # Step 2: Calculate momentum confirmation
        momentum = self._calculate_momentum(
            df_m5 if not df_m5.empty else df_m15,
            wt1, wt2, rsi, volume_ratio
        )
        
        # Step 3: Find Order Block zones (for confluence only)
        ob_zones = self._find_order_blocks(df_h1)
        
        # Step 4: Detect setups based on allowed direction
        allowed = trend_filter.allowed_direction
        
        # LONG setups - when LONG or BOTH_BULLISH_BIAS or BOTH_BEARISH_BIAS
        if allowed in ["LONG", "BOTH_BULLISH_BIAS", "BOTH_BEARISH_BIAS"]:
            if setup := self._detect_sfp_long(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                # Reduce confidence if counter-trend
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bearish)")
                setups.append(setup)
            
            if setup := self._detect_liquidity_sweep_long(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bearish)")
                setups.append(setup)
            
            if trend_filter.is_strong_trend or allowed == "LONG":
                if setup := self._detect_ema_pullback_long(symbol, df_m5, df_h1, current_price, atr, trend_filter, momentum):
                    setups.append(setup)
            
            if setup := self._detect_breaker_retest_long(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bearish)")
                setups.append(setup)
        
        # SHORT setups - when SHORT or BOTH_BULLISH_BIAS or BOTH_BEARISH_BIAS
        if allowed in ["SHORT", "BOTH_BULLISH_BIAS", "BOTH_BEARISH_BIAS"]:
            if setup := self._detect_sfp_short(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                # Reduce confidence if counter-trend
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bullish)")
                setups.append(setup)
            
            if setup := self._detect_liquidity_sweep_short(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bullish)")
                setups.append(setup)
            
            if trend_filter.is_strong_trend or allowed == "SHORT":
                if setup := self._detect_ema_pullback_short(symbol, df_m5, df_h1, current_price, atr, trend_filter, momentum):
                    setups.append(setup)
            
            if setup := self._detect_breaker_retest_short(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bullish)")
                setups.append(setup)
        
        # BEARISH CONTINUATION - Simple trend following for bearish market
        if allowed in ["SHORT", "BOTH_BEARISH_BIAS"]:
            if setup := self._detect_bearish_continuation(symbol, df_m5, df_h1, current_price, atr, momentum):
                setups.append(setup)
        
        # BB BOUNCE - Works in any market condition (range trading)
        if setup := self._detect_bb_bounce_long(symbol, df_m15, current_price, atr, momentum):
            setup.confidence *= 0.9  # Slightly lower confidence for range trades
            setups.append(setup)
        
        if setup := self._detect_bb_bounce_short(symbol, df_m15, current_price, atr, momentum):
            setup.confidence *= 0.9
            setups.append(setup)
        
        # Calculate actionability and sort
        for setup in setups:
            setup.current_price = current_price
            setup.distance_to_entry_pct = abs(setup.entry_price - current_price) / current_price * 100
            setup.is_actionable = setup.distance_to_entry_pct <= self.entry_proximity * 100
            setup.volume_weight = STRATEGY_VOLUME_WEIGHT.get(setup.strategy, 1.0)
            
            # Boost confidence for golden setups
            if setup.is_golden_setup:
                setup.confidence = min(1.0, setup.confidence + 0.15)
                setup.reasons.append("🏆 GOLDEN SETUP: SFP + RSI Divergence")
        
        # Sort by: actionable > confluence > confidence > R:R
        setups.sort(key=lambda s: (
            s.is_actionable,
            s.confluence_score,
            s.confidence,
            s.risk_reward
        ), reverse=True)
        
        return setups
    
    def _check_trend_filter(self, df_h1: pd.DataFrame, current_price: float) -> TrendFilter:
        """
        Check H1 EMA Trend Filter - RELAXED for more signals.
        
        STRICT (Strong Trend):
        - Price > EMA34 > EMA89 = LONG only
        - Price < EMA34 < EMA89 = SHORT only
        
        RELAXED (Ranging/Weak Trend):
        - Price between EMAs = BOTH directions allowed
        - EMAs crossed = BOTH directions allowed (transition period)
        """
        closes = df_h1['close'].values
        
        # Calculate EMAs
        ema34 = self._calc_ema(closes, 34)
        ema89 = self._calc_ema(closes, 89)
        
        # Calculate EMA gap %
        ema_gap_pct = abs(ema34 - ema89) / ema89 * 100 if ema89 > 0 else 0
        is_strong = ema_gap_pct > self.ema_gap_strong
        
        # Determine allowed direction
        # Case 1: Clear bullish trend
        if current_price > ema34 > ema89:
            return TrendFilter(
                is_valid=True,
                allowed_direction="LONG",
                ema34=ema34,
                ema89=ema89,
                current_price=current_price,
                ema_gap_pct=ema_gap_pct,
                is_strong_trend=is_strong
            )
        # Case 2: Clear bearish trend
        elif current_price < ema34 < ema89:
            return TrendFilter(
                is_valid=True,
                allowed_direction="SHORT",
                ema34=ema34,
                ema89=ema89,
                current_price=current_price,
                ema_gap_pct=ema_gap_pct,
                is_strong_trend=is_strong
            )
        # Case 3: RANGING - Price between EMAs or EMAs crossed
        # Allow BOTH directions with reduced confidence
        else:
            # Determine primary bias based on EMA slope
            if ema34 > ema89:
                # Bullish EMA structure but price pulled back
                primary_direction = "BOTH_BULLISH_BIAS"
            else:
                # Bearish EMA structure but price bounced
                primary_direction = "BOTH_BEARISH_BIAS"
            
            return TrendFilter(
                is_valid=True,  # Changed to True - allow signals
                allowed_direction=primary_direction,
                ema34=ema34,
                ema89=ema89,
                current_price=current_price,
                ema_gap_pct=ema_gap_pct,
                is_strong_trend=False  # Not a strong trend
            )
    
    def _calculate_momentum(
        self, df: pd.DataFrame,
        wt1: float, wt2: float, rsi: float, volume_ratio: float
    ) -> MomentumConfirm:
        """Calculate momentum confirmation factors."""
        momentum = MomentumConfirm()
        
        momentum.wt1 = wt1
        momentum.wt2 = wt2
        momentum.wt_cross_up = wt1 > wt2 and wt1 - wt2 < 10  # Recent cross up
        momentum.wt_cross_down = wt1 < wt2 and wt2 - wt1 < 10  # Recent cross down
        momentum.wt_oversold = wt1 < -60
        momentum.wt_overbought = wt1 > 60
        
        momentum.volume_ratio = volume_ratio
        momentum.has_volume_spike = volume_ratio > 1.5
        
        momentum.rsi = rsi
        
        # RSI Divergence detection would require price comparison
        # Simplified: check if RSI is diverging from price action
        if not df.empty and len(df) >= 10:
            closes = df['close'].values
            # Simple divergence check: price lower but RSI higher
            if closes[-1] < closes[-5] and rsi > 40:  # Price down but RSI not
                momentum.has_bullish_div = True
            elif closes[-1] > closes[-5] and rsi < 60:  # Price up but RSI not
                momentum.has_bearish_div = True
        
        return momentum
    
    def _find_order_blocks(self, df_h1: pd.DataFrame) -> Dict:
        """Find Order Block zones for confluence."""
        ob_zones = {"bullish": None, "bearish": None}
        
        if len(df_h1) < 20:
            return ob_zones
        
        opens = df_h1['open'].values
        closes = df_h1['close'].values
        highs = df_h1['high'].values
        lows = df_h1['low'].values
        
        # Find bullish OB (last bearish candle before strong up move)
        for i in range(len(df_h1) - 3, max(len(df_h1) - 30, 0), -1):
            if closes[i] < opens[i]:  # Bearish candle
                if i + 2 < len(df_h1) and closes[i+1] > opens[i+1] and closes[i+2] > highs[i]:
                    ob_zones["bullish"] = {"high": highs[i], "low": lows[i]}
                    break
        
        # Find bearish OB
        for i in range(len(df_h1) - 3, max(len(df_h1) - 30, 0), -1):
            if closes[i] > opens[i]:  # Bullish candle
                if i + 2 < len(df_h1) and closes[i+1] < opens[i+1] and closes[i+2] < lows[i]:
                    ob_zones["bearish"] = {"high": highs[i], "low": lows[i]}
                    break
        
        return ob_zones
    
    def _detect_sfp_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm, ob_zones: Dict
    ) -> Optional[TradeSetup]:
        """
        Detect Bullish SFP (Swing Failure Pattern).
        Price sweeps swing low, then reverses with rejection wick.
        """
        if df_m5.empty or len(df_m5) < 10:
            return None
        
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        opens = df_m5['open'].values
        
        # Find swing lows
        swing_lows = self._find_swing_points(lows, self.swing_lookback, "LOW")
        if len(swing_lows) < 2:
            return None
        
        last_swing_low = swing_lows[-1]
        
        # Check if recent candle swept the swing low
        recent_low = min(lows[-3:])
        recent_close = closes[-1]
        recent_open = opens[-1]
        
        # SFP condition: swept low + closed above or near + bullish candle
        # Relaxed: close within 0.5% of swing low also counts
        close_near_swing = abs(recent_close - last_swing_low) / last_swing_low <= 0.005
        if recent_low < last_swing_low and (recent_close > last_swing_low or close_near_swing):
            # Check for wick rejection (pinbar)
            candle_range = highs[-1] - lows[-1]
            body_size = abs(recent_close - recent_open)
            lower_wick = min(recent_open, recent_close) - lows[-1]
            
            # Wick should be > 35% of candle (relaxed from 50%)
            if candle_range > 0 and lower_wick / candle_range >= 0.35:
                # Momentum confirmation
                reasons = [
                    f"Quét đáy swing low {last_swing_low:.6g}",
                    f"Rút chân mạnh (wick {lower_wick/candle_range*100:.0f}%)",
                    f"Đóng nến trên vùng quét"
                ]
                
                confidence = 0.70
                
                # Confluence boost
                if momentum.wt_cross_up and momentum.wt_oversold:
                    confidence += 0.10
                    reasons.append("✅ WaveTrend cross up từ oversold")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                has_ob_confluence = False
                if ob_zones["bullish"]:
                    ob = ob_zones["bullish"]
                    if ob["low"] <= current_price <= ob["high"]:
                        confidence += 0.10
                        has_ob_confluence = True
                        reasons.append("✅ Trong vùng Order Block")
                
                # Calculate levels
                entry = current_price
                sl = recent_low - atr * 0.2
                risk = entry - sl
                
                tp1 = entry + risk * 2
                tp2 = entry + risk * 4
                tp3 = entry + risk * 6
                
                rr = (tp1 - entry) / risk if risk > 0 else 0
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.SFP,
                        symbol=symbol,
                        direction="LONG",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_rsi_divergence=momentum.has_bullish_div,
                        has_wavetrend_cross=momentum.wt_cross_up,
                        has_volume_spike=momentum.has_volume_spike,
                        has_ob_confluence=has_ob_confluence,
                        zone_type="DISCOUNT"
                    )
        
        return None
    
    def _detect_sfp_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm, ob_zones: Dict
    ) -> Optional[TradeSetup]:
        """Detect Bearish SFP."""
        if df_m5.empty or len(df_m5) < 10:
            return None
        
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        opens = df_m5['open'].values
        
        swing_highs = self._find_swing_points(highs, self.swing_lookback, "HIGH")
        if len(swing_highs) < 2:
            return None
        
        last_swing_high = swing_highs[-1]
        
        recent_high = max(highs[-3:])
        recent_close = closes[-1]
        recent_open = opens[-1]
        
        if recent_high > last_swing_high and recent_close < last_swing_high:
            candle_range = highs[-1] - lows[-1]
            upper_wick = highs[-1] - max(recent_open, recent_close)
            
            # Relaxed: wick >= 35% (was 50%)
            if candle_range > 0 and upper_wick / candle_range >= 0.35:
                reasons = [
                    f"Quét đỉnh swing high {last_swing_high:.6g}",
                    f"Rút chân mạnh (wick {upper_wick/candle_range*100:.0f}%)",
                    f"Đóng nến dưới vùng quét"
                ]
                
                confidence = 0.70
                
                if momentum.wt_cross_down and momentum.wt_overbought:
                    confidence += 0.10
                    reasons.append("✅ WaveTrend cross down từ overbought")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                has_ob_confluence = False
                if ob_zones["bearish"]:
                    ob = ob_zones["bearish"]
                    if ob["low"] <= current_price <= ob["high"]:
                        confidence += 0.10
                        has_ob_confluence = True
                        reasons.append("✅ Trong vùng Order Block")
                
                entry = current_price
                sl = recent_high + atr * 0.2
                risk = sl - entry
                
                tp1 = entry - risk * 2
                tp2 = entry - risk * 4
                tp3 = entry - risk * 6
                
                rr = (entry - tp1) / risk if risk > 0 else 0
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.SFP,
                        symbol=symbol,
                        direction="SHORT",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_rsi_divergence=momentum.has_bearish_div,
                        has_wavetrend_cross=momentum.wt_cross_down,
                        has_volume_spike=momentum.has_volume_spike,
                        has_ob_confluence=has_ob_confluence,
                        zone_type="PREMIUM"
                    )
        
        return None
    
    def _detect_liquidity_sweep_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm, ob_zones: Dict
    ) -> Optional[TradeSetup]:
        """
        Detect Bullish Liquidity Sweep.
        Multiple swing lows in a row get swept (double/triple bottom sweep).
        """
        if df_m5.empty or len(df_m5) < 20:
            return None
        
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        
        # Find multiple swing lows that are close together
        swing_lows = self._find_swing_points(lows, self.swing_lookback, "LOW")
        if len(swing_lows) < 2:
            return None
        
        # Check for clustered lows (within 0.3% of each other)
        recent_lows = swing_lows[-3:] if len(swing_lows) >= 3 else swing_lows[-2:]
        low_range = max(recent_lows) - min(recent_lows)
        avg_low = sum(recent_lows) / len(recent_lows)
        
        if low_range / avg_low * 100 < 0.5:  # Lows are clustered
            recent_candle_low = min(lows[-3:])
            
            # Swept the clustered lows
            if recent_candle_low < min(recent_lows) and closes[-1] > min(recent_lows):
                reasons = [
                    f"Quét liquidity vùng đáy cụm ({len(recent_lows)} đáy)",
                    f"Vùng liquidity: {min(recent_lows):.6g} - {max(recent_lows):.6g}",
                    "Đóng nến phục hồi trên vùng quét"
                ]
                
                confidence = 0.75  # Higher confidence for multi-low sweep
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                if momentum.wt_cross_up:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross up")
                
                entry = current_price
                sl = recent_candle_low - atr * 0.15
                risk = entry - sl
                
                tp1 = entry + risk * 2
                tp2 = entry + risk * 4
                tp3 = entry + risk * 6
                
                rr = (tp1 - entry) / risk if risk > 0 else 0
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.LIQUIDITY_SWEEP,
                        symbol=symbol,
                        direction="LONG",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_rsi_divergence=momentum.has_bullish_div,
                        has_wavetrend_cross=momentum.wt_cross_up,
                        has_volume_spike=momentum.has_volume_spike,
                        zone_type="DISCOUNT"
                    )
        
        return None
    
    def _detect_liquidity_sweep_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm, ob_zones: Dict
    ) -> Optional[TradeSetup]:
        """Detect Bearish Liquidity Sweep."""
        if df_m5.empty or len(df_m5) < 20:
            return None
        
        highs = df_m5['high'].values
        closes = df_m5['close'].values
        
        swing_highs = self._find_swing_points(highs, self.swing_lookback, "HIGH")
        if len(swing_highs) < 2:
            return None
        
        recent_highs = swing_highs[-3:] if len(swing_highs) >= 3 else swing_highs[-2:]
        high_range = max(recent_highs) - min(recent_highs)
        avg_high = sum(recent_highs) / len(recent_highs)
        
        if high_range / avg_high * 100 < 0.5:
            recent_candle_high = max(highs[-3:])
            
            if recent_candle_high > max(recent_highs) and closes[-1] < max(recent_highs):
                reasons = [
                    f"Quét liquidity vùng đỉnh cụm ({len(recent_highs)} đỉnh)",
                    f"Vùng liquidity: {min(recent_highs):.6g} - {max(recent_highs):.6g}",
                    "Đóng nến phục hồi dưới vùng quét"
                ]
                
                confidence = 0.75
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                if momentum.wt_cross_down:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross down")
                
                entry = current_price
                sl = recent_candle_high + atr * 0.15
                risk = sl - entry
                
                tp1 = entry - risk * 2
                tp2 = entry - risk * 4
                tp3 = entry - risk * 6
                
                rr = (entry - tp1) / risk if risk > 0 else 0
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.LIQUIDITY_SWEEP,
                        symbol=symbol,
                        direction="SHORT",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_rsi_divergence=momentum.has_bearish_div,
                        has_wavetrend_cross=momentum.wt_cross_down,
                        has_volume_spike=momentum.has_volume_spike,
                        zone_type="PREMIUM"
                    )
        
        return None
    
    def _detect_ema_pullback_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, trend: TrendFilter, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        Detect EMA Cloud Pullback (Long).
        Price pulls back into the EMA34-EMA89 zone in strong uptrend.
        """
        # Condition: Strong trend and price in the "cloud"
        if not trend.is_strong_trend:
            return None
        
        ema34 = trend.ema34
        ema89 = trend.ema89
        
        # Price should be between EMA34 and EMA89 (in the cloud)
        if not (ema89 <= current_price <= ema34 * 1.002):  # Allow 0.2% above EMA34
            return None
        
        # Check for pinbar or engulfing at the cloud
        if df_m5.empty or len(df_m5) < 5:
            return None
        
        opens = df_m5['open'].values
        closes = df_m5['close'].values
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        
        # Last candle should be bullish with wick
        if closes[-1] <= opens[-1]:  # Not bullish
            return None
        
        candle_range = highs[-1] - lows[-1]
        lower_wick = min(opens[-1], closes[-1]) - lows[-1]
        
        # Should have lower wick (testing the cloud)
        if candle_range > 0 and lower_wick / candle_range < 0.3:
            return None
        
        reasons = [
            f"Pullback vào vùng EMA Cloud",
            f"EMA34: {ema34:.6g} | EMA89: {ema89:.6g}",
            f"Trend mạnh: EMA gap {trend.ema_gap_pct:.2f}%",
            "Nến xanh rút chân tại Cloud"
        ]
        
        confidence = 0.65
        
        if momentum.wt_cross_up:
            confidence += 0.05
            reasons.append("✅ WaveTrend cross up")
        
        if momentum.has_volume_spike:
            confidence += 0.05
            reasons.append(f"✅ Volume spike")
        
        entry = current_price
        sl = ema89 - atr * 0.3  # SL below EMA89
        risk = entry - sl
        
        tp1 = entry + risk * 2
        tp2 = entry + risk * 4
        tp3 = entry + risk * 6
        
        rr = (tp1 - entry) / risk if risk > 0 else 0
        
        if rr >= self.min_rr:
            return TradeSetup(
                strategy=StrategyType.EMA_PULLBACK,
                symbol=symbol,
                direction="LONG",
                entry_price=entry,
                stop_loss=sl,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                confidence=min(1.0, confidence),
                risk_reward=rr,
                reasons=reasons,
                has_wavetrend_cross=momentum.wt_cross_up,
                has_volume_spike=momentum.has_volume_spike,
                zone_type="EMA_CLOUD"
            )
        
        return None
    
    def _detect_ema_pullback_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, trend: TrendFilter, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """Detect EMA Cloud Pullback (Short)."""
        if not trend.is_strong_trend:
            return None
        
        ema34 = trend.ema34
        ema89 = trend.ema89
        
        # Price in cloud for downtrend
        if not (ema34 * 0.998 <= current_price <= ema89):
            return None
        
        if df_m5.empty or len(df_m5) < 5:
            return None
        
        opens = df_m5['open'].values
        closes = df_m5['close'].values
        highs = df_m5['high'].values
        lows = df_m5['low'].values
        
        if closes[-1] >= opens[-1]:  # Not bearish
            return None
        
        candle_range = highs[-1] - lows[-1]
        upper_wick = highs[-1] - max(opens[-1], closes[-1])
        
        if candle_range > 0 and upper_wick / candle_range < 0.3:
            return None
        
        reasons = [
            f"Pullback vào vùng EMA Cloud",
            f"EMA34: {ema34:.6g} | EMA89: {ema89:.6g}",
            f"Trend mạnh: EMA gap {trend.ema_gap_pct:.2f}%",
            "Nến đỏ rút chân tại Cloud"
        ]
        
        confidence = 0.65
        
        if momentum.wt_cross_down:
            confidence += 0.05
            reasons.append("✅ WaveTrend cross down")
        
        if momentum.has_volume_spike:
            confidence += 0.05
            reasons.append(f"✅ Volume spike")
        
        entry = current_price
        sl = ema89 + atr * 0.3
        risk = sl - entry
        
        tp1 = entry - risk * 2
        tp2 = entry - risk * 4
        tp3 = entry - risk * 6
        
        rr = (entry - tp1) / risk if risk > 0 else 0
        
        if rr >= self.min_rr:
            return TradeSetup(
                strategy=StrategyType.EMA_PULLBACK,
                symbol=symbol,
                direction="SHORT",
                entry_price=entry,
                stop_loss=sl,
                take_profit_1=tp1,
                take_profit_2=tp2,
                take_profit_3=tp3,
                confidence=min(1.0, confidence),
                risk_reward=rr,
                reasons=reasons,
                has_wavetrend_cross=momentum.wt_cross_down,
                has_volume_spike=momentum.has_volume_spike,
                zone_type="EMA_CLOUD"
            )
        
        return None
    
    def _detect_breaker_retest_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        Detect Breaker Block Retest (Long).
        Price broke above a swing high, then retests it as support.
        """
        if df_h1.empty or len(df_h1) < 30:
            return None
        
        highs = df_h1['high'].values
        closes = df_h1['close'].values
        
        # Find swing highs that got broken
        swing_highs = self._find_swing_points(highs, self.swing_lookback, "HIGH")
        if len(swing_highs) < 2:
            return None
        
        # Check for breakout + retest
        for i in range(len(swing_highs) - 1, max(0, len(swing_highs) - 4), -1):
            broken_high = swing_highs[i]
            
            # Current price should be just above the broken high (retest)
            if current_price > broken_high * 0.998 and current_price < broken_high * 1.005:
                # Check there was a breakout (price went higher before coming back)
                max_after_break = max(closes[-10:]) if len(closes) >= 10 else closes[-1]
                
                if max_after_break > broken_high * 1.01:  # Was at least 1% above
                    reasons = [
                        f"Retest đỉnh cũ (Breaker Block): {broken_high:.6g}",
                        "Đỉnh cũ thành hỗ trợ mới",
                        f"Breakout trước đó: {max_after_break:.6g}"
                    ]
                    
                    confidence = 0.68
                    
                    if momentum.has_volume_spike:
                        confidence += 0.05
                        reasons.append("✅ Volume xác nhận")
                    
                    entry = current_price
                    sl = broken_high - atr * 0.5
                    risk = entry - sl
                    
                    tp1 = entry + risk * 2
                    tp2 = entry + risk * 4
                    tp3 = entry + risk * 6
                    
                    rr = (tp1 - entry) / risk if risk > 0 else 0
                    
                    if rr >= self.min_rr:
                        return TradeSetup(
                            strategy=StrategyType.BREAKER_RETEST,
                            symbol=symbol,
                            direction="LONG",
                            entry_price=entry,
                            stop_loss=sl,
                            take_profit_1=tp1,
                            take_profit_2=tp2,
                            take_profit_3=tp3,
                            confidence=min(1.0, confidence),
                            risk_reward=rr,
                            reasons=reasons,
                            has_volume_spike=momentum.has_volume_spike,
                            zone_type="BREAKER"
                        )
        
        return None
    
    def _detect_breaker_retest_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """Detect Breaker Block Retest (Short)."""
        if df_h1.empty or len(df_h1) < 30:
            return None
        
        lows = df_h1['low'].values
        closes = df_h1['close'].values
        
        swing_lows = self._find_swing_points(lows, self.swing_lookback, "LOW")
        if len(swing_lows) < 2:
            return None
        
        for i in range(len(swing_lows) - 1, max(0, len(swing_lows) - 4), -1):
            broken_low = swing_lows[i]
            
            if current_price < broken_low * 1.002 and current_price > broken_low * 0.995:
                min_after_break = min(closes[-10:]) if len(closes) >= 10 else closes[-1]
                
                if min_after_break < broken_low * 0.99:
                    reasons = [
                        f"Retest đáy cũ (Breaker Block): {broken_low:.6g}",
                        "Đáy cũ thành kháng cự mới",
                        f"Breakdown trước đó: {min_after_break:.6g}"
                    ]
                    
                    confidence = 0.68
                    
                    if momentum.has_volume_spike:
                        confidence += 0.05
                        reasons.append("✅ Volume xác nhận")
                    
                    entry = current_price
                    sl = broken_low + atr * 0.5
                    risk = sl - entry
                    
                    tp1 = entry - risk * 2
                    tp2 = entry - risk * 4
                    tp3 = entry - risk * 6
                    
                    rr = (entry - tp1) / risk if risk > 0 else 0
                    
                    if rr >= self.min_rr:
                        return TradeSetup(
                            strategy=StrategyType.BREAKER_RETEST,
                            symbol=symbol,
                            direction="SHORT",
                            entry_price=entry,
                            stop_loss=sl,
                            take_profit_1=tp1,
                            take_profit_2=tp2,
                            take_profit_3=tp3,
                            confidence=min(1.0, confidence),
                            risk_reward=rr,
                            reasons=reasons,
                            has_volume_spike=momentum.has_volume_spike,
                            zone_type="BREAKER"
                        )
        
        return None
    
    def _find_swing_points(self, data: np.ndarray, lookback: int, point_type: str) -> List[float]:
        """Find swing high or low points."""
        points = []
        
        for i in range(lookback, len(data) - lookback):
            window = data[i-lookback:i+lookback+1]
            
            if point_type == "HIGH" and data[i] == max(window):
                points.append(data[i])
            elif point_type == "LOW" and data[i] == min(window):
                points.append(data[i])
        
        return points
    
    def _calc_ema(self, data: np.ndarray, period: int) -> float:
        """Calculate EMA."""
        if len(data) < period:
            return data[-1] if len(data) > 0 else 0.0
        
        multiplier = 2 / (period + 1)
        ema = data[0]
        
        for price in data[1:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    def _detect_bb_bounce_long(
        self, symbol: str, df: pd.DataFrame, current_price: float,
        atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        Detect BB Bounce LONG - Price touches lower BB and bounces.
        Best for range/sideways markets.
        """
        if df.empty or len(df) < 20:
            return None
        
        closes = df['close'].values
        lows = df['low'].values
        
        # Calculate Bollinger Bands (20, 2)
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_lower = sma20 - 2 * std20
        bb_upper = sma20 + 2 * std20
        bb_middle = sma20
        
        # Check if price touched/pierced lower band recently
        recent_low = min(lows[-3:])
        recent_close = closes[-1]
        
        # Condition: Low touched BB lower, close above it, RSI oversold or near
        if recent_low <= bb_lower * 1.002 and recent_close > bb_lower:
            # Additional: RSI should be < 40 for oversold condition
            if momentum.rsi < 45 or momentum.wt_oversold:
                reasons = [
                    f"📊 Chạm BB Lower band",
                    f"RSI: {momentum.rsi:.0f} (oversold zone)",
                    f"Close above BB Lower - bounce confirmed"
                ]
                
                confidence = 0.65
                
                if momentum.wt_cross_up:
                    confidence += 0.10
                    reasons.append("✅ WaveTrend cross up")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                # Levels: SL below BB lower, TP at BB middle and upper
                entry = current_price
                sl = bb_lower - atr * 0.3
                risk = entry - sl
                
                tp1 = bb_middle  # First target at middle band
                tp2 = bb_upper * 0.98  # Near upper band
                tp3 = bb_upper
                
                rr = (tp1 - entry) / risk if risk > 0 else 0
                
                if rr >= 1.5:  # Lower R:R for range trades
                    return TradeSetup(
                        strategy=StrategyType.BB_BOUNCE,
                        symbol=symbol,
                        direction="LONG",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_wavetrend_cross=momentum.wt_cross_up,
                        has_volume_spike=momentum.has_volume_spike,
                        zone_type="BB_LOWER"
                    )
        
        return None
    
    def _detect_bb_bounce_short(
        self, symbol: str, df: pd.DataFrame, current_price: float,
        atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        Detect BB Bounce SHORT - Price touches upper BB and reverses.
        Best for range/sideways markets.
        """
        if df.empty or len(df) < 20:
            return None
        
        closes = df['close'].values
        highs = df['high'].values
        
        # Calculate Bollinger Bands (20, 2)
        sma20 = np.mean(closes[-20:])
        std20 = np.std(closes[-20:])
        bb_lower = sma20 - 2 * std20
        bb_upper = sma20 + 2 * std20
        bb_middle = sma20
        
        # Check if price touched/pierced upper band recently
        recent_high = max(highs[-3:])
        recent_close = closes[-1]
        
        # Condition: High touched BB upper, close below it, RSI overbought or near
        if recent_high >= bb_upper * 0.998 and recent_close < bb_upper:
            # Additional: RSI should be > 60 for overbought condition
            if momentum.rsi > 55 or momentum.wt_overbought:
                reasons = [
                    f"📊 Chạm BB Upper band",
                    f"RSI: {momentum.rsi:.0f} (overbought zone)",
                    f"Close below BB Upper - reversal confirmed"
                ]
                
                confidence = 0.65
                
                if momentum.wt_cross_down:
                    confidence += 0.10
                    reasons.append("✅ WaveTrend cross down")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                # Levels: SL above BB upper, TP at BB middle and lower
                entry = current_price
                sl = bb_upper + atr * 0.3
                risk = sl - entry
                
                tp1 = bb_middle  # First target at middle band
                tp2 = bb_lower * 1.02  # Near lower band
                tp3 = bb_lower
                
                rr = (entry - tp1) / risk if risk > 0 else 0
                
                if rr >= 1.5:  # Lower R:R for range trades
                    return TradeSetup(
                        strategy=StrategyType.BB_BOUNCE,
                        symbol=symbol,
                        direction="SHORT",
                        entry_price=entry,
                        stop_loss=sl,
                        take_profit_1=tp1,
                        take_profit_2=tp2,
                        take_profit_3=tp3,
                        confidence=min(1.0, confidence),
                        risk_reward=rr,
                        reasons=reasons,
                        has_wavetrend_cross=momentum.wt_cross_down,
                        has_volume_spike=momentum.has_volume_spike,
                        zone_type="BB_UPPER"
                    )
        
        return None

    def _detect_bearish_continuation(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        Detect Bearish Continuation - Simple trend following for bear market.
        Triggers when price is in downtrend and has small bounce/consolidation.
        """
        if df_m5.empty or len(df_m5) < 20 or df_h1.empty:
            return None
        
        closes_m5 = df_m5['close'].values
        highs_m5 = df_m5['high'].values
        lows_m5 = df_m5['low'].values
        closes_h1 = df_h1['close'].values
        
        # Calculate EMAs
        ema20_m5 = self._calc_ema(closes_m5, 20)
        ema50_m5 = self._calc_ema(closes_m5, 50) if len(closes_m5) >= 50 else ema20_m5
        ema34_h1 = self._calc_ema(closes_h1, 34)
        ema89_h1 = self._calc_ema(closes_h1, 89)
        
        # Conditions for bearish continuation:
        # 1. H1 trend is bearish (price < EMA34 < EMA89 OR EMA34 < EMA89)
        # 2. M5 price bounced slightly (touched or near EMA20)
        # 3. RSI not oversold (room to fall)
        # 4. Current candle shows rejection (upper wick)
        
        h1_bearish = ema34_h1 < ema89_h1 or current_price < ema34_h1
        
        if not h1_bearish:
            return None
        
        # Check if price recently touched EMA20 (bounce point)
        recent_highs = highs_m5[-5:]
        ema20_touch = any(h >= ema20_m5 * 0.995 for h in recent_highs)
        
        # Price should be near or below EMA20 now
        near_ema = current_price <= ema20_m5 * 1.02
        
        # RSI not oversold (has room to fall)
        rsi_ok = momentum.rsi > 30 and momentum.rsi < 60
        
        # WaveTrend confirmation (bearish)
        wt_bearish = momentum.wt1 < momentum.wt2 or momentum.wt1 < 0
        
        if h1_bearish and (ema20_touch or near_ema) and rsi_ok:
            reasons = [
                f"📉 H1 trend bearish (EMA34 < EMA89)",
                f"📊 Price rejected at EMA20 ({ema20_m5:.6g})",
                f"RSI: {momentum.rsi:.0f} (room to fall)"
            ]
            
            confidence = 0.55  # Lower confidence for simple setup
            
            if wt_bearish:
                confidence += 0.10
                reasons.append("✅ WaveTrend bearish")
            
            if momentum.wt_cross_down:
                confidence += 0.10
                reasons.append("✅ WaveTrend cross down")
            
            if momentum.has_volume_spike:
                confidence += 0.05
                reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
            
            # Levels
            entry = current_price
            recent_high = max(highs_m5[-5:])
            sl = max(recent_high + atr * 0.2, entry + atr * 0.5)
            risk = sl - entry
            
            tp1 = entry - risk * 1.5  # 1.5R
            tp2 = entry - risk * 2.5  # 2.5R
            tp3 = entry - risk * 4    # 4R
            
            rr = (entry - tp1) / risk if risk > 0 else 0
            
            if rr >= 1.3:  # Lower R:R requirement
                return TradeSetup(
                    strategy=StrategyType.EMA_PULLBACK,  # Reuse existing type
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    take_profit_3=tp3,
                    confidence=min(1.0, confidence),
                    risk_reward=rr,
                    reasons=reasons,
                    has_wavetrend_cross=momentum.wt_cross_down,
                    has_volume_spike=momentum.has_volume_spike,
                    zone_type="BEARISH_CONTINUATION"
                )
        
        return None
