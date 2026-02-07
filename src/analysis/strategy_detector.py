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
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from .poi_manager import POIManager, POI, POIType
from .fvg_bridge import FVGBridge

logger = logging.getLogger(__name__)


class StrategyType(Enum):
    """Active trading strategies — Dream Team v2.0."""
    SFP = "SFP"                          # Swing Failure Pattern (MAIN)
    LIQUIDITY_SWEEP = "LIQ_SWEEP"        # Liquidity Sweep (MAIN)
    SILVER_BULLET = "SILVER_BULLET"      # ICT Silver Bullet + Judas Swing (TIME-BASED)
    UNICORN = "UNICORN"                  # ICT Unicorn Model: Breaker + FVG (STRUCTURE)
    TURTLE_SOUP = "TURTLE_SOUP"          # ICT Turtle Soup: HTF False Breakout (REVERSAL)


STRATEGY_ICONS = {
    StrategyType.SFP: "🔄",
    StrategyType.LIQUIDITY_SWEEP: "🌊",
    StrategyType.SILVER_BULLET: "🎯",
    StrategyType.UNICORN: "🦄",
    StrategyType.TURTLE_SOUP: "🐢",
}

STRATEGY_NAMES = {
    StrategyType.SFP: "Swing Failure Pattern",
    StrategyType.LIQUIDITY_SWEEP: "Liquidity Sweep",
    StrategyType.SILVER_BULLET: "ICT Silver Bullet",
    StrategyType.UNICORN: "ICT Unicorn Model",
    StrategyType.TURTLE_SOUP: "ICT Turtle Soup",
}

# Volume allocation per strategy
STRATEGY_VOLUME_WEIGHT = {
    StrategyType.SFP: 1.0,              # 100% standard volume
    StrategyType.LIQUIDITY_SWEEP: 1.0,  # 100% standard volume
    StrategyType.SILVER_BULLET: 1.0,    # 100% (time-confirmed ICT)
    StrategyType.UNICORN: 0.9,          # 90% (structure-based)
    StrategyType.TURTLE_SOUP: 0.9,      # 90% (HTF reversal)
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
    
    # ICT-specific fields
    is_kill_zone: bool = False          # Trade in ICT kill zone window
    has_htf_poi: bool = False           # Aligned with H1/H4 POI level
    has_bpr_confluence: bool = False    # Entry at BPR zone
    has_ifvg_confluence: bool = False   # Entry at Inverse FVG level
    is_super_setup: bool = False        # Qualifies for LONG exception
    is_judas_swing: bool = False        # Judas Swing variant detected
    ict_conditions_met: int = 0         # Count of ICT conditions met
    
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
    Strategy Detector v3.0 — ICT Dream Team.
    
    5 Strategies:
    1. SFP (Swing Failure Pattern) — reversal
    2. Liquidity Sweep — momentum
    3. Silver Bullet + Judas — time-based ICT
    4. Unicorn Model — structure ICT (Breaker + FVG)
    5. Turtle Soup — HTF false breakout
    
    Logic Flow:
    1. Check H1 Trend Filter (EMA34/89)
    2. Detect setups on M5/M15 with POI alignment
    3. Confirm with momentum + ICT conditions
    """
    
    def __init__(self):
        self.swing_lookback = 5
        self.entry_proximity = 0.003  # 0.3% for actionable
        self.min_rr = 2.0  # Minimum 2:1 R:R
        self.ema_gap_strong = 0.5  # 0.5% gap = strong trend
        
        # ICT components
        self.poi_manager = POIManager()
        self.fvg_bridge = FVGBridge()
        
        # ICT Kill Zone windows (UTC hours)
        # Silver Bullet AM: 15:00-16:00 UTC (10-11 AM EST)
        # Silver Bullet PM: 19:00-20:00 UTC (2-3 PM EST)
        # Judas Swing: 07:00-07:30 UTC (EU Open)
        self.sb_am_start = 15
        self.sb_am_end = 16
        self.sb_pm_start = 19
        self.sb_pm_end = 20
        self.judas_start = 7
        self.judas_end_minute = 30  # 07:00-07:30
    
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
        Analyze and return valid setups using Dream Team v3.0.
        Only returns setups that pass all filters.
        """
        setups = []
        
        if df_h1.empty or len(df_h1) < 50:
            return setups
        
        # --- Phase 1: Foundation ---
        # Update POI levels for this symbol
        self.poi_manager.update(symbol, df_h1)
        
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
        
        # Step 4: Get current UTC time for kill zone checks
        utc_now = datetime.now(timezone.utc)
        
        # Step 5: Detect setups based on allowed direction
        allowed = trend_filter.allowed_direction
        
        # --- LONG setups ---
        if allowed in ["LONG", "BOTH_BULLISH_BIAS", "BOTH_BEARISH_BIAS"]:
            # Core strategies (kept)
            if setup := self._detect_sfp_long(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bearish)")
                setups.append(setup)
            
            if setup := self._detect_liquidity_sweep_long(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bearish)")
                setups.append(setup)
            
            # New ICT strategies
            if setup := self._detect_silver_bullet_long(symbol, df_m5, df_m15, current_price, atr, momentum, utc_now):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
            
            if setup := self._detect_unicorn_long(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
            
            if setup := self._detect_turtle_soup_long(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BEARISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
        
        # --- SHORT setups ---
        if allowed in ["SHORT", "BOTH_BULLISH_BIAS", "BOTH_BEARISH_BIAS"]:
            # Core strategies (kept)
            if setup := self._detect_sfp_short(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bullish)")
                setups.append(setup)
            
            if setup := self._detect_liquidity_sweep_short(symbol, df_m5, df_h1, current_price, atr, momentum, ob_zones):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                    setup.reasons.append("⚠️ Counter-trend (EMA bullish)")
                setups.append(setup)
            
            # New ICT strategies
            if setup := self._detect_silver_bullet_short(symbol, df_m5, df_m15, current_price, atr, momentum, utc_now):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
            
            if setup := self._detect_unicorn_short(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
            
            if setup := self._detect_turtle_soup_short(symbol, df_m5, df_h1, current_price, atr, momentum):
                if allowed == "BOTH_BULLISH_BIAS":
                    setup.confidence *= 0.85
                setups.append(setup)
        
        # --- Enrich with ICT confluence data ---
        pois = self.poi_manager.get_pois(symbol)
        for setup in setups:
            self._enrich_ict_confluence(setup, pois, current_price, utc_now)
        
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
                    f"Swept swing low {last_swing_low:.6g}",
                    f"Strong wick rejection (wick {lower_wick/candle_range*100:.0f}%)",
                    f"Closed above sweep zone"
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
                    f"Swept swing high {last_swing_high:.6g}",
                    f"Strong wick rejection (wick {upper_wick/candle_range*100:.0f}%)",
                    f"Closed below sweep zone"
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
                    f"Swept clustered lows ({len(recent_lows)} lows)",
                    f"Liquidity zone: {min(recent_lows):.6g} - {max(recent_lows):.6g}",
                    "Recovered above sweep zone"
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
                    f"Swept clustered highs ({len(recent_highs)} highs)",
                    f"Liquidity zone: {min(recent_highs):.6g} - {max(recent_highs):.6g}",
                    "Recovered below sweep zone"
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
    
    # ==================== ICT SILVER BULLET ====================
    
    def _is_silver_bullet_window(self, utc_now: datetime) -> Tuple[bool, str]:
        """Check if current time is in Silver Bullet kill zone window."""
        hour = utc_now.hour
        minute = utc_now.minute
        
        # AM window: 15:00-16:00 UTC (10-11 AM EST)
        if self.sb_am_start <= hour < self.sb_am_end:
            return True, "AM"
        # PM window: 19:00-20:00 UTC (2-3 PM EST)
        if self.sb_pm_start <= hour < self.sb_pm_end:
            return True, "PM"
        # Judas Swing: 07:00-07:30 UTC (EU open)
        if hour == self.judas_start and minute < self.judas_end_minute:
            return True, "JUDAS"
        
        return False, ""
    
    def _detect_silver_bullet_long(
        self, symbol: str, df_m5: pd.DataFrame, df_m15: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm,
        utc_now: datetime
    ) -> Optional[TradeSetup]:
        """
        ICT Silver Bullet LONG + Judas Swing variant.
        
        Conditions:
        1. Current time in AM/PM kill zone OR Judas window (07:00-07:30 UTC)
        2. M15 swing low was swept (liquidity taken)
        3. M5 Bullish FVG formed during the window
        4. Entry at FVG midpoint
        
        Judas variant: Asian session low swept at EU open + strong reversal
        """
        in_window, window_type = self._is_silver_bullet_window(utc_now)
        if not in_window:
            return None
        
        if df_m5.empty or len(df_m5) < 10 or df_m15.empty or len(df_m15) < 10:
            return None
        
        # Check if M15 swing low was recently swept
        m15_lows = df_m15['low'].values
        swing_lows = self._find_swing_points(m15_lows, 3, "LOW")
        if len(swing_lows) < 1:
            return None
        
        last_swing_low = swing_lows[-1]
        recent_m5_low = min(df_m5['low'].values[-5:])
        recent_m5_close = float(df_m5['close'].values[-1])
        
        # Liquidity sweep: M5 went below M15 swing low then recovered
        if recent_m5_low < last_swing_low and recent_m5_close > last_swing_low:
            # === BACKTEST-TUNED: Displacement candle required (v3.1) ===
            # Without displacement, Silver Bullet had 0% WR in 90-day backtest
            m5_range = float(df_m5['high'].values[-1]) - float(df_m5['low'].values[-1])
            m5_body = abs(float(df_m5['close'].values[-1]) - float(df_m5['open'].values[-1]))
            if m5_range <= 0 or m5_body / m5_range < 0.50 or momentum.volume_ratio < 1.5:
                return None  # No displacement = no entry
            
            # Look for M5 bullish FVG
            fvg_entry = self.fvg_bridge.find_best_entry_fvg(df_m5, symbol, "LONG", current_price)
            
            reasons = [
                f"🎯 Silver Bullet {window_type} window",
                f"Swept M15 swing low: {last_swing_low:.6g}",
                f"Recovered above sweep zone",
                f"✅ Displacement candle ({m5_body/m5_range*100:.0f}% body, {momentum.volume_ratio:.1f}x vol)"
            ]
            
            confidence = 0.65  # Reduced from 0.72 after backtest
            is_judas = window_type == "JUDAS"
            
            if is_judas:
                reasons[0] = "🎯 Judas Swing: Asian low swept at EU open"
                # Judas requires stronger reversal candle
                m5_opens = df_m5['open'].values
                body_size = abs(recent_m5_close - float(m5_opens[-1]))
                candle_range = float(df_m5['high'].values[-1]) - float(df_m5['low'].values[-1])
                if candle_range > 0 and body_size / candle_range >= 0.65:
                    confidence += 0.05
                    reasons.append("✅ Strong reversal candle (Judas)")
                else:
                    return None  # Judas requires strong reversal
            
            if fvg_entry:
                entry_price, fvg_top, fvg_bottom = fvg_entry
                reasons.append(f"✅ M5 FVG entry: {entry_price:.6g}")
                confidence += 0.08  # FVG quality bonus
            else:
                entry_price = current_price
            
            if momentum.wt_cross_up:
                confidence += 0.05
                reasons.append("✅ WaveTrend cross up")
            
            if momentum.has_volume_spike:
                confidence += 0.05
                reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
            
            sl = recent_m5_low - atr * 0.2
            risk = entry_price - sl
            if risk <= 0:
                return None
            
            tp1 = entry_price + risk * 2
            tp2 = entry_price + risk * 4
            tp3 = entry_price + risk * 6
            rr = (tp1 - entry_price) / risk
            
            if rr >= self.min_rr:
                return TradeSetup(
                    strategy=StrategyType.SILVER_BULLET,
                    symbol=symbol,
                    direction="LONG",
                    entry_price=entry_price,
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
                    is_kill_zone=True,
                    is_judas_swing=is_judas,
                    is_super_setup=True,  # Silver Bullet LONG = super setup
                    zone_type="SILVER_BULLET"
                )
        
        return None
    
    def _detect_silver_bullet_short(
        self, symbol: str, df_m5: pd.DataFrame, df_m15: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm,
        utc_now: datetime
    ) -> Optional[TradeSetup]:
        """ICT Silver Bullet SHORT: Kill zone + M15 swing high sweep + M5 bearish FVG."""
        in_window, window_type = self._is_silver_bullet_window(utc_now)
        if not in_window or window_type == "JUDAS":
            return None  # Judas is LONG-only
        
        if df_m5.empty or len(df_m5) < 10 or df_m15.empty or len(df_m15) < 10:
            return None
        
        m15_highs = df_m15['high'].values
        swing_highs = self._find_swing_points(m15_highs, 3, "HIGH")
        if len(swing_highs) < 1:
            return None
        
        last_swing_high = swing_highs[-1]
        recent_m5_high = max(df_m5['high'].values[-5:])
        recent_m5_close = float(df_m5['close'].values[-1])
        
        if recent_m5_high > last_swing_high and recent_m5_close < last_swing_high:
            # === BACKTEST-TUNED: Displacement candle required (v3.1) ===
            m5_range = float(df_m5['high'].values[-1]) - float(df_m5['low'].values[-1])
            m5_body = abs(float(df_m5['close'].values[-1]) - float(df_m5['open'].values[-1]))
            if m5_range <= 0 or m5_body / m5_range < 0.50 or momentum.volume_ratio < 1.5:
                return None
            
            fvg_entry = self.fvg_bridge.find_best_entry_fvg(df_m5, symbol, "SHORT", current_price)
            
            reasons = [
                f"🎯 Silver Bullet {window_type} window",
                f"Swept M15 swing high: {last_swing_high:.6g}",
                f"Rejected below sweep zone",
                f"✅ Displacement candle ({m5_body/m5_range*100:.0f}% body, {momentum.volume_ratio:.1f}x vol)"
            ]
            
            confidence = 0.65  # Reduced from 0.72 after backtest
            
            if fvg_entry:
                entry_price, _, _ = fvg_entry
                reasons.append(f"✅ M5 FVG entry: {entry_price:.6g}")
                confidence += 0.08
            else:
                entry_price = current_price
            
            if momentum.wt_cross_down:
                confidence += 0.05
                reasons.append("✅ WaveTrend cross down")
            
            if momentum.has_volume_spike:
                confidence += 0.05
                reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
            
            sl = recent_m5_high + atr * 0.2
            risk = sl - entry_price
            if risk <= 0:
                return None
            
            tp1 = entry_price - risk * 2
            tp2 = entry_price - risk * 4
            tp3 = entry_price - risk * 6
            rr = (entry_price - tp1) / risk
            
            if rr >= self.min_rr:
                return TradeSetup(
                    strategy=StrategyType.SILVER_BULLET,
                    symbol=symbol,
                    direction="SHORT",
                    entry_price=entry_price,
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
                    is_kill_zone=True,
                    zone_type="SILVER_BULLET"
                )
        
        return None
    
    # ==================== ICT UNICORN MODEL ====================
    
    def _detect_unicorn_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        ICT Unicorn Model LONG: H1 bullish breaker + FVG overlap + M5 rejection.
        
        1. Find H1 bullish breaker block (bearish candle before strong up move)
        2. Find H1 bullish FVG overlapping the breaker zone
        3. M5 price retests this overlap zone + bullish rejection candle (wick >= 30%)
        """
        pois = self.poi_manager.get_pois(symbol)
        breakers = [p for p in pois if p.poi_type == POIType.BREAKER_H1 and p.direction == "BULLISH"]
        fvgs = [p for p in pois if p.poi_type == POIType.FVG_H1 and p.direction == "BULLISH"]
        
        if not breakers or not fvgs:
            return None
        
        # Find breaker+FVG overlap
        for breaker in breakers:
            for fvg in fvgs:
                overlap_low = max(breaker.price_low, fvg.price_low)
                overlap_high = min(breaker.price_high, fvg.price_high)
                
                if overlap_low >= overlap_high:
                    continue
                
                # Price must be in or near the overlap zone
                zone_mid = (overlap_low + overlap_high) / 2
                if not (overlap_low * 0.998 <= current_price <= overlap_high * 1.002):
                    continue
                
                # Check M5 for bullish rejection candle
                if df_m5.empty or len(df_m5) < 3:
                    continue
                
                opens = df_m5['open'].values
                closes = df_m5['close'].values
                highs = df_m5['high'].values
                lows = df_m5['low'].values
                
                candle_range = highs[-1] - lows[-1]
                lower_wick = min(opens[-1], closes[-1]) - lows[-1]
                
                if candle_range <= 0 or lower_wick / candle_range < 0.30:
                    continue
                
                reasons = [
                    f"🦄 Unicorn Model: Breaker + FVG confluence",
                    f"H1 Breaker zone: {breaker.price_low:.6g}-{breaker.price_high:.6g}",
                    f"H1 FVG overlap: {overlap_low:.6g}-{overlap_high:.6g}",
                    f"M5 rejection wick: {lower_wick/candle_range*100:.0f}%"
                ]
                
                confidence = 0.73
                
                if momentum.wt_cross_up:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross up")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                entry = current_price
                sl = overlap_low - atr * 0.3
                risk = entry - sl
                if risk <= 0:
                    continue
                
                tp1 = entry + risk * 2
                tp2 = entry + risk * 4
                tp3 = entry + risk * 6
                rr = (tp1 - entry) / risk
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.UNICORN,
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
                        has_htf_poi=True,
                        is_super_setup=True,  # Unicorn LONG = super setup
                        zone_type="UNICORN"
                    )
        
        return None
    
    def _detect_unicorn_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """ICT Unicorn Model SHORT: H1 bearish breaker + FVG overlap + M5 rejection."""
        pois = self.poi_manager.get_pois(symbol)
        breakers = [p for p in pois if p.poi_type == POIType.BREAKER_H1 and p.direction == "BEARISH"]
        fvgs = [p for p in pois if p.poi_type == POIType.FVG_H1 and p.direction == "BEARISH"]
        
        if not breakers or not fvgs:
            return None
        
        for breaker in breakers:
            for fvg in fvgs:
                overlap_low = max(breaker.price_low, fvg.price_low)
                overlap_high = min(breaker.price_high, fvg.price_high)
                
                if overlap_low >= overlap_high:
                    continue
                
                zone_mid = (overlap_low + overlap_high) / 2
                if not (overlap_low * 0.998 <= current_price <= overlap_high * 1.002):
                    continue
                
                if df_m5.empty or len(df_m5) < 3:
                    continue
                
                opens = df_m5['open'].values
                closes = df_m5['close'].values
                highs = df_m5['high'].values
                lows = df_m5['low'].values
                
                candle_range = highs[-1] - lows[-1]
                upper_wick = highs[-1] - max(opens[-1], closes[-1])
                
                if candle_range <= 0 or upper_wick / candle_range < 0.30:
                    continue
                
                reasons = [
                    f"🦄 Unicorn Model: Breaker + FVG confluence",
                    f"H1 Breaker zone: {breaker.price_low:.6g}-{breaker.price_high:.6g}",
                    f"H1 FVG overlap: {overlap_low:.6g}-{overlap_high:.6g}",
                    f"M5 rejection wick: {upper_wick/candle_range*100:.0f}%"
                ]
                
                confidence = 0.73
                
                if momentum.wt_cross_down:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross down")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                entry = current_price
                sl = overlap_high + atr * 0.3
                risk = sl - entry
                if risk <= 0:
                    continue
                
                tp1 = entry - risk * 2
                tp2 = entry - risk * 4
                tp3 = entry - risk * 6
                rr = (entry - tp1) / risk
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.UNICORN,
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
                        has_htf_poi=True,
                        zone_type="UNICORN"
                    )
        
        return None
    
    # ==================== ICT TURTLE SOUP ====================
    
    def _detect_turtle_soup_long(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """
        ICT Turtle Soup LONG: HTF key level false breakout.
        
        1. Identify PDL or H1 swing low from POI manager
        2. M5 price sweeps the level by <= 0.3%
        3. Immediate reversal candle with wick >= 35%
        4. Close back inside range (above the level)
        
        Differs from SFP: requires HTF structural level (not just M5 swing)
        """
        pois = self.poi_manager.get_pois(symbol)
        # Target levels: PDL and H1 swing lows
        target_pois = [p for p in pois if p.poi_type in [POIType.PDL, POIType.SWING_LOW_H1]]
        
        if not target_pois or df_m5.empty or len(df_m5) < 5:
            return None
        
        lows = df_m5['low'].values
        closes = df_m5['close'].values
        opens = df_m5['open'].values
        highs = df_m5['high'].values
        
        recent_low = min(lows[-3:])
        recent_close = closes[-1]
        
        for poi in target_pois:
            level = poi.price_low
            
            # Sweep: went below by <= 0.3%
            sweep_depth = (level - recent_low) / level * 100
            if recent_low < level and 0 < sweep_depth <= 0.3 and recent_close > level:
                # === BACKTEST-TUNED GATES (v3.1) ===
                # Volume gate: require strong volume (>=2.0x avg)
                if momentum.volume_ratio < 2.0:
                    continue
                # Exhaustion gate: require RSI or WT extreme
                if not (momentum.rsi < 30 or momentum.wt_oversold):
                    continue
                
                # Check for rejection wick
                candle_range = highs[-1] - lows[-1]
                lower_wick = min(opens[-1], closes[-1]) - lows[-1]
                
                if candle_range <= 0 or lower_wick / candle_range < 0.35:
                    continue
                
                poi_label = "PDL" if poi.poi_type == POIType.PDL else "H1 Swing Low"
                reasons = [
                    f"🐢 Turtle Soup: False break of {poi_label}",
                    f"Level: {level:.6g} (sweep depth: {sweep_depth:.2f}%)",
                    f"Rejection wick: {lower_wick/candle_range*100:.0f}%",
                    f"Closed back above level"
                ]
                
                confidence = 0.75  # Higher than SFP due to HTF significance
                
                if momentum.wt_cross_up and momentum.wt_oversold:
                    confidence += 0.08
                    reasons.append("✅ WaveTrend cross up from oversold")
                elif momentum.wt_cross_up:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross up")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                entry = current_price
                sl = recent_low - atr * 0.2
                risk = entry - sl
                if risk <= 0:
                    continue
                
                tp1 = entry + risk * 2
                tp2 = entry + risk * 4
                tp3 = entry + risk * 6
                rr = (tp1 - entry) / risk
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.TURTLE_SOUP,
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
                        has_htf_poi=True,
                        is_super_setup=True,  # Turtle Soup LONG = super setup
                        zone_type="TURTLE_SOUP"
                    )
        
        return None
    
    def _detect_turtle_soup_short(
        self, symbol: str, df_m5: pd.DataFrame, df_h1: pd.DataFrame,
        current_price: float, atr: float, momentum: MomentumConfirm
    ) -> Optional[TradeSetup]:
        """ICT Turtle Soup SHORT: PDH/H1 swing high false breakout."""
        pois = self.poi_manager.get_pois(symbol)
        target_pois = [p for p in pois if p.poi_type in [POIType.PDH, POIType.SWING_HIGH_H1]]
        
        if not target_pois or df_m5.empty or len(df_m5) < 5:
            return None
        
        highs = df_m5['high'].values
        closes = df_m5['close'].values
        opens = df_m5['open'].values
        lows = df_m5['low'].values
        
        recent_high = max(highs[-3:])
        recent_close = closes[-1]
        
        for poi in target_pois:
            level = poi.price_high
            
            sweep_depth = (recent_high - level) / level * 100
            if recent_high > level and 0 < sweep_depth <= 0.3 and recent_close < level:
                # === BACKTEST-TUNED GATES (v3.1) ===
                # Volume gate: require strong volume (>=2.0x avg)
                if momentum.volume_ratio < 2.0:
                    continue
                # Exhaustion gate: require RSI or WT extreme
                if not (momentum.rsi > 70 or momentum.wt_overbought):
                    continue
                
                candle_range = highs[-1] - lows[-1]
                upper_wick = highs[-1] - max(opens[-1], closes[-1])
                
                if candle_range <= 0 or upper_wick / candle_range < 0.35:
                    continue
                
                poi_label = "PDH" if poi.poi_type == POIType.PDH else "H1 Swing High"
                reasons = [
                    f"🐢 Turtle Soup: False break of {poi_label}",
                    f"Level: {level:.6g} (sweep depth: {sweep_depth:.2f}%)",
                    f"Rejection wick: {upper_wick/candle_range*100:.0f}%",
                    f"Closed back below level"
                ]
                
                confidence = 0.75
                
                if momentum.wt_cross_down and momentum.wt_overbought:
                    confidence += 0.08
                    reasons.append("✅ WaveTrend cross down from overbought")
                elif momentum.wt_cross_down:
                    confidence += 0.05
                    reasons.append("✅ WaveTrend cross down")
                
                if momentum.has_volume_spike:
                    confidence += 0.05
                    reasons.append(f"✅ Volume spike {momentum.volume_ratio:.1f}x")
                
                entry = current_price
                sl = recent_high + atr * 0.2
                risk = sl - entry
                if risk <= 0:
                    continue
                
                tp1 = entry - risk * 2
                tp2 = entry - risk * 4
                tp3 = entry - risk * 6
                rr = (entry - tp1) / risk
                
                if rr >= self.min_rr:
                    return TradeSetup(
                        strategy=StrategyType.TURTLE_SOUP,
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
                        has_htf_poi=True,
                        zone_type="TURTLE_SOUP"
                    )
        
        return None
    
    # ==================== ICT CONFLUENCE ENRICHMENT ====================
    
    def _enrich_ict_confluence(
        self, setup: TradeSetup, pois: List[POI],
        current_price: float, utc_now: datetime
    ) -> None:
        """Enrich a TradeSetup with ICT confluence data for scoring bonus."""
        ict_count = 0
        
        # Check kill zone timing
        in_kz, _ = self._is_silver_bullet_window(utc_now)
        if in_kz:
            setup.is_kill_zone = True
            ict_count += 1
        
        # Check HTF POI alignment
        poi_hits = [p for p in pois if p.is_price_at_level(current_price, 0.3)]
        if poi_hits and not setup.has_htf_poi:
            setup.has_htf_poi = True
            ict_count += 1
        elif setup.has_htf_poi:
            ict_count += 1
        
        # Check BPR confluence
        bpr_pois = [p for p in pois if p.poi_type == POIType.BPR_H1]
        for bpr in bpr_pois:
            if bpr.is_price_at_level(current_price, 0.3):
                setup.has_bpr_confluence = True
                ict_count += 1
                break
        
        # Check IFVG confluence
        ifvg_pois = [p for p in pois if p.poi_type == POIType.IFVG_H1]
        for ifvg in ifvg_pois:
            if ifvg.is_price_at_level(current_price, 0.3):
                setup.has_ifvg_confluence = True
                ict_count += 1
                break
        
        setup.ict_conditions_met = ict_count
    
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
