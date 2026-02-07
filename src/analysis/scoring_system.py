"""
Scoring System v3.0 - ICT Dream Team + Confidence Matrix + Tier System

Scoring Matrix (Max 100):
  Strategy: Pump Fade +30, SFP +25, Silver Bullet +30, Unicorn +30, Turtle Soup +28
  ICT Confluence: BPR +15, IFVG +10, Kill Zone +10, HTF POI +12, Judas +8
  Confirm:  RSI Divergence +15, Volume Spike +10
  Penalty:  Counter-trend -25
  Combos: Kill Shot +50, ICT Confluence +20, Dream Setup +35

Tiers: DIAMOND (>=70), GOLD (>=40), SILVER (>=35), REJECT (<35)

Rate Limiter: 8 alerts/hour, tightens at 4+ and 6+.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum
import pandas as pd
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


# --- Enums & Constants ---

class SignalTier(Enum):
    """Signal tier based on score."""
    DIAMOND = "DIAMOND"   # Score >= 80 (Confluence)
    GOLD = "GOLD"         # 60 <= Score < 80 (Single Strategy)
    SILVER = "SILVER"     # 40 <= Score < 60 (Weak)
    REJECT = "REJECT"     # Score < 40


TIER_ICONS = {
    SignalTier.DIAMOND: "💎 DIAMOND SETUP",
    SignalTier.GOLD: "🥇 GOLD SETUP",
    SignalTier.SILVER: "🥈 SILVER (Weak)",
    SignalTier.REJECT: "❌ REJECTED",
}

TIER_VOLUME_WEIGHT = {
    SignalTier.DIAMOND: 1.0,
    SignalTier.GOLD: 0.7,
    SignalTier.SILVER: 0.0,
    SignalTier.REJECT: 0.0,
}


# --- Scoring Constants ---

SCORE_POINTS = {
    # Strategy Points (kept)
    'pump_fade': 30,           # Pump Fade / Shooting Star
    'sfp': 25,                 # Swing Failure Pattern
    'ema_alignment': 20,       # EMA Trend Alignment
    'liquidity_sweep': 25,     # Liquidity Sweep
    
    # ICT Strategy Points (new)
    'silver_bullet': 30,       # ICT Silver Bullet
    'unicorn_model': 30,       # ICT Unicorn Model
    'turtle_soup': 28,         # ICT Turtle Soup
    
    # ICT Confluence Points (new)
    'bpr_confluence': 15,      # Balanced Price Range confluence
    'ifvg_confluence': 10,     # Inverse FVG confluence
    'kill_zone_timing': 10,    # Kill zone timing window
    'htf_poi_alignment': 12,   # HTF POI alignment
    'judas_swing': 8,          # Judas Swing sub-mode active
    
    # Confirmation Points
    'rsi_divergence': 15,      # RSI Divergence
    'volume_spike': 10,        # Volume > 2x MA20
    'wavetrend_cross': 10,     # WaveTrend Cross
    'ob_confluence': 10,       # Order Block Confluence
    'fib_golden_pocket': 15,   # Fib 0.618-0.786 Golden Pocket (stronger)
    'fib_standard': 10,        # Fib 0.5-0.618 Standard Zone
    'macd_divergence': 10,     # MACD Histogram Divergence (for Pump Fade)
    
    # Penalties
    'counter_trend': -25,      # Counter-trend H1
    'no_macd_confirm': -10,    # Pump Fade without MACD confirm
    
    # Bonus Combos
    'kill_shot_bonus': 50,     # Shooting Star + SFP = Kill Shot
    'insurance_bonus': 15,     # Shooting Star + RSI Div
    
    # ICT Combo Bonuses (new)
    'ict_confluence_bonus': 20,  # Multiple ICT confluences stacked
    'dream_setup_bonus': 35,    # Super setup (3+ ICT conditions)
    'htf_mtf_alignment': 15,    # HTF + MTF trend alignment
}

# Tier thresholds (bear market optimized)
THRESHOLD_DIAMOND = 70
THRESHOLD_GOLD = 40
THRESHOLD_SILVER = 35

# Minimum confirmations required for each tier
MIN_CONFIRMATIONS_DIAMOND = 3  # Need at least 3 confirmations for DIAMOND
MIN_CONFIRMATIONS_GOLD = 2     # Need at least 2 confirmations for GOLD


# --- Data Classes ---

@dataclass
class ShootingStarResult:
    """Shooting Star candle detection result."""
    is_valid: bool = False
    upper_wick_pct: float = 0.0
    body_pct: float = 0.0
    close_position_pct: float = 0.0
    volume_ratio: float = 0.0
    is_above_bb_upper: bool = False
    is_rsi_overbought: bool = False
    detail: str = ""


@dataclass
class PumpFadeResult:
    """Pump Fade (Shooting Star + SFP) setup result."""
    is_valid: bool = False
    is_shooting_star: bool = False
    is_sfp: bool = False
    is_kill_shot: bool = False
    has_rsi_divergence: bool = False
    has_volume_spike: bool = False
    swing_high_broken: float = 0.0
    signal_high: float = 0.0
    entry_price: float = 0.0
    stop_loss: float = 0.0
    confidence: int = 0
    detail: str = ""


@dataclass
class ConfidenceScore:
    """Confidence scoring result (0-100)."""
    
    # === STRATEGY POINTS ===
    pump_fade_points: int = 0
    sfp_points: int = 0
    ema_alignment_points: int = 0
    other_strategy_points: int = 0
    
    # === CONFIRMATION POINTS ===
    rsi_divergence_points: int = 0
    volume_spike_points: int = 0
    wavetrend_points: int = 0
    ob_confluence_points: int = 0
    
    # === ICT CONFLUENCE POINTS (new) ===
    bpr_confluence_points: int = 0
    ifvg_confluence_points: int = 0
    kill_zone_timing_points: int = 0
    htf_poi_alignment_points: int = 0
    judas_swing_points: int = 0
    
    # === PENALTY ===
    counter_trend_penalty: int = 0
    
    # === BONUS ===
    kill_shot_bonus: int = 0
    insurance_bonus: int = 0
    ict_combo_bonus: int = 0
    
    # === DETAILS ===
    breakdown: List[str] = field(default_factory=list)
    
    # === FLAGS ===
    is_pump_fade: bool = False
    is_kill_shot: bool = False
    is_counter_trend: bool = False
    suggested_direction: str = "NONE"
    
    # === DYNAMIC SL ===
    use_dynamic_sl: bool = False
    dynamic_sl_price: float = 0.0
    
    @property
    def total_score(self) -> int:
        """Total score capped at 0-100."""
        raw_score = (
            self.pump_fade_points +
            self.sfp_points +
            self.ema_alignment_points +
            self.other_strategy_points +
            self.rsi_divergence_points +
            self.volume_spike_points +
            self.wavetrend_points +
            self.ob_confluence_points +
            self.bpr_confluence_points +
            self.ifvg_confluence_points +
            self.kill_zone_timing_points +
            self.htf_poi_alignment_points +
            self.judas_swing_points +
            self.counter_trend_penalty +
            self.kill_shot_bonus +
            self.insurance_bonus +
            self.ict_combo_bonus
        )
        return min(max(raw_score, 0), 100)
    
    @property
    def confirmation_count(self) -> int:
        """Count number of confirmations (v3.0 - ICT aware)."""
        count = 0
        if self.rsi_divergence_points > 0:
            count += 1
        if self.volume_spike_points > 0:
            count += 1
        if self.wavetrend_points > 0:
            count += 1
        if self.ob_confluence_points > 0:
            count += 1
        if self.ema_alignment_points > 0:
            count += 1
        if self.is_kill_shot:
            count += 2  # Kill shot counts as 2 confirmations
        # ICT confluence confirmations
        if self.bpr_confluence_points > 0:
            count += 1
        if self.ifvg_confluence_points > 0:
            count += 1
        if self.kill_zone_timing_points > 0:
            count += 1
        if self.htf_poi_alignment_points > 0:
            count += 1
        return count
    
    @property
    def tier(self) -> SignalTier:
        """Determine tier based on score AND confirmation count."""
        score = self.total_score
        confirmations = self.confirmation_count
        
        # DIAMOND: needs score >= 80 AND at least 3 confirmations
        if score >= THRESHOLD_DIAMOND and confirmations >= MIN_CONFIRMATIONS_DIAMOND:
            return SignalTier.DIAMOND
        # Downgrade to GOLD if score high but not enough confirmations
        elif score >= THRESHOLD_DIAMOND and confirmations < MIN_CONFIRMATIONS_DIAMOND:
            return SignalTier.GOLD  # Downgrade
        # GOLD: needs score >= 60 AND at least 2 confirmations
        elif score >= THRESHOLD_GOLD and confirmations >= MIN_CONFIRMATIONS_GOLD:
            return SignalTier.GOLD
        # Downgrade to SILVER if score OK but not enough confirmations
        elif score >= THRESHOLD_GOLD and confirmations < MIN_CONFIRMATIONS_GOLD:
            return SignalTier.SILVER  # Downgrade
        elif score >= THRESHOLD_SILVER:
            return SignalTier.SILVER
        else:
            return SignalTier.REJECT
    
    @property
    def tier_label(self) -> str:
        """Display label for tier."""
        return TIER_ICONS.get(self.tier, "")
    
    @property
    def volume_weight(self) -> float:
        """Volume weight based on tier."""
        return TIER_VOLUME_WEIGHT.get(self.tier, 0.0)
    
    @property
    def is_tradeable(self) -> bool:
        """True if score qualifies for trading."""
        return self.tier in [SignalTier.DIAMOND, SignalTier.GOLD]


@dataclass
class RateLimitStatus:
    """Rate limiter status."""
    alerts_last_hour: int = 0
    can_send_diamond: bool = True
    can_send_gold: bool = True
    mode: str = "OPEN"  # OPEN, TIGHT, SNIPER, CLOSED
    next_available: Optional[datetime] = None


# --- Legacy Compatibility (kept to not break imports) ---

class SignalGrade(Enum):
    """Legacy enum - mapped to SignalTier."""
    A_SNIPER = "A"       # → DIAMOND
    B_SCALP = "B"        # → GOLD
    C_WEAK = "C"         # → SILVER
    D_REJECT = "D"       # → REJECT


GRADE_ICONS = {
    SignalGrade.A_SNIPER: "🚨 STRONG SIGNAL 🚨",
    SignalGrade.B_SCALP: "⚠️ RISKY SETUP (COUNTER-TREND)",
    SignalGrade.C_WEAK: "📊 WEAK SIGNAL",
    SignalGrade.D_REJECT: "❌ REJECTED",
}

GRADE_VOLUME_WEIGHT = {
    SignalGrade.A_SNIPER: 1.0,
    SignalGrade.B_SCALP: 0.5,
    SignalGrade.C_WEAK: 0.0,
    SignalGrade.D_REJECT: 0.0,
}


@dataclass
class ChecklistScore:
    """Legacy class - wrapper around ConfidenceScore."""
    
    # Original fields for compatibility
    ema_trend_score: int = 0
    market_structure_score: int = 0
    sfp_sweep_score: int = 0
    retest_zone_score: int = 0
    rsi_wavetrend_score: int = 0
    volume_spike_score: int = 0
    
    ema_trend_detail: str = ""
    market_structure_detail: str = ""
    trigger_detail: str = ""
    momentum_detail: str = ""
    
    suggested_direction: str = "NONE"
    is_counter_trend: bool = False
    
    # New v2.0 fields
    _confidence_score: Optional[ConfidenceScore] = None
    
    @property
    def context_score(self) -> int:
        return self.ema_trend_score + self.market_structure_score
    
    @property
    def trigger_score(self) -> int:
        return max(self.sfp_sweep_score, self.retest_zone_score)
    
    @property
    def momentum_score(self) -> int:
        return self.rsi_wavetrend_score + self.volume_spike_score
    
    @property
    def has_trigger(self) -> bool:
        return self.sfp_sweep_score > 0 or self.retest_zone_score > 0
    
    @property
    def total_score(self) -> int:
        """Legacy 0-3 score."""
        ctx = 1 if self.context_score >= 1 else 0
        trg = 1 if self.has_trigger else 0
        mom = 1 if self.momentum_score >= 1 else 0
        return ctx + trg + mom
    
    @property
    def confidence_points(self) -> int:
        """New 0-100 score from ConfidenceScore."""
        if self._confidence_score:
            return self._confidence_score.total_score
        # Fallback conversion
        return self.total_score * 30  # 0→0, 1→30, 2→60, 3→90
    
    @property
    def tier(self) -> SignalTier:
        """Get tier from confidence points."""
        score = self.confidence_points
        if score >= THRESHOLD_DIAMOND:
            return SignalTier.DIAMOND
        elif score >= THRESHOLD_GOLD:
            return SignalTier.GOLD
        elif score >= THRESHOLD_SILVER:
            return SignalTier.SILVER
        else:
            return SignalTier.REJECT
    
    @property
    def grade(self) -> SignalGrade:
        """Legacy grade mapping."""
        if not self.has_trigger:
            return SignalGrade.D_REJECT
        
        tier = self.tier
        if tier == SignalTier.DIAMOND:
            return SignalGrade.A_SNIPER
        elif tier == SignalTier.GOLD:
            return SignalGrade.B_SCALP
        elif tier == SignalTier.SILVER:
            return SignalGrade.C_WEAK
        else:
            return SignalGrade.D_REJECT
    
    @property
    def is_tradeable(self) -> bool:
        return self.tier in [SignalTier.DIAMOND, SignalTier.GOLD]
    
    @property
    def volume_weight(self) -> float:
        return TIER_VOLUME_WEIGHT.get(self.tier, 0.0)
    
    @property
    def grade_label(self) -> str:
        return GRADE_ICONS.get(self.grade, "")
    
    @property
    def tier_label(self) -> str:
        return TIER_ICONS.get(self.tier, "")


@dataclass
class FourLayerResult:
    """Legacy class for 4-Layer Filter."""
    layer1_pass: bool = False
    layer1_reason: str = ""
    is_overextended: bool = False
    rsi_value: float = 50.0
    bb_position: str = "Normal"
    
    layer2_pass: bool = False
    layer2_reason: str = ""
    is_shooting_star: bool = False
    upper_wick_pct: float = 0.0
    body_pct: float = 0.0
    close_position_pct: float = 0.0
    
    layer3_pass: bool = False
    layer3_reason: str = ""
    volume_ratio: float = 0.0
    
    layer4_pass: bool = False
    layer4_reason: str = ""
    prev_candle_green: bool = False
    prev_candle_body_pct: float = 0.0
    
    @property
    def layers_passed(self) -> int:
        return sum([self.layer1_pass, self.layer2_pass, self.layer3_pass, self.layer4_pass])
    
    @property
    def is_valid_short(self) -> bool:
        return self.layers_passed >= 3


# --- Scoring System Class ---

class ScoringSystem:
    """
    Advanced Scoring System v2.0
    
    Features:
    - Confidence Matrix scoring (0-100)
    - Pump Fade / Shooting Star detection
    - Kill Shot combo (SFP + Shooting Star)
    - Tier system (Diamond/Gold/Silver/Reject)
    - Rate limiter integration
    - Dynamic Stoploss for counter-trend
    """
    
    def __init__(self):
        # Shooting Star thresholds
        self.shooting_star_wick_min = 0.50   # Upper wick > 50% range
        self.shooting_star_body_max = 0.30   # Body < 30% range
        self.shooting_star_close_max = 0.35  # Close in bottom 35%
        
        # Volume thresholds
        self.volume_spike_ratio = 2.0        # Volume > 2x MA20
        
        # RSI thresholds
        self.rsi_overbought = 75
        self.rsi_oversold = 30
        
        # Rate limiter thresholds - stricter for quality (v2.1)
        self.max_alerts_per_hour = 8   # Reduced from 10 for quality
        self.tight_threshold = 4       # Reduced from 5
        self.sniper_threshold = 6      # Reduced from 8
        
        # Confluence requirements (v2.1)
        self.min_confirmations = 2     # Minimum confirmations to trade
        
        # Fib Zone thresholds - Golden Pocket (0.618-0.786)
        self.fib_zone_standard = (0.5, 0.618)     # Standard retracement
        self.fib_zone_golden = (0.618, 0.786)     # Golden pocket (stronger)
        self.short_volume_ratio = 2.0
        self.prev_candle_body_min = 0.40
    
    # --- Shooting Star Detection ---
    
    def detect_shooting_star(
        self,
        df: pd.DataFrame,
        indicators: Dict
    ) -> ShootingStarResult:
        """
        Detect Shooting Star candle pattern.
        
        Conditions:
        1. Upper wick > 50% of range
        2. Body < 30% of range
        3. Close in bottom 35%
        4. Volume > 2x MA20
        5. Price above BB Upper or RSI > 75
        """
        result = ShootingStarResult()
        
        if df.empty or len(df) < 21:
            return result
        
        # Current candle
        curr = df.iloc[-1]
        high = float(curr['high'])
        low = float(curr['low'])
        open_p = float(curr['open'])
        close = float(curr['close'])
        volume = float(curr['volume'])
        
        total_range = high - low
        if total_range == 0:
            return result
        
        # Calculate candle anatomy
        upper_wick = high - max(open_p, close)
        body = abs(close - open_p)
        
        result.upper_wick_pct = upper_wick / total_range
        result.body_pct = body / total_range
        result.close_position_pct = (close - low) / total_range
        
        # Volume ratio
        volumes = df['volume'].tail(21).values
        avg_vol_20 = float(np.mean(volumes[:-1]))
        result.volume_ratio = volume / avg_vol_20 if avg_vol_20 > 0 else 1.0
        
        # Context checks
        bb_upper = indicators.get('bb_upper', high)
        rsi = indicators.get('rsi_15m', 50)
        
        result.is_above_bb_upper = high > bb_upper
        result.is_rsi_overbought = rsi > self.rsi_overbought
        
        # Validate Shooting Star
        is_shooting_star = (
            result.upper_wick_pct >= self.shooting_star_wick_min and
            result.body_pct <= self.shooting_star_body_max and
            result.close_position_pct <= self.shooting_star_close_max and
            result.volume_ratio >= self.volume_spike_ratio and
            (result.is_above_bb_upper or result.is_rsi_overbought)
        )
        
        result.is_valid = is_shooting_star
        
        if is_shooting_star:
            result.detail = (
                f"Shooting Star: Wick {result.upper_wick_pct:.0%}, "
                f"Body {result.body_pct:.0%}, Vol x{result.volume_ratio:.1f}"
            )
        
        return result
    
    # --- Pump Fade Detection (Shooting Star + SFP Combo) ---
    
    def detect_pump_fade(
        self,
        df: pd.DataFrame,
        indicators: Dict,
        swing_high_20: float
    ) -> PumpFadeResult:
        """
        Detect Pump Fade setup (Shooting Star + SFP combo).
        Kill Shot = Shooting Star pierces Swing High then pulls back.
        """
        result = PumpFadeResult()
        
        if df.empty or len(df) < 21:
            return result
        
        # Detect Shooting Star
        shooting_star = self.detect_shooting_star(df, indicators)
        result.is_shooting_star = shooting_star.is_valid
        
        if not shooting_star.is_valid:
            return result
        
        # Current candle data
        curr = df.iloc[-1]
        high = float(curr['high'])
        close = float(curr['close'])
        
        result.signal_high = high
        result.entry_price = close
        
        # Dynamic Stoploss = Signal High + 0.1%
        result.stop_loss = high * 1.001
        
        # Check SFP - Price pierced Swing High then closed below
        if swing_high_20 > 0:
            is_sfp = high > swing_high_20 and close < swing_high_20
            result.is_sfp = is_sfp
            result.swing_high_broken = swing_high_20
            
            if is_sfp:
                # KILL SHOT! Shooting Star + SFP
                result.is_kill_shot = True
                result.detail = f"🔥 KILL SHOT: SS + SFP (broke {swing_high_20:.4f})"
        
        # Check RSI Divergence
        rsi_div = indicators.get('rsi_divergence', 'None')
        result.has_rsi_divergence = rsi_div == "Bearish"
        
        # Check Volume Spike
        result.has_volume_spike = shooting_star.volume_ratio >= self.volume_spike_ratio
        
        # Calculate confidence
        confidence = 0
        if result.is_shooting_star:
            confidence += 30
        if result.is_sfp:
            confidence += 25
        if result.is_kill_shot:
            confidence += 50  # Kill Shot bonus
        if result.has_rsi_divergence:
            confidence += 15
        if result.has_volume_spike:
            confidence += 10
        
        result.confidence = min(confidence, 100)
        result.is_valid = result.is_shooting_star
        
        if not result.detail:
            result.detail = f"Pump Fade: SS detected, Vol x{shooting_star.volume_ratio:.1f}"
        
        return result
    
    # --- Confidence Scoring ---
    
    def calculate_confidence(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        indicators: Dict,
        setup_type: str = None,
        detected_direction: str = None,
        has_sfp: bool = False,
        has_rsi_divergence: bool = False,
        has_volume_spike: bool = False,
        has_wavetrend_cross: bool = False,
        has_ob_confluence: bool = False,
        swing_high_20: float = 0.0,
        swing_low_20: float = 0.0
    ) -> ConfidenceScore:
        """
        Calculate Confidence score using the Scoring Matrix.
        
        Returns:
            ConfidenceScore with total_score, tier, and breakdown
        """
        score = ConfidenceScore()
        score.suggested_direction = detected_direction or "NONE"
        
        if df_h1.empty or df_m15.empty:
            return score
        
        # === 1. DETECT PUMP FADE ===
        if detected_direction == "SHORT":
            pump_fade = self.detect_pump_fade(df_m15, indicators, swing_high_20)
            
            if pump_fade.is_valid:
                score.is_pump_fade = True
                score.pump_fade_points = SCORE_POINTS['pump_fade']
                score.breakdown.append(f"✅ Pump Fade: +{SCORE_POINTS['pump_fade']} pts")
                
                # Dynamic SL for counter-trend
                score.use_dynamic_sl = True
                score.dynamic_sl_price = pump_fade.stop_loss
                
                if pump_fade.is_kill_shot:
                    score.is_kill_shot = True
                    score.kill_shot_bonus = SCORE_POINTS['kill_shot_bonus']
                    score.breakdown.append(f"🔥 KILL SHOT Bonus: +{SCORE_POINTS['kill_shot_bonus']} pts")
                
                if pump_fade.has_rsi_divergence:
                    score.insurance_bonus = SCORE_POINTS['insurance_bonus']
                    score.breakdown.append(f"⚡ Insurance (RSI Div): +{SCORE_POINTS['insurance_bonus']} pts")
        
        # === 2. SFP POINTS (v2.1 - stricter requirements) ===
        if has_sfp or setup_type in ["SFP", "LIQUIDITY_SWEEP", "LIQ_SWEEP"]:
            if not score.is_kill_shot:  # Avoid double-counting with Kill Shot
                # SFP requires volume spike to confirm liquidity grab
                if has_volume_spike:
                    score.sfp_points = SCORE_POINTS['sfp']
                    score.breakdown.append(f"✅ SFP/Sweep + Vol: +{SCORE_POINTS['sfp']} pts")
                else:
                    # Reduce points if no volume confirmation
                    score.sfp_points = SCORE_POINTS['sfp'] - 10  # 15 instead of 25
                    score.breakdown.append(f"⚠️ SFP (no vol): +{SCORE_POINTS['sfp'] - 10} pts")
        
        # === 3. EMA ALIGNMENT ===
        price = float(df_m15['close'].iloc[-1])
        ema34_h1 = indicators.get('ema34_h1', 0)
        ema89_h1 = indicators.get('ema89_h1', 0)
        
        h1_trend = "NEUTRAL"
        if price > ema34_h1 > ema89_h1:
            h1_trend = "BULLISH"
        elif price < ema34_h1 < ema89_h1:
            h1_trend = "BEARISH"
        
        # Check trend alignment
        is_aligned = (
            (h1_trend == "BULLISH" and detected_direction == "LONG") or
            (h1_trend == "BEARISH" and detected_direction == "SHORT")
        )
        
        if is_aligned:
            score.ema_alignment_points = SCORE_POINTS['ema_alignment']
            score.breakdown.append(f"✅ EMA Trend Align: +{SCORE_POINTS['ema_alignment']} pts")
        
        # Check counter-trend penalty
        # REVERSAL strategies are MEANT to be counter-trend - don't penalize them
        strategy_str = str(setup_type).upper() if setup_type else ""
        reversal_strategies = ["SFP", "PUMP_FADE", "LIQUIDITY_SWEEP", "SHOOTING_STAR",
                               "SILVER_BULLET", "UNICORN", "TURTLE_SOUP"]
        is_reversal_strategy = any(s in strategy_str for s in reversal_strategies)
        
        is_counter = (
            (h1_trend == "BULLISH" and detected_direction == "SHORT") or
            (h1_trend == "BEARISH" and detected_direction == "LONG")
        )
        
        # Only penalize counter-trend for pure trend-following strategies
        if is_counter and not is_reversal_strategy:
            score.is_counter_trend = True
            score.counter_trend_penalty = SCORE_POINTS['counter_trend']
            score.breakdown.append(f"⚠️ Counter-Trend: {SCORE_POINTS['counter_trend']} pts")
        
        # === 4. ICT STRATEGY POINTS ===
        if "SILVER_BULLET" in strategy_str:
            score.other_strategy_points = SCORE_POINTS['silver_bullet']
            score.breakdown.append(f"🎯 Silver Bullet: +{SCORE_POINTS['silver_bullet']} pts")
        elif "UNICORN" in strategy_str:
            score.other_strategy_points = SCORE_POINTS['unicorn_model']
            score.breakdown.append(f"🦄 Unicorn Model: +{SCORE_POINTS['unicorn_model']} pts")
        elif "TURTLE_SOUP" in strategy_str:
            score.other_strategy_points = SCORE_POINTS['turtle_soup']
            score.breakdown.append(f"🐢 Turtle Soup: +{SCORE_POINTS['turtle_soup']} pts")
        
        # === 5. CONFIRMATION POINTS ===
        if has_rsi_divergence and not score.is_pump_fade:
            score.rsi_divergence_points = SCORE_POINTS['rsi_divergence']
            score.breakdown.append(f"✅ RSI Divergence: +{SCORE_POINTS['rsi_divergence']} pts")
        
        if has_volume_spike:
            score.volume_spike_points = SCORE_POINTS['volume_spike']
            score.breakdown.append(f"✅ Volume Spike: +{SCORE_POINTS['volume_spike']} pts")
        
        if has_wavetrend_cross:
            score.wavetrend_points = SCORE_POINTS['wavetrend_cross']
            score.breakdown.append(f"✅ WaveTrend Cross: +{SCORE_POINTS['wavetrend_cross']} pts")
        
        if has_ob_confluence:
            score.ob_confluence_points = SCORE_POINTS['ob_confluence']
            score.breakdown.append(f"✅ OB Confluence: +{SCORE_POINTS['ob_confluence']} pts")
        
        # === 6. ICT CONFLUENCE SCORING ===
        # These read from TradeSetup fields passed via indicators dict
        setup_has_bpr = indicators.get('has_bpr_confluence', False)
        setup_has_ifvg = indicators.get('has_ifvg_confluence', False)
        setup_is_kill_zone = indicators.get('is_kill_zone', False)
        setup_has_htf_poi = indicators.get('has_htf_poi', False)
        setup_is_judas = indicators.get('is_judas_swing', False)
        setup_is_super = indicators.get('is_super_setup', False)
        ict_conditions = indicators.get('ict_conditions_met', 0)
        
        if setup_has_bpr:
            score.bpr_confluence_points = SCORE_POINTS['bpr_confluence']
            score.breakdown.append(f"✅ BPR Confluence: +{SCORE_POINTS['bpr_confluence']} pts")
        
        if setup_has_ifvg:
            score.ifvg_confluence_points = SCORE_POINTS['ifvg_confluence']
            score.breakdown.append(f"✅ IFVG Confluence: +{SCORE_POINTS['ifvg_confluence']} pts")
        
        if setup_is_kill_zone:
            score.kill_zone_timing_points = SCORE_POINTS['kill_zone_timing']
            score.breakdown.append(f"✅ Kill Zone Timing: +{SCORE_POINTS['kill_zone_timing']} pts")
        
        if setup_has_htf_poi:
            score.htf_poi_alignment_points = SCORE_POINTS['htf_poi_alignment']
            score.breakdown.append(f"✅ HTF POI Alignment: +{SCORE_POINTS['htf_poi_alignment']} pts")
        
        if setup_is_judas:
            score.judas_swing_points = SCORE_POINTS['judas_swing']
            score.breakdown.append(f"✅ Judas Swing: +{SCORE_POINTS['judas_swing']} pts")
        
        # ICT combo bonuses
        if ict_conditions >= 3:
            score.ict_combo_bonus = SCORE_POINTS['dream_setup_bonus']
            score.breakdown.append(f"🌟 Dream Setup ({ict_conditions} ICT): +{SCORE_POINTS['dream_setup_bonus']} pts")
        elif ict_conditions >= 2:
            score.ict_combo_bonus = SCORE_POINTS['ict_confluence_bonus']
            score.breakdown.append(f"⚡ ICT Confluence ({ict_conditions}): +{SCORE_POINTS['ict_confluence_bonus']} pts")
        
        # HTF + MTF alignment bonus
        is_ict_strategy = any(s in strategy_str for s in ["SILVER_BULLET", "UNICORN", "TURTLE_SOUP"])
        if is_ict_strategy and is_aligned and setup_has_htf_poi:
            score.ict_combo_bonus += SCORE_POINTS['htf_mtf_alignment']
            score.breakdown.append(f"✅ HTF+MTF Alignment: +{SCORE_POINTS['htf_mtf_alignment']} pts")
        
        # === SUMMARY ===
        score.breakdown.append(f"━━━━━━━━━━━━━━━━━━━━━")
        score.breakdown.append(f"📊 TOTAL: {score.total_score}/100 → {score.tier.value}")
        
        return score
    
    # --- Rate Limiter ---
    
    def check_rate_limit(
        self,
        alerts_last_hour: int,
        score: int
    ) -> Tuple[bool, str]:
        """
        Kiểm tra rate limit và quyết định có gửi alert không.
        
        Args:
            alerts_last_hour: Số alerts đã gửi trong 1 giờ qua
            score: Điểm của signal hiện tại
            
        Returns:
            (can_send, reason)
        """
        if alerts_last_hour >= self.max_alerts_per_hour:
            return False, "CLOSED: Đã đạt giới hạn 10 alerts/hour"
        
        if alerts_last_hour >= self.sniper_threshold:
            # Sniper mode: chỉ >= 90
            if score >= 90:
                return True, "SNIPER: Super setup (>=90)"
            return False, f"SNIPER: Score {score} < 90 (cần Super)"
        
        if alerts_last_hour >= self.tight_threshold:
            # Tight mode: chỉ Diamond >= 80
            if score >= THRESHOLD_DIAMOND:
                return True, "TIGHT: Diamond setup (>=80)"
            return False, f"TIGHT: Score {score} < 80 (cần Diamond)"
        
        # Open mode: cả Diamond và Gold
        if score >= THRESHOLD_GOLD:
            return True, "OPEN: Gold+ setup (>=60)"
        return False, f"OPEN: Score {score} < 60 (minimum threshold)"
    
    def get_rate_limit_status(self, alerts_last_hour: int) -> RateLimitStatus:
        """Get rate limiter status."""
        status = RateLimitStatus(alerts_last_hour=alerts_last_hour)
        
        if alerts_last_hour >= self.max_alerts_per_hour:
            status.mode = "CLOSED"
            status.can_send_diamond = False
            status.can_send_gold = False
        elif alerts_last_hour >= self.sniper_threshold:
            status.mode = "SNIPER"
            status.can_send_diamond = True  # But requires >= 90
            status.can_send_gold = False
        elif alerts_last_hour >= self.tight_threshold:
            status.mode = "TIGHT"
            status.can_send_diamond = True
            status.can_send_gold = False
        else:
            status.mode = "OPEN"
            status.can_send_diamond = True
            status.can_send_gold = True
        
        return status
    
    # --- Legacy Methods (kept for backward compatibility) ---
    
    def evaluate_checklist(
        self,
        df_m15: pd.DataFrame,
        df_h1: pd.DataFrame,
        indicators: Dict,
        setup_type: str = None,
        detected_direction: str = None
    ) -> ChecklistScore:
        """
        Legacy method - evaluate checklist scoring.
        Now internally uses calculate_confidence() for scoring.
        """
        score = ChecklistScore()
        
        if df_h1.empty or df_m15.empty:
            return score
        
        # Get data
        price = float(df_m15['close'].iloc[-1])
        
        # === CONDITION 1: EMA TREND ===
        ema34_h1 = indicators.get('ema34_h1', 0)
        ema89_h1 = indicators.get('ema89_h1', 0)
        
        h1_trend = "NEUTRAL"
        if price > ema34_h1 > ema89_h1:
            h1_trend = "BULLISH"
            score.ema_trend_score = 1 if detected_direction == "LONG" else 0
            score.ema_trend_detail = f"Price > EMA34 > EMA89 (H1 UPTREND)"
        elif price < ema34_h1 < ema89_h1:
            h1_trend = "BEARISH"
            score.ema_trend_score = 1 if detected_direction == "SHORT" else 0
            score.ema_trend_detail = f"Price < EMA34 < EMA89 (H1 DOWNTREND)"
        else:
            score.ema_trend_detail = f"EMA no clear trend (H1 SIDEWAYS)"
        
        # Counter-trend check
        if detected_direction:
            if (h1_trend == "BULLISH" and detected_direction == "SHORT") or \
               (h1_trend == "BEARISH" and detected_direction == "LONG"):
                score.is_counter_trend = True
        
        # === CONDITION 2: MARKET STRUCTURE ===
        structure = self._detect_market_structure(df_m15)
        if structure == "HH_HL" and detected_direction == "LONG":
            score.market_structure_score = 1
            score.market_structure_detail = "M15 forming Higher High + Higher Low (Uptrend)"
        elif structure == "LL_LH" and detected_direction == "SHORT":
            score.market_structure_score = 1
            score.market_structure_detail = "M15 forming Lower Low + Lower High (Downtrend)"
        else:
            score.market_structure_detail = f"M15 structure: {structure}"
        
        # === CONDITION 3: SFP/SWEEP ===
        if setup_type in ["SFP", "LIQUIDITY_SWEEP", "LIQ_SWEEP"]:
            score.sfp_sweep_score = 1
            score.trigger_detail = f"Trigger: {setup_type} detected"
        
        # === CONDITION 4: RETEST ZONE ===
        in_ob_zone = indicators.get('in_ob_zone', False)
        in_fib_zone, fib_zone_type = self._check_fib_zone(df_m15, detected_direction)
        
        if in_ob_zone or in_fib_zone:
            score.retest_zone_score = 1
            if in_ob_zone:
                score.trigger_detail += " + Order Block Zone"
            if in_fib_zone:
                if fib_zone_type == "golden":
                    score.trigger_detail += " + Fib 0.618-0.786 Golden Pocket"
                else:
                    score.trigger_detail += " + Fib 0.5-0.618 Zone"
        
        if setup_type in ["BREAKER_RETEST", "EMA_PULLBACK", "SILVER_BULLET", "UNICORN", "TURTLE_SOUP"]:
            score.retest_zone_score = 1
            score.trigger_detail = f"Trigger: {setup_type} detected"
        
        # === CONDITION 5: RSI/WAVETREND ===
        rsi = indicators.get('rsi_15m', 50)
        rsi_div = indicators.get('rsi_divergence', 'None')
        wt1 = indicators.get('wt1', 0)
        wt2 = indicators.get('wt2', 0)
        wt_cross = indicators.get('wt_signal', 'Neutral')
        
        rsi_ok = False
        wt_ok = False
        
        if detected_direction == "LONG":
            if rsi < self.rsi_oversold or rsi_div == "Bullish":
                rsi_ok = True
            if wt_cross == "Bullish Cross" or (wt1 < -60 and wt1 > wt2):
                wt_ok = True
        elif detected_direction == "SHORT":
            if rsi > self.rsi_overbought or rsi_div == "Bearish":
                rsi_ok = True
            if wt_cross == "Bearish Cross" or (wt1 > 60 and wt1 < wt2):
                wt_ok = True
        
        if rsi_ok or wt_ok:
            score.rsi_wavetrend_score = 1
            details = []
            if rsi_ok:
                details.append(f"RSI={rsi:.0f}")
                if rsi_div != "None":
                    details.append(f"Div={rsi_div}")
            if wt_ok:
                details.append(f"WT Cross")
            score.momentum_detail = " + ".join(details)
        
        # === CONDITION 6: VOLUME SPIKE ===
        volume_ratio = indicators.get('volume_ratio', 1.0)
        if volume_ratio >= 1.5:
            score.volume_spike_score = 1
            score.momentum_detail += f" + Vol x{volume_ratio:.1f}"
        
        # Set direction
        score.suggested_direction = detected_direction or "NONE"
        
        # Calculate new confidence score
        swing_high_20 = float(df_m15['high'].tail(20).max())
        swing_low_20 = float(df_m15['low'].tail(20).min())
        
        confidence = self.calculate_confidence(
            df_m15=df_m15,
            df_h1=df_h1,
            indicators=indicators,
            setup_type=setup_type,
            detected_direction=detected_direction,
            has_sfp=setup_type in ["SFP", "LIQUIDITY_SWEEP", "LIQ_SWEEP"],
            has_rsi_divergence=rsi_div in ["Bullish", "Bearish"],
            has_volume_spike=volume_ratio >= 2.0,
            has_wavetrend_cross=wt_cross in ["Bullish Cross", "Bearish Cross"],
            has_ob_confluence=in_ob_zone,
            swing_high_20=swing_high_20,
            swing_low_20=swing_low_20
        )
        
        score._confidence_score = confidence
        
        return score
    
    def evaluate_4layer_short(
        self,
        df: pd.DataFrame,
        indicators: Dict
    ) -> FourLayerResult:
        """Legacy 4-Layer Filter for Short signals."""
        result = FourLayerResult()
        
        if df.empty or len(df) < 21:
            return result
        
        # Current candle
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        high = float(curr['high'])
        low = float(curr['low'])
        open_p = float(curr['open'])
        close = float(curr['close'])
        volume = float(curr['volume'])
        
        total_range = high - low
        if total_range == 0:
            return result
        
        # === LAYER 1: CONTEXT FILTER ===
        rsi = indicators.get('rsi_15m', 50)
        bb_upper = indicators.get('bb_upper', high)
        
        result.rsi_value = rsi
        
        if high > bb_upper:
            result.layer1_pass = True
            result.is_overextended = True
            result.bb_position = "Above Upper Band"
            result.layer1_reason = f"Price above BB Upper"
        elif rsi > self.rsi_overbought:
            result.layer1_pass = True
            result.is_overextended = True
            result.layer1_reason = f"RSI overbought ({rsi:.0f} > 75)"
        else:
            result.layer1_reason = f"Price not yet overextended"
        
        # === LAYER 2: CANDLE ANATOMY ===
        upper_wick = high - max(open_p, close)
        body = abs(close - open_p)
        
        upper_wick_pct = upper_wick / total_range
        body_pct = body / total_range
        close_position_pct = (close - low) / total_range
        
        result.upper_wick_pct = upper_wick_pct
        result.body_pct = body_pct
        result.close_position_pct = close_position_pct
        
        is_shooting_star = (
            upper_wick_pct >= self.shooting_star_wick_min and
            body_pct <= self.shooting_star_body_max and
            close_position_pct <= self.shooting_star_close_max
        )
        
        result.is_shooting_star = is_shooting_star
        
        if is_shooting_star:
            result.layer2_pass = True
            result.layer2_reason = f"Shooting Star (Wick {upper_wick_pct:.0%})"
        else:
            result.layer2_reason = f"Not a Shooting Star"
        
        # === LAYER 3: VOLUME CONFIRMATION ===
        volumes = df['volume'].tail(21).values
        avg_vol_20 = float(np.mean(volumes[:-1]))
        
        vol_ratio = volume / avg_vol_20 if avg_vol_20 > 0 else 1.0
        result.volume_ratio = vol_ratio
        
        if vol_ratio >= self.short_volume_ratio:
            result.layer3_pass = True
            result.layer3_reason = f"Volume Spike x{vol_ratio:.1f}"
        else:
            result.layer3_reason = f"Low volume x{vol_ratio:.1f}"
        
        # === LAYER 4: SAFETY CHECK ===
        prev_open = float(prev['open'])
        prev_close = float(prev['close'])
        prev_high = float(prev['high'])
        prev_low = float(prev['low'])
        
        prev_range = prev_high - prev_low
        prev_body = abs(prev_close - prev_open)
        prev_body_pct = prev_body / prev_range if prev_range > 0 else 0
        prev_is_green = prev_close > prev_open
        
        result.prev_candle_green = prev_is_green
        result.prev_candle_body_pct = prev_body_pct
        
        if prev_is_green and prev_body_pct >= self.prev_candle_body_min:
            result.layer4_pass = True
            result.layer4_reason = f"Previous candle: Green body {prev_body_pct:.0%}"
        else:
            result.layer4_reason = f"Previous candle does not meet criteria"
        
        return result
    
    def _detect_market_structure(self, df: pd.DataFrame) -> str:
        """Detect Market Structure on M15."""
        if len(df) < 20:
            return "NEUTRAL"
        
        highs = df['high'].tail(20).values
        lows = df['low'].tail(20).values
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i-2] and \
               lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                swing_lows.append(lows[i])
        
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return "NEUTRAL"
        
        if swing_highs[-1] > swing_highs[-2] and swing_lows[-1] > swing_lows[-2]:
            return "HH_HL"
        
        if swing_lows[-1] < swing_lows[-2] and swing_highs[-1] < swing_highs[-2]:
            return "LL_LH"
        
        return "NEUTRAL"
    
    def _check_fib_zone(self, df: pd.DataFrame, direction: str) -> Tuple[bool, str]:
        """
        Check if price is in a Fib retracement zone.
        
        Returns:
            (is_in_zone, zone_type)
            zone_type: 'golden' (0.618-0.786), 'standard' (0.5-0.618), or ''
        """
        if len(df) < 20 or not direction:
            return False, ""
        
        high_20 = float(df['high'].tail(20).max())
        low_20 = float(df['low'].tail(20).min())
        price = float(df['close'].iloc[-1])
        
        fib_range = high_20 - low_20
        if fib_range <= 0:
            return False, ""
        
        if direction == "LONG":
            # LONG: Looking for support levels (price pulled back from high)
            fib_50 = high_20 - fib_range * 0.50
            fib_618 = high_20 - fib_range * 0.618
            fib_786 = high_20 - fib_range * 0.786
            
            # Golden pocket (stronger): 0.618-0.786
            if fib_786 <= price <= fib_618:
                return True, "golden"
            # Standard zone: 0.5-0.618
            elif fib_618 <= price <= fib_50:
                return True, "standard"
        else:
            # SHORT: Looking for resistance levels (price bounced from low)
            fib_50 = low_20 + fib_range * 0.50
            fib_618 = low_20 + fib_range * 0.618
            fib_786 = low_20 + fib_range * 0.786
            
            # Golden pocket (stronger): 0.618-0.786
            if fib_618 <= price <= fib_786:
                return True, "golden"
            # Standard zone: 0.5-0.618
            elif fib_50 <= price <= fib_618:
                return True, "standard"
        
        return False, ""
    
    def get_combined_score(
        self,
        checklist: ChecklistScore,
        four_layer: Optional[FourLayerResult] = None
    ) -> Tuple[SignalGrade, float, List[str]]:
        """Legacy combined score method."""
        reasons = []
        
        grade = checklist.grade
        vol_weight = checklist.volume_weight
        
        if checklist.ema_trend_score > 0:
            reasons.append(f"✓ {checklist.ema_trend_detail}")
        if checklist.market_structure_score > 0:
            reasons.append(f"✓ {checklist.market_structure_detail}")
        if checklist.trigger_detail:
            reasons.append(f"✓ {checklist.trigger_detail}")
        if checklist.momentum_detail:
            reasons.append(f"✓ {checklist.momentum_detail}")
        
        if four_layer and checklist.suggested_direction == "SHORT":
            if four_layer.is_valid_short:
                reasons.append(f"✓ 4-Layer Filter: {four_layer.layers_passed}/4 passed")
                if grade == SignalGrade.B_SCALP:
                    grade = SignalGrade.A_SNIPER
                    vol_weight = 1.0
            else:
                reasons.append(f"⚠ 4-Layer Filter: {four_layer.layers_passed}/4 (weak)")
                if grade == SignalGrade.A_SNIPER:
                    grade = SignalGrade.B_SCALP
                    vol_weight = 0.5
        
        if checklist.is_counter_trend:
            reasons.append("⚠ COUNTER-TREND: Trading against H1 trend")
        
        # Add new tier info
        if checklist._confidence_score:
            reasons.append(f"📊 Confidence: {checklist.confidence_points}/100 → {checklist.tier.value}")
        
        return grade, vol_weight, reasons


# Singleton instance
scoring_system = ScoringSystem()
