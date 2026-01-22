"""
Risk Manager v1.0 - Complete Risk Management System

Features:
1. BTC Correlation Filter - Block LONG when BTC dumping
2. Dynamic Position Sizing - Risk-based sizing
3. Circuit Breaker - Daily/Weekly max loss protection
4. Signal Invalidation Tracker - Cancel alerts when setup breaks
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# ENUMS & CONSTANTS
# ═══════════════════════════════════════════════════════════════════

class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "CLOSED"     # Normal operation
    OPEN = "OPEN"         # Paused - max loss reached
    HALF_OPEN = "HALF_OPEN"  # Testing recovery


class BTCState(Enum):
    """BTC market state."""
    BULLISH = "BULLISH"      # BTC up, safe for LONG
    BEARISH = "BEARISH"      # BTC down, block LONG altcoin
    SIDEWAYS = "SIDEWAYS"    # Neutral


class MarketRegime(Enum):
    """Market regime for adaptive trading."""
    BULLISH = "BULLISH"    # Price > EMA200 H4 - SHORT only Diamond
    BEARISH = "BEARISH"    # Price < EMA200 H4 - SHORT can be Gold
    NEUTRAL = "NEUTRAL"    # Near EMA200 - normal rules


# Risk thresholds
DAILY_MAX_LOSS_PCT = 3.0      # 3% daily max loss -> PAUSE 24h
WEEKLY_MAX_LOSS_PCT = 10.0    # 10% weekly max loss -> STOP
BTC_DUMP_THRESHOLD_1H = -1.0  # -1% in 1h = dumping


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BTCCorrelationResult:
    """Result of BTC correlation check."""
    is_safe: bool
    btc_trend: str
    btc_change_1h: float
    btc_change_4h: float
    btc_ema_position: str  # "ABOVE" or "BELOW" EMAs
    reason: str
    
    
@dataclass
class PositionSizeResult:
    """Result of dynamic position sizing."""
    position_size_usd: float
    margin_usd: float
    risk_usd: float
    sl_distance_pct: float
    leverage: int
    risk_reward: float
    is_valid: bool
    reason: str


@dataclass
class CircuitBreakerStatus:
    """Circuit breaker status."""
    state: CircuitBreakerState
    daily_pnl_pct: float
    weekly_pnl_pct: float
    is_trading_allowed: bool
    resume_at: Optional[datetime] = None
    reason: str = ""


@dataclass
class TrackedSignal:
    """A signal being tracked for invalidation."""
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    swing_invalidation: float  # Price that invalidates setup
    created_at: datetime
    expires_at: datetime
    is_invalidated: bool = False
    invalidation_reason: str = ""


# ═══════════════════════════════════════════════════════════════════
# BTC CORRELATION FILTER
# ═══════════════════════════════════════════════════════════════════

class BTCCorrelationFilter:
    """
    BTC Correlation Filter - Block LONG altcoin when BTC dumping.
    
    Rules:
    1. BTC H1 Trend DOWN + BTC Change 1H < -1% → BLOCK ALL LONG
    2. BTC below EMA89 H4 → CAUTION for LONG
    3. Only affects altcoins, majors (BTC/ETH) follow own trend
    """
    
    def __init__(self):
        self.btc_change_1h: float = 0.0
        self.btc_change_4h: float = 0.0
        self.btc_price: float = 0.0
        self.btc_ema34_h4: float = 0.0
        self.btc_ema89_h4: float = 0.0
        self.btc_ema200_h4: float = 0.0  # For Market Regime detection
        self.btc_trend: BTCState = BTCState.SIDEWAYS
        self.market_regime: MarketRegime = MarketRegime.NEUTRAL
        self.last_update: Optional[datetime] = None
        
        # Major coins exempt from BTC filter
        self.major_coins = {"BTC-USDT", "ETH-USDT", "SOL-USDT"}
    
    def update_btc_state(
        self,
        price: float,
        change_1h: float,
        change_4h: float,
        ema34_h4: float,
        ema89_h4: float,
        ema200_h4: float = 0.0  # Optional EMA200
    ):
        """Update BTC state from context manager."""
        self.btc_price = price
        self.btc_change_1h = change_1h
        self.btc_change_4h = change_4h
        self.btc_ema34_h4 = ema34_h4
        self.btc_ema89_h4 = ema89_h4
        self.btc_ema200_h4 = ema200_h4 if ema200_h4 > 0 else ema89_h4  # Fallback to EMA89
        self.last_update = datetime.now()
        
        # Determine BTC trend
        if price > ema34_h4 > ema89_h4 and change_1h >= 0:
            self.btc_trend = BTCState.BULLISH
        elif price < ema89_h4 or change_1h < BTC_DUMP_THRESHOLD_1H:
            self.btc_trend = BTCState.BEARISH
        else:
            self.btc_trend = BTCState.SIDEWAYS
        
        # === MARKET REGIME DETECTION (based on EMA200 H4) ===
        # This affects SHORT trading rules
        ema200_buffer = self.btc_ema200_h4 * 0.01  # 1% buffer zone
        
        if price > self.btc_ema200_h4 + ema200_buffer:
            self.market_regime = MarketRegime.BULLISH
        elif price < self.btc_ema200_h4 - ema200_buffer:
            self.market_regime = MarketRegime.BEARISH
        else:
            self.market_regime = MarketRegime.NEUTRAL
        
        logger.debug(f"📊 BTC State: {self.btc_trend.value}, Regime: {self.market_regime.value}, "
                    f"Price: ${price:,.0f}, EMA200: ${self.btc_ema200_h4:,.0f}")
    
    def get_short_allowed_tier(self) -> str:
        """
        Get minimum tier allowed for SHORT based on market regime.
        
        ADAPTIVE SHORT RULES:
        - BULLISH (Price > EMA200): SHORT only Diamond (Sniper mode)
        - BEARISH (Price < EMA200): SHORT can be Gold (Scalp mode)
        - NEUTRAL: SHORT only Diamond (conservative)
        
        Returns:
            "DIAMOND" or "GOLD"
        """
        if self.market_regime == MarketRegime.BEARISH:
            return "GOLD"  # Allow Gold tier SHORTs in bear market
        else:
            return "DIAMOND"  # Sniper mode - only Diamond SHORTs
    
    def check_correlation(
        self,
        symbol: str,
        direction: str
    ) -> BTCCorrelationResult:
        """
        Check if trade is allowed based on BTC correlation.
        
        Args:
            symbol: Trading pair (e.g., "PEPE-USDT")
            direction: "LONG" or "SHORT"
            
        Returns:
            BTCCorrelationResult with safety status
        """
        # Major coins exempt
        if symbol in self.major_coins:
            return BTCCorrelationResult(
                is_safe=True,
                btc_trend=self.btc_trend.value,
                btc_change_1h=self.btc_change_1h,
                btc_change_4h=self.btc_change_4h,
                btc_ema_position="ABOVE" if self.btc_price > self.btc_ema89_h4 else "BELOW",
                reason="Major coin - follows own trend"
            )
        
        # Check context freshness (max 2 minutes old)
        if self.last_update is None:
            return BTCCorrelationResult(
                is_safe=False,
                btc_trend="UNKNOWN",
                btc_change_1h=0,
                btc_change_4h=0,
                btc_ema_position="UNKNOWN",
                reason="❌ BTC context not available"
            )
        
        age_seconds = (datetime.now() - self.last_update).total_seconds()
        if age_seconds > 120:  # 2 minutes
            return BTCCorrelationResult(
                is_safe=False,
                btc_trend=self.btc_trend.value,
                btc_change_1h=self.btc_change_1h,
                btc_change_4h=self.btc_change_4h,
                btc_ema_position="STALE",
                reason=f"❌ BTC context stale ({age_seconds:.0f}s old)"
            )
        
        ema_position = "ABOVE" if self.btc_price > self.btc_ema89_h4 else "BELOW"
        
        # Rule 1: Block LONG when BTC dumping
        if direction == "LONG":
            if self.btc_trend == BTCState.BEARISH:
                return BTCCorrelationResult(
                    is_safe=False,
                    btc_trend=self.btc_trend.value,
                    btc_change_1h=self.btc_change_1h,
                    btc_change_4h=self.btc_change_4h,
                    btc_ema_position=ema_position,
                    reason=f"🚫 BTC DUMP ({self.btc_change_1h:+.2f}% 1H) - LONG blocked"
                )
            
            # Caution if BTC below EMA89
            if self.btc_price < self.btc_ema89_h4:
                return BTCCorrelationResult(
                    is_safe=True,  # Allow but warn
                    btc_trend=self.btc_trend.value,
                    btc_change_1h=self.btc_change_1h,
                    btc_change_4h=self.btc_change_4h,
                    btc_ema_position=ema_position,
                    reason=f"⚠️ BTC below EMA89 - LONG cautious"
                )
        
        # All other cases are safe
        return BTCCorrelationResult(
            is_safe=True,
            btc_trend=self.btc_trend.value,
            btc_change_1h=self.btc_change_1h,
            btc_change_4h=self.btc_change_4h,
            btc_ema_position=ema_position,
            reason="✅ BTC correlation OK"
        )


# ═══════════════════════════════════════════════════════════════════
# DYNAMIC POSITION SIZING
# ═══════════════════════════════════════════════════════════════════

class DynamicPositionSizer:
    """
    Dynamic Position Sizing based on risk.
    
    Formula: Position Size = (Account * Risk%) / SL Distance%
    
    Example:
    - Account: $1000, Risk: 1% ($10)
    - SL Distance: 5% → Position = $10 / 5% = $200
    - SL Distance: 10% → Position = $10 / 10% = $100
    """
    
    def __init__(
        self,
        account_balance: float = 1000.0,
        risk_per_trade_pct: float = 1.0,  # 1% risk per trade
        max_position_pct: float = 20.0,   # Max 20% of account per position
        min_position_usd: float = 5.0,    # Minimum $5 position
    ):
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.min_position_usd = min_position_usd
    
    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.account_balance = new_balance
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        leverage: int = 10,
        tier_volume_weight: float = 1.0
    ) -> PositionSizeResult:
        """
        Calculate optimal position size based on risk.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            leverage: Leverage to use
            tier_volume_weight: 1.0 for Diamond, 0.7 for Gold
            
        Returns:
            PositionSizeResult with position sizing details
        """
        # Calculate SL distance %
        sl_distance = abs(entry_price - stop_loss)
        sl_distance_pct = (sl_distance / entry_price) * 100
        
        if sl_distance_pct <= 0:
            return PositionSizeResult(
                position_size_usd=0,
                margin_usd=0,
                risk_usd=0,
                sl_distance_pct=0,
                leverage=leverage,
                risk_reward=0,
                is_valid=False,
                reason="Invalid SL distance"
            )
        
        # Calculate risk amount
        risk_usd = self.account_balance * (self.risk_per_trade_pct / 100)
        risk_usd *= tier_volume_weight  # Adjust for tier
        
        # Calculate position size: Position = Risk / (SL% / 100)
        # But with leverage, we need: Position = Risk / (SL% / leverage)
        # Because actual loss = Position * SL% / leverage
        position_size_usd = risk_usd / (sl_distance_pct / 100)
        
        # Apply leverage effect on margin
        margin_usd = position_size_usd / leverage
        
        # Cap at max position
        max_position = self.account_balance * (self.max_position_pct / 100) * leverage
        if position_size_usd > max_position:
            position_size_usd = max_position
            margin_usd = position_size_usd / leverage
            risk_usd = position_size_usd * (sl_distance_pct / 100)
        
        # Minimum check
        if position_size_usd < self.min_position_usd:
            return PositionSizeResult(
                position_size_usd=self.min_position_usd,
                margin_usd=self.min_position_usd / leverage,
                risk_usd=self.min_position_usd * (sl_distance_pct / 100),
                sl_distance_pct=sl_distance_pct,
                leverage=leverage,
                risk_reward=0,
                is_valid=True,
                reason=f"Min position applied (${self.min_position_usd})"
            )
        
        return PositionSizeResult(
            position_size_usd=round(position_size_usd, 2),
            margin_usd=round(margin_usd, 2),
            risk_usd=round(risk_usd, 2),
            sl_distance_pct=round(sl_distance_pct, 4),
            leverage=leverage,
            risk_reward=0,  # Will be calculated later with TP
            is_valid=True,
            reason=f"Risk-based sizing: ${risk_usd:.2f} risk for {sl_distance_pct:.2f}% SL"
        )


# ═══════════════════════════════════════════════════════════════════
# CIRCUIT BREAKER
# ═══════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Circuit Breaker - Stop trading when max loss reached.
    
    Rules:
    1. Daily Max Loss > 3% → PAUSE 24 hours
    2. Weekly Max Loss > 10% → STOP (manual intervention)
    """
    
    def __init__(
        self,
        daily_max_loss_pct: float = DAILY_MAX_LOSS_PCT,
        weekly_max_loss_pct: float = WEEKLY_MAX_LOSS_PCT,
    ):
        self.daily_max_loss_pct = daily_max_loss_pct
        self.weekly_max_loss_pct = weekly_max_loss_pct
        
        self.state = CircuitBreakerState.CLOSED
        self.daily_pnl_pct: float = 0.0
        self.weekly_pnl_pct: float = 0.0
        self.resume_at: Optional[datetime] = None
        
        # PnL tracking
        self.daily_trades: List[float] = []  # PnL per trade today
        self.weekly_trades: List[float] = []  # PnL per trade this week
        self.last_daily_reset: datetime = datetime.now().replace(hour=0, minute=0, second=0)
        self.last_weekly_reset: datetime = datetime.now()
    
    def record_trade_pnl(self, pnl_pct: float):
        """Record trade PnL for tracking."""
        self._check_reset()
        
        self.daily_trades.append(pnl_pct)
        self.weekly_trades.append(pnl_pct)
        
        self.daily_pnl_pct = sum(self.daily_trades)
        self.weekly_pnl_pct = sum(self.weekly_trades)
        
        # Check thresholds
        self._check_thresholds()
    
    def _check_reset(self):
        """Check if daily/weekly counters need reset."""
        now = datetime.now()
        
        # Daily reset at midnight
        today_midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
        if self.last_daily_reset < today_midnight:
            self.daily_trades = []
            self.daily_pnl_pct = 0.0
            self.last_daily_reset = today_midnight
            
            # Auto-resume after 24h pause
            if self.state == CircuitBreakerState.OPEN and self.resume_at:
                if now >= self.resume_at:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info("🟡 Circuit Breaker: HALF-OPEN - testing recovery")
        
        # Weekly reset on Monday
        days_since_monday = now.weekday()
        monday_this_week = now - timedelta(days=days_since_monday)
        monday_this_week = monday_this_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        if self.last_weekly_reset < monday_this_week:
            self.weekly_trades = []
            self.weekly_pnl_pct = 0.0
            self.last_weekly_reset = monday_this_week
    
    def _check_thresholds(self):
        """Check if circuit breaker should trip."""
        # Weekly threshold - STOP completely
        if self.weekly_pnl_pct <= -self.weekly_max_loss_pct:
            self.state = CircuitBreakerState.OPEN
            self.resume_at = None  # Manual intervention required
            logger.warning(
                f"🔴 CIRCUIT BREAKER OPEN: Weekly loss {self.weekly_pnl_pct:.2f}% "
                f"exceeded {self.weekly_max_loss_pct}% - MANUAL INTERVENTION REQUIRED"
            )
            return
        
        # Daily threshold - PAUSE 24h
        if self.daily_pnl_pct <= -self.daily_max_loss_pct:
            self.state = CircuitBreakerState.OPEN
            self.resume_at = datetime.now() + timedelta(hours=24)
            logger.warning(
                f"🟠 CIRCUIT BREAKER OPEN: Daily loss {self.daily_pnl_pct:.2f}% "
                f"exceeded {self.daily_max_loss_pct}% - PAUSED until {self.resume_at}"
            )
            return
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        self._check_reset()
        return self.state != CircuitBreakerState.OPEN
    
    def get_status(self) -> CircuitBreakerStatus:
        """Get circuit breaker status."""
        self._check_reset()
        
        return CircuitBreakerStatus(
            state=self.state,
            daily_pnl_pct=self.daily_pnl_pct,
            weekly_pnl_pct=self.weekly_pnl_pct,
            is_trading_allowed=self.is_trading_allowed(),
            resume_at=self.resume_at,
            reason=self._get_status_reason()
        )
    
    def _get_status_reason(self) -> str:
        """Get human-readable status reason."""
        if self.state == CircuitBreakerState.CLOSED:
            return f"✅ Trading OK (Daily: {self.daily_pnl_pct:+.2f}%, Weekly: {self.weekly_pnl_pct:+.2f}%)"
        elif self.resume_at:
            return f"⏸️ PAUSED until {self.resume_at.strftime('%Y-%m-%d %H:%M')}"
        else:
            return "🔴 STOPPED - Manual intervention required"
    
    def force_resume(self):
        """Force resume trading (admin action)."""
        self.state = CircuitBreakerState.CLOSED
        self.resume_at = None
        logger.info("🟢 Circuit Breaker: FORCE RESUMED by admin")


# ═══════════════════════════════════════════════════════════════════
# SIGNAL INVALIDATION TRACKER
# ═══════════════════════════════════════════════════════════════════

class InvalidationTracker:
    """
    Track active signals and detect invalidation.
    
    When a signal is sent:
    1. Add to tracking list with entry, SL, invalidation level
    2. Monitor price for 15-30 minutes
    3. If SL hit or structure broken → Send CANCEL message
    """
    
    def __init__(self, validity_minutes: int = 30):
        self.validity_minutes = validity_minutes
        self.active_signals: Dict[str, TrackedSignal] = {}
    
    def add_signal(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        swing_invalidation: Optional[float] = None
    ) -> str:
        """
        Add a signal to tracking.
        
        Args:
            symbol: Trading pair
            direction: LONG/SHORT
            entry_price: Entry price
            stop_loss: Stop loss price
            swing_invalidation: Price that breaks the setup structure
            
        Returns:
            Signal ID
        """
        signal_id = f"{symbol}_{direction}_{datetime.now().strftime('%H%M%S')}"
        
        # For SHORT, invalidation is above entry; for LONG, below entry
        if swing_invalidation is None:
            if direction == "LONG":
                swing_invalidation = stop_loss * 0.99  # Slightly below SL
            else:
                swing_invalidation = stop_loss * 1.01  # Slightly above SL
        
        signal = TrackedSignal(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            swing_invalidation=swing_invalidation,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(minutes=self.validity_minutes),
            is_invalidated=False
        )
        
        self.active_signals[signal_id] = signal
        logger.debug(f"Tracking signal: {signal_id}")
        
        return signal_id
    
    def check_invalidation(
        self,
        symbol: str,
        current_price: float
    ) -> List[Tuple[str, TrackedSignal]]:
        """
        Check if any signals for symbol are invalidated.
        
        Returns:
            List of (signal_id, signal) that are invalidated
        """
        invalidated = []
        now = datetime.now()
        
        for signal_id, signal in list(self.active_signals.items()):
            if signal.symbol != symbol:
                continue
            
            # Check if expired
            if now > signal.expires_at:
                del self.active_signals[signal_id]
                continue
            
            # Check if already invalidated
            if signal.is_invalidated:
                continue
            
            # Check SL hit
            if signal.direction == "LONG" and current_price <= signal.stop_loss:
                signal.is_invalidated = True
                signal.invalidation_reason = f"SL HIT: Price {current_price:.6f} <= SL {signal.stop_loss:.6f}"
                invalidated.append((signal_id, signal))
                
            elif signal.direction == "SHORT" and current_price >= signal.stop_loss:
                signal.is_invalidated = True
                signal.invalidation_reason = f"SL HIT: Price {current_price:.6f} >= SL {signal.stop_loss:.6f}"
                invalidated.append((signal_id, signal))
            
            # Check structure break
            elif signal.direction == "LONG" and current_price < signal.swing_invalidation:
                signal.is_invalidated = True
                signal.invalidation_reason = f"STRUCTURE BROKEN: Price below {signal.swing_invalidation:.6f}"
                invalidated.append((signal_id, signal))
                
            elif signal.direction == "SHORT" and current_price > signal.swing_invalidation:
                signal.is_invalidated = True
                signal.invalidation_reason = f"STRUCTURE BROKEN: Price above {signal.swing_invalidation:.6f}"
                invalidated.append((signal_id, signal))
        
        return invalidated
    
    def remove_signal(self, signal_id: str):
        """Remove a signal from tracking."""
        if signal_id in self.active_signals:
            del self.active_signals[signal_id]
    
    def cleanup_expired(self):
        """Remove expired signals."""
        now = datetime.now()
        expired = [
            sid for sid, sig in self.active_signals.items()
            if now > sig.expires_at
        ]
        for sid in expired:
            del self.active_signals[sid]
        
        return len(expired)
    
    def get_active_count(self) -> int:
        """Get count of active signals."""
        self.cleanup_expired()
        return len(self.active_signals)


# ═══════════════════════════════════════════════════════════════════
# UNIFIED RISK MANAGER
# ═══════════════════════════════════════════════════════════════════

class RiskManager:
    """
    Unified Risk Manager combining all risk management features.
    """
    
    def __init__(
        self,
        account_balance: float = 1000.0,
        risk_per_trade_pct: float = 1.0,
    ):
        self.btc_filter = BTCCorrelationFilter()
        self.position_sizer = DynamicPositionSizer(
            account_balance=account_balance,
            risk_per_trade_pct=risk_per_trade_pct
        )
        self.circuit_breaker = CircuitBreaker()
        self.invalidation_tracker = InvalidationTracker()
    
    def update_btc_state(
        self,
        price: float,
        change_1h: float,
        change_4h: float,
        ema34_h4: float,
        ema89_h4: float,
        ema200_h4: float = 0.0
    ):
        """Update BTC state for correlation filter and market regime."""
        self.btc_filter.update_btc_state(
            price, change_1h, change_4h, ema34_h4, ema89_h4, ema200_h4
        )
    
    def update_balance(self, new_balance: float):
        """Update account balance."""
        self.position_sizer.update_balance(new_balance)
    
    def can_trade(
        self,
        symbol: str,
        direction: str,
    ) -> Tuple[bool, str]:
        """
        Check if trade is allowed considering all risk factors.
        
        Returns:
            (is_allowed, reason)
        """
        # Check circuit breaker first
        if not self.circuit_breaker.is_trading_allowed():
            status = self.circuit_breaker.get_status()
            return False, status.reason
        
        # Check BTC correlation
        btc_result = self.btc_filter.check_correlation(symbol, direction)
        if not btc_result.is_safe:
            return False, btc_result.reason
        
        return True, "✅ All risk checks passed"
    
    def calculate_position(
        self,
        entry_price: float,
        stop_loss: float,
        leverage: int = 10,
        tier_weight: float = 1.0
    ) -> PositionSizeResult:
        """Calculate dynamic position size."""
        return self.position_sizer.calculate_position_size(
            entry_price, stop_loss, leverage, tier_weight
        )
    
    def record_trade_result(self, pnl_pct: float):
        """Record trade result for circuit breaker."""
        self.circuit_breaker.record_trade_pnl(pnl_pct)
    
    def track_signal(
        self,
        symbol: str,
        direction: str,
        entry: float,
        stop_loss: float,
        swing_invalidation: Optional[float] = None
    ) -> str:
        """Track a new signal for invalidation."""
        return self.invalidation_tracker.add_signal(
            symbol, direction, entry, stop_loss, swing_invalidation
        )
    
    def check_invalidations(
        self,
        symbol: str,
        current_price: float
    ) -> List[Tuple[str, TrackedSignal]]:
        """Check for signal invalidations."""
        return self.invalidation_tracker.check_invalidation(symbol, current_price)
    
    def get_status(self) -> dict:
        """Get comprehensive risk status."""
        cb_status = self.circuit_breaker.get_status()
        
        return {
            "circuit_breaker": {
                "state": cb_status.state.value,
                "daily_pnl": f"{cb_status.daily_pnl_pct:+.2f}%",
                "weekly_pnl": f"{cb_status.weekly_pnl_pct:+.2f}%",
                "trading_allowed": cb_status.is_trading_allowed
            },
            "btc_filter": {
                "trend": self.btc_filter.btc_trend.value,
                "change_1h": f"{self.btc_filter.btc_change_1h:+.2f}%",
                "price": self.btc_filter.btc_price
            },
            "position_sizer": {
                "balance": f"${self.position_sizer.account_balance:.2f}",
                "risk_per_trade": f"{self.position_sizer.risk_per_trade_pct}%"
            },
            "active_signals": self.invalidation_tracker.get_active_count()
        }
