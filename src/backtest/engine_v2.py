"""
Backtesting Engine v2.0 - Full Signal Simulation

IMPROVEMENTS over v1.0:
═══════════════════════════════════════════════════════════════════
1. Full Scoring System v2.1 integration (not random signals)
2. BTC Correlation Filter (block LONG when BTC dumps >0.5%)
3. 4-Layer Filter for SHORT (BB Upper/RSI>75 + Shooting Star + Vol + Prev Green)
4. Dynamic ATR-based TP/SL (not static 2%/4%/6%)
5. Multi-timeframe analysis (15m signals + H1 trend)
6. Strategy-specific testing
7. Drawdown calculation
8. Slippage & fees simulation

Author: BingX Alert Bot
Version: 2.0
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════

# Tier thresholds (from scoring_system.py)
THRESHOLD_DIAMOND = 80
THRESHOLD_GOLD = 60
THRESHOLD_SILVER = 45

# BTC Correlation Filter
BTC_DUMP_THRESHOLD = -0.5  # Block LONG if BTC drops >0.5% in 15m

# Fees & Slippage
SLIPPAGE_PERCENT = 0.05     # 0.05% slippage per trade
TRADING_FEE = 0.04          # 0.04% per trade (BingX taker fee)
FUNDING_FEE_PER_8H = 0.01   # 0.01% funding rate per 8h


# ═══════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════════

class TradeResult(Enum):
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    PENDING = "PENDING"


@dataclass
class BacktestTradeV2:
    """Enhanced trade record for backtest v2."""
    symbol: str
    direction: str  # LONG/SHORT
    entry_price: float
    entry_time: datetime
    stoploss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Signal info
    strategy: str
    confidence_score: int
    tier: str  # DIAMOND, GOLD, SILVER, REJECT
    
    # 4-Layer info (for SHORT)
    four_layer_passed: int = 0  # 0-4
    
    # ATR info
    atr_value: float = 0
    
    # Results
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    result: TradeResult = TradeResult.PENDING
    pnl_percent: float = 0
    pnl_usd: float = 0
    hit_level: str = ""  # TP1, TP2, TP3, SL
    
    # Filters applied
    passed_btc_filter: bool = True
    passed_4layer_filter: bool = True
    
    # Breakdown
    breakdown: List[str] = field(default_factory=list)


@dataclass
class BacktestResultV2:
    """Enhanced backtest result."""
    start_date: datetime
    end_date: datetime
    total_candles: int
    total_trades: int
    
    # Win/Loss
    wins: int
    losses: int
    breakevens: int
    winrate: float
    
    # By tier
    diamond_trades: int
    diamond_wins: int
    diamond_winrate: float
    gold_trades: int
    gold_wins: int
    gold_winrate: float
    
    # By direction
    long_trades: int
    long_winrate: float
    short_trades: int
    short_winrate: float
    
    # PnL
    total_pnl_percent: float
    total_pnl_usd: float
    average_win_percent: float
    average_loss_percent: float
    profit_factor: float
    
    # Drawdown
    max_drawdown_percent: float
    max_drawdown_usd: float
    
    # Filters stats
    blocked_by_btc_filter: int
    blocked_by_4layer_filter: int
    blocked_by_cooldown: int
    
    # Strategy breakdown
    strategy_stats: Dict[str, Dict]
    
    # All trades
    trades: List[BacktestTradeV2]


# ═══════════════════════════════════════════════════════════════════
# BACKTEST ENGINE V2
# ═══════════════════════════════════════════════════════════════════

class BacktestEngineV2:
    """
    Enhanced Backtesting Engine with full signal simulation.
    
    Features:
    - Full Scoring System v2.1
    - BTC Correlation Filter
    - 4-Layer SHORT Filter
    - Dynamic ATR-based TP/SL
    - Strategy-specific analysis
    """
    
    def __init__(self, rest_client=None):
        """Initialize backtest engine v2."""
        self.rest_client = rest_client
        self.trades: List[BacktestTradeV2] = []
        
        # Simulation settings
        self.initial_balance = 1000.0
        self.position_size_diamond = 2.0  # $2 for DIAMOND
        self.position_size_gold = 1.0     # $1 for GOLD
        
        # BTC data cache
        self.btc_data: pd.DataFrame = None
        
        # Blocked trades stats
        self.blocked_btc = 0
        self.blocked_4layer = 0
        
    async def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15m"
    ) -> BacktestResultV2:
        """
        Run backtest with full signal simulation.
        
        Args:
            symbols: List of trading pairs
            start_date: Start date
            end_date: End date
            timeframe: Candle timeframe
            
        Returns:
            BacktestResultV2 with complete statistics
        """
        logger.info(f"🔄 Backtest V2: {start_date.date()} to {end_date.date()}")
        logger.info(f"📊 Symbols: {len(symbols)}, Timeframe: {timeframe}")
        
        self.trades = []
        self.blocked_btc = 0
        self.blocked_4layer = 0
        self.blocked_cooldown = 0
        self.cooldown_tracker = {}  # {symbol_direction: last_signal_candle_index}
        total_candles = 0
        
        # 1. Fetch BTC data first for correlation filter
        logger.info("📈 Fetching BTC data for correlation filter...")
        self.btc_data = await self._fetch_symbol_data("BTC-USDT", start_date, end_date, timeframe)
        
        if self.btc_data is None or len(self.btc_data) < 100:
            logger.warning("⚠️ BTC data insufficient - BTC filter will be skipped")
        else:
            # Calculate EMA200 for BTC to determine Market Regime
            self.btc_data['ema200'] = self.btc_data['close'].ewm(span=200, adjust=False).mean()
            # Calculate EMA89 for BTC to determine Market State
            self.btc_data['ema89'] = self.btc_data['close'].ewm(span=89, adjust=False).mean()
            logger.info("✅ BTC EMA89/EMA200 calculated for Market State/Regime detection")
        
        # 2. Process each symbol
        for symbol in symbols:
            candle_count = await self._backtest_symbol(symbol, start_date, end_date, timeframe)
            total_candles += candle_count
            await asyncio.sleep(0.05)  # Rate limit
        
        logger.info(f"📊 Total candles analyzed: {total_candles}")
        
        return self._calculate_results(start_date, end_date, total_candles)
    
    async def _fetch_symbol_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> Optional[pd.DataFrame]:
        """Fetch and convert candle data to DataFrame."""
        if not self.rest_client:
            return None
            
        try:
            candles = await self.rest_client.get_futures_klines_historical(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                max_candles=5000
            )
            
            if not candles or len(candles) < 50:
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(candles)
            
            # Ensure numeric columns
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Parse timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {e}")
            return None
    
    async def _backtest_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> int:
        """Backtest single symbol with full signal detection."""
        logger.info(f"📈 Backtesting {symbol}...")
        
        try:
            # Fetch data
            df = await self._fetch_symbol_data(symbol, start_date, end_date, timeframe)
            
            if df is None or len(df) < 100:
                logger.warning(f"⚠️ Insufficient data for {symbol}")
                return 0
            
            logger.info(f"   {symbol}: {len(df)} candles")
            
            # Calculate indicators
            df = self._calculate_indicators(df)
            
            # Slide through candles
            lookback = 100
            
            for i in range(lookback, len(df) - 100):  # Leave 100 candles for TP/SL simulation
                current_idx = df.index[i]
                window = df.iloc[i-lookback:i+1]
                future = df.iloc[i+1:i+101]  # Next 100 candles for TP/SL check (25 hours)
                
                # Detect signals
                signal = self._detect_signal(symbol, window)
                
                if signal:
                    # === COOLDOWN CHECK (24 candles = 6 hours) ===
                    cooldown_key = f"{symbol}_{signal['direction']}"
                    last_signal_idx = self.cooldown_tracker.get(cooldown_key, -999)
                    if i - last_signal_idx < 24:  # 24 candles = 6 hours cooldown
                        self.blocked_cooldown += 1
                        continue  # Skip this signal
                    
                    # Apply filters
                    passed, filter_reason = self._apply_filters(signal, window, current_idx)
                    
                    if passed:
                        # Create and simulate trade
                        trade = self._create_trade(symbol, window, signal, future)
                        if trade:
                            self.trades.append(trade)
                            # Update cooldown tracker
                            self.cooldown_tracker[cooldown_key] = i
            
            return len(df)
            
        except Exception as e:
            logger.error(f"❌ Backtest error for {symbol}: {e}")
            return 0
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators on DataFrame."""
        try:
            # EMA
            df['ema34'] = df['close'].ewm(span=34, adjust=False).mean()
            df['ema89'] = df['close'].ewm(span=89, adjust=False).mean()
            df['ema200'] = df['close'].ewm(span=200, adjust=False).mean()  # For Market Regime
            
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 0.0001)
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # ATR (14-period)
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=14).mean()
            
            # Volume MA
            df['volume_ma20'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma20'].replace(0, 1)
            
            # Swing High/Low (20 period)
            df['swing_high_20'] = df['high'].rolling(window=20).max()
            df['swing_low_20'] = df['low'].rolling(window=20).min()
            
            # WaveTrend (simplified)
            ap = (df['high'] + df['low'] + df['close']) / 3
            esa = ap.ewm(span=10, adjust=False).mean()
            d = (ap - esa).abs().ewm(span=10, adjust=False).mean()
            ci = (ap - esa) / (0.015 * d.replace(0, 0.0001))
            df['wt1'] = ci.ewm(span=21, adjust=False).mean()
            df['wt2'] = df['wt1'].rolling(window=4).mean()
            
            # Candle patterns
            df['body'] = abs(df['close'] - df['open'])
            df['upper_wick'] = df['high'] - df[['close', 'open']].max(axis=1)
            df['lower_wick'] = df[['close', 'open']].min(axis=1) - df['low']
            df['candle_range'] = df['high'] - df['low']
            
            # Prev candle color
            df['prev_green'] = (df['close'].shift(1) > df['open'].shift(1))
            
            # ADX (Average Directional Index) - 14 period
            plus_dm = df['high'].diff()
            minus_dm = df['low'].diff().abs() * -1
            plus_dm = plus_dm.where((plus_dm > minus_dm.abs()) & (plus_dm > 0), 0)
            minus_dm = minus_dm.abs().where((minus_dm.abs() > plus_dm) & (minus_dm < 0), 0)
            
            tr_adx = pd.concat([df['high'] - df['low'], 
                               (df['high'] - df['close'].shift()).abs(),
                               (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)
            atr_adx = tr_adx.rolling(window=14).mean()
            
            plus_di = 100 * (plus_dm.rolling(window=14).mean() / atr_adx.replace(0, 0.0001))
            minus_di = 100 * (minus_dm.rolling(window=14).mean() / atr_adx.replace(0, 0.0001))
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, 0.0001))
            df['adx'] = dx.rolling(window=14).mean()
            
            # RSI Divergence detection (simplified)
            # Bullish div: Price makes lower low, RSI makes higher low
            # Bearish div: Price makes higher high, RSI makes lower high
            df['price_higher_high'] = (df['high'] > df['high'].shift(5)) & (df['high'].shift(5) > df['high'].shift(10))
            df['price_lower_low'] = (df['low'] < df['low'].shift(5)) & (df['low'].shift(5) < df['low'].shift(10))
            df['rsi_lower_high'] = (df['rsi'] < df['rsi'].shift(5)) & (df['rsi'].shift(5) > df['rsi'].shift(10))
            df['rsi_higher_low'] = (df['rsi'] > df['rsi'].shift(5)) & (df['rsi'].shift(5) < df['rsi'].shift(10))
            df['bearish_div'] = df['price_higher_high'] & df['rsi_lower_high']  # SHORT signal
            df['bullish_div'] = df['price_lower_low'] & df['rsi_higher_low']    # LONG signal
            
        except Exception as e:
            logger.error(f"Indicator calculation error: {e}")
        
        return df
    
    def _detect_signal(self, symbol: str, window: pd.DataFrame) -> Optional[Dict]:
        """
        Detect trading signals using real strategies.
        
        Strategies detected (checked independently, best confidence wins):
        - SFP (Swing Failure Pattern)
        - BB_BOUNCE (Bollinger Band Bounce)
        - PUMP_FADE (Shooting Star / Pump & Dump)
        - LIQ_SWEEP (Liquidity Sweep)
        """
        if len(window) < 20:
            return None
        
        current = window.iloc[-1]
        prev = window.iloc[-2]
        
        price = current['close']
        rsi = current.get('rsi', 50)
        bb_upper = current.get('bb_upper', price * 1.02)
        bb_lower = current.get('bb_lower', price * 0.98)
        swing_high = current.get('swing_high_20', price)
        swing_low = current.get('swing_low_20', price)
        volume_ratio = current.get('volume_ratio', 1.0)
        atr = current.get('atr', price * 0.01)
        wt1 = current.get('wt1', 0)
        wt2 = current.get('wt2', 0)
        
        # Collect all potential signals (check independently)
        candidates = []
        
        # === 1. SFP (Swing Failure Pattern) - Highest priority ===
        # SHORT SFP: Price breaks above swing high then closes below
        if current['high'] > swing_high and current['close'] < swing_high:
            if volume_ratio >= 1.5:  # Need volume confirmation
                conf = 65  # Higher base for SFP (better quality)
                if volume_ratio >= 2:
                    conf += 15
                if rsi > 70:
                    conf += 10
                if wt1 > 60:
                    conf += 5
                candidates.append({
                    'strategy': 'SFP',
                    'direction': 'SHORT',
                    'entry': price,
                    'confidence': conf,
                    'breakdown': [f"SFP: High broke swing {swing_high:.4f}, closed below"]
                })
        
        # LONG SFP: Price breaks below swing low then closes above
        if current['low'] < swing_low and current['close'] > swing_low:
            if volume_ratio >= 1.5:
                conf = 65  # Higher base for SFP
                if volume_ratio >= 2:
                    conf += 15
                if rsi < 30:
                    conf += 10
                if wt1 < -60:
                    conf += 5
                candidates.append({
                    'strategy': 'SFP',
                    'direction': 'LONG',
                    'entry': price,
                    'confidence': conf,
                    'breakdown': [f"SFP: Low broke swing {swing_low:.4f}, closed above"]
                })
        
        # === 2. BB_BOUNCE - DISABLED (0% WR in backtest, not profitable) ===
        # Get candle components for other strategies
        body = current['body']
        upper_wick = current['upper_wick']
        lower_wick = current['lower_wick']
        
        # Reversal candle checks (used by other strategies)
        is_bearish_candle = current['close'] < current['open']
        is_bullish_candle = current['close'] > current['open']
        
        # Get ADX and divergence (may be used later)
        adx = current.get('adx', 20)
        bearish_div = current.get('bearish_div', False)
        bullish_div = current.get('bullish_div', False)
        
        # BB_BOUNCE strategy disabled - was producing 0% WR
        # Keeping code structure for future improvement
        
        # === 3. PUMP_FADE / SHOOTING_STAR ===
        # Re-enabled with stricter conditions
        body = current['body']
        upper_wick = current['upper_wick']
        lower_wick = current['lower_wick']
        candle_range = current['candle_range']
        
        if candle_range > 0:
            upper_wick_pct = upper_wick / candle_range
            lower_wick_pct = lower_wick / candle_range
            body_pct = body / candle_range
            
            # Shooting star for SHORT: Upper wick >= 60%, body <= 30%, RSI > 70
            if upper_wick_pct >= 0.6 and body_pct <= 0.3:
                if volume_ratio >= 2.0 and rsi > 70:
                    conf = 55
                    if rsi > 80:
                        conf += 20
                    if volume_ratio >= 3:
                        conf += 10
                    candidates.append({
                        'strategy': 'PUMP_FADE',
                        'direction': 'SHORT',
                        'entry': price,
                        'confidence': conf,
                        'breakdown': [f"Shooting Star: wick {upper_wick_pct:.0%}, vol {volume_ratio:.1f}x, RSI {rsi:.1f}"]
                    })
            
            # Hammer for LONG: Lower wick >= 60%, body <= 30%, RSI < 30
            if lower_wick_pct >= 0.6 and body_pct <= 0.3:
                if volume_ratio >= 2.0 and rsi < 30:
                    conf = 55
                    if rsi < 20:
                        conf += 20
                    if volume_ratio >= 3:
                        conf += 10
                    candidates.append({
                        'strategy': 'PUMP_FADE',
                        'direction': 'LONG',
                        'entry': price,
                        'confidence': conf,
                        'breakdown': [f"Hammer: wick {lower_wick_pct:.0%}, vol {volume_ratio:.1f}x, RSI {rsi:.1f}"]
                    })
        
        # === 4. LIQ_SWEEP ===
        # Re-enabled with stricter conditions
        # SHORT: Break above prev high, close below with volume
        if current['high'] > prev['high'] * 1.008:
            if current['close'] < prev['high']:
                if volume_ratio >= 2.0 and rsi > 65:
                    conf = 50
                    if volume_ratio >= 2.5:
                        conf += 15
                    if rsi > 75:
                        conf += 10
                    candidates.append({
                        'strategy': 'LIQ_SWEEP',
                        'direction': 'SHORT',
                        'entry': price,
                        'confidence': conf,
                        'breakdown': [f"Liquidity sweep above {prev['high']:.4f}"]
                    })
        
        # LONG: Break below prev low, close above with volume
        if current['low'] < prev['low'] * 0.992:
            if current['close'] > prev['low']:
                if volume_ratio >= 2.0 and rsi < 35:
                    conf = 50
                    if volume_ratio >= 2.5:
                        conf += 15
                    if rsi < 25:
                        conf += 10
                    candidates.append({
                        'strategy': 'LIQ_SWEEP',
                        'direction': 'LONG',
                        'entry': price,
                        'confidence': conf,
                        'breakdown': [f"Liquidity sweep below {prev['low']:.4f}"]
                    })
        
        # === SELECT BEST SIGNAL (highest confidence) ===
        if not candidates:
            return None
        
        signal = max(candidates, key=lambda x: x['confidence'])
        
        # Add additional confirmations if signal detected
        if signal:
            # WaveTrend confirmation
            if signal['direction'] == 'SHORT' and wt1 > 60 and wt1 > wt2:
                signal['confidence'] += 10
                signal['breakdown'].append("WaveTrend overbought cross")
            elif signal['direction'] == 'LONG' and wt1 < -60 and wt1 < wt2:
                signal['confidence'] += 10
                signal['breakdown'].append("WaveTrend oversold cross")
            
            # EMA alignment
            ema34 = current.get('ema34', price)
            ema89 = current.get('ema89', price)
            
            if signal['direction'] == 'SHORT' and price < ema34 < ema89:
                signal['confidence'] += 20
                signal['breakdown'].append("EMA bearish alignment")
                signal['ema_aligned'] = True
            elif signal['direction'] == 'LONG' and price > ema34 > ema89:
                signal['confidence'] += 20
                signal['breakdown'].append("EMA bullish alignment")
                signal['ema_aligned'] = True
            else:
                signal['ema_aligned'] = False
            
            # RSI divergence check (simplified)
            rsi_prev5 = window['rsi'].iloc[-5:-1]
            if len(rsi_prev5) >= 4:
                if signal['direction'] == 'SHORT' and rsi < rsi_prev5.max() - 5:
                    signal['confidence'] += 15
                    signal['breakdown'].append("Bearish RSI divergence")
                elif signal['direction'] == 'LONG' and rsi > rsi_prev5.min() + 5:
                    signal['confidence'] += 15
                    signal['breakdown'].append("Bullish RSI divergence")
            
            # Store ATR for TP/SL calculation
            signal['atr'] = atr
            signal['swing_high'] = swing_high
            signal['swing_low'] = swing_low
            signal['rsi'] = rsi
            signal['volume_ratio'] = volume_ratio
            signal['prev_green'] = prev.get('prev_green', False)
        
        return signal
    
    def _apply_filters(
        self,
        signal: Dict,
        window: pd.DataFrame,
        current_time
    ) -> Tuple[bool, str]:
        """
        Apply BTC Correlation and 4-Layer filters.
        
        Returns:
            (passed, reason)
        """
        direction = signal['direction']
        
        # === 1. BTC CORRELATION FILTER ===
        if self.btc_data is not None and direction == "LONG":
            try:
                # Find BTC candle at same time (with tolerance for timestamp matching)
                btc_idx = None
                if current_time in self.btc_data.index:
                    btc_idx = self.btc_data.index.get_loc(current_time)
                else:
                    # Try to find closest timestamp within 15 minutes
                    time_diffs = abs(self.btc_data.index - current_time)
                    min_diff_idx = time_diffs.argmin()
                    if time_diffs[min_diff_idx].total_seconds() <= 900:  # 15 minutes tolerance
                        btc_idx = min_diff_idx
                
                if btc_idx is not None and btc_idx > 0:
                    btc_current = self.btc_data.iloc[btc_idx]
                    btc_prev = self.btc_data.iloc[btc_idx - 1]
                    
                    btc_change = (btc_current['close'] - btc_prev['close']) / btc_prev['close'] * 100
                    
                    # Track BTC changes for debugging
                    if not hasattr(self, 'btc_changes'):
                        self.btc_changes = []
                    self.btc_changes.append(btc_change)
                    
                    if btc_change < BTC_DUMP_THRESHOLD:
                        self.blocked_btc += 1
                        return False, f"BTC dump {btc_change:.2f}%"
            except Exception as e:
                logger.debug(f"BTC filter error: {e}")
                pass  # Skip filter if BTC data unavailable
        
        # === 2. 4-LAYER FILTER FOR SHORT (relaxed to 1/4) ===
        if direction == "SHORT":
            current = window.iloc[-1]
            prev = window.iloc[-2]
            
            layers_passed = 0
            
            # Layer 1: Price at/above BB Upper OR RSI > 75
            if current['close'] >= current.get('bb_upper', current['close'] * 1.02) or current.get('rsi', 50) > 75:
                layers_passed += 1
            
            # Layer 2: Shooting Star pattern (upper wick > 50% of range)
            candle_range = current['candle_range']
            if candle_range > 0:
                upper_wick_pct = current['upper_wick'] / candle_range
                if upper_wick_pct > 0.5:
                    layers_passed += 1
            
            # Layer 3: Volume > 2x MA20
            if current.get('volume_ratio', 1) >= 2.0:
                layers_passed += 1
            
            # Layer 4: Previous candle was green
            if prev.get('close', 0) > prev.get('open', 0):
                layers_passed += 1
            
            signal['four_layer_passed'] = layers_passed
            
            # === ADAPTIVE SHORT: Require 3/4 layers ===
            # (Tier check based on Market Regime is done in _create_trade)
            if layers_passed < 3:
                self.blocked_4layer += 1
                return False, f"4-Layer: {layers_passed}/4"
        
        return True, "OK"
    
    def _get_market_regime(self, window: pd.DataFrame) -> str:
        """
        Determine market regime based on BTC price vs EMA200.
        
        Returns:
            "BULLISH" if BTC > EMA200 * 1.01 (1% buffer)
            "BEARISH" if BTC < EMA200 * 0.99
            "NEUTRAL" otherwise
        """
        if self.btc_data is None or 'ema200' not in self.btc_data.columns:
            return "NEUTRAL"
        
        # Find closest BTC candle to current time
        current_time = window.index[-1]
        
        # Get BTC data up to current time
        btc_mask = self.btc_data.index <= current_time
        if not btc_mask.any():
            return "NEUTRAL"
        
        btc_current = self.btc_data.loc[btc_mask].iloc[-1]
        btc_price = btc_current['close']
        btc_ema200 = btc_current['ema200']
        
        if pd.isna(btc_ema200) or btc_ema200 <= 0:
            return "NEUTRAL"
        
        # 1% buffer zone
        if btc_price > btc_ema200 * 1.01:
            return "BULLISH"
        elif btc_price < btc_ema200 * 0.99:
            return "BEARISH"
        else:
            return "NEUTRAL"
    
    def _get_market_state(self, window: pd.DataFrame) -> str:
        """
        Determine market state based on BTC conditions (v2.0 tuned).
        
        REFINED STATES:
        - CRASH: BTC drop > 5%/1h (emergency stop)
        - DANGER: BTC drop > 3%/1h OR (drop > 2% + below EMA89)
        - WARNING: BTC below EMA89 * 0.98
        - CAUTION/FAVORABLE: Normal conditions
        
        Returns:
            "CRASH", "DANGER", "WARNING", "CAUTION", or "FAVORABLE"
        """
        if self.btc_data is None:
            return "CAUTION"
        
        current_time = window.index[-1]
        btc_mask = self.btc_data.index <= current_time
        if not btc_mask.any():
            return "CAUTION"
        
        btc_current = self.btc_data.loc[btc_mask].iloc[-1]
        btc_price = btc_current['close']
        
        # Calculate 1h change (4 candles for 15m data)
        btc_slice = self.btc_data.loc[btc_mask].tail(5)
        if len(btc_slice) >= 5:
            btc_1h_ago = btc_slice.iloc[0]['close']
            change_1h = ((btc_price - btc_1h_ago) / btc_1h_ago) * 100
        else:
            change_1h = 0
        
        # Get EMA89
        if 'ema89' not in self.btc_data.columns:
            self.btc_data['ema89'] = self.btc_data['close'].ewm(span=89, adjust=False).mean()
        
        btc_ema89 = btc_current.get('ema89', btc_price)
        if pd.isna(btc_ema89):
            btc_ema89 = btc_price
        
        # CRASH: Emergency stop
        if change_1h < -5.0:
            return "CRASH"
        
        # DANGER: Hard dump
        if change_1h < -3.0:
            return "DANGER"
        
        # DANGER: Below EMA89 + moderate dump
        if btc_price < btc_ema89 and change_1h < -2.0:
            return "DANGER"
        
        # WARNING: Below EMA89 with 2% buffer
        if btc_price < btc_ema89 * 0.98:
            return "WARNING"
        
        # WARNING: Moderate dump
        if change_1h < -1.5:
            return "WARNING"
        
        return "CAUTION"
    
    def _create_trade(
        self,
        symbol: str,
        window: pd.DataFrame,
        signal: Dict,
        future: pd.DataFrame
    ) -> Optional[BacktestTradeV2]:
        """Create trade with dynamic ATR-based TP/SL and simulate result."""
        current = window.iloc[-1]
        entry = signal['entry']
        direction = signal['direction']
        atr = signal.get('atr', entry * 0.01)
        confidence = signal['confidence']
        
        # === DYNAMIC ATR-BASED TP/SL ===
        # SL = 1.5 * ATR from entry
        # TP1 = 2R (SL distance * 2)
        # TP2 = 4R
        # TP3 = 6R
        
        sl_distance = atr * 1.5
        
        if direction == "LONG":
            sl = entry - sl_distance
            tp1 = entry + (sl_distance * 2)  # 2R
            tp2 = entry + (sl_distance * 4)  # 4R
            tp3 = entry + (sl_distance * 6)  # 6R
        else:  # SHORT
            sl = entry + sl_distance
            tp1 = entry - (sl_distance * 2)
            tp2 = entry - (sl_distance * 4)
            tp3 = entry - (sl_distance * 6)
        
        # Determine tier
        if confidence >= THRESHOLD_DIAMOND:
            tier = "DIAMOND"
        elif confidence >= THRESHOLD_GOLD:
            tier = "GOLD"
        elif confidence >= THRESHOLD_SILVER:
            tier = "SILVER"
        else:
            return None  # Reject trades below SILVER threshold
        
        # Only trade DIAMOND and GOLD
        if tier not in ["DIAMOND", "GOLD"]:
            return None
        
        # === ADAPTIVE SHORT FILTER based on Market Regime + State ===
        if direction == "SHORT":
            market_regime = self._get_market_regime(window)
            market_state = self._get_market_state(window)
            
            # CRASH state: Block all trades
            if market_state == "CRASH":
                return None
            
            # DANGER/WARNING state: Only Diamond SHORT allowed
            if market_state in ["DANGER", "WARNING"]:
                if tier != "DIAMOND":
                    return None  # Reject Gold tier SHORT in DANGER/WARNING
            
            # Apply Market Regime filter (EMA200-based)
            if market_regime == "BEARISH":
                # BEARISH regime: Allow both Diamond and Gold SHORT
                pass
            else:
                # BULLISH or NEUTRAL: SHORT only DIAMOND (Sniper mode)
                if tier != "DIAMOND":
                    return None  # Reject Gold tier SHORT in bullish/neutral market
        
        # Simulate trade result
        result, exit_price, hit_level = self._simulate_trade(
            direction, entry, sl, tp1, tp2, tp3, future
        )
        
        # Calculate PnL
        if direction == "LONG":
            pnl_pct = ((exit_price - entry) / entry) * 100
        else:
            pnl_pct = ((entry - exit_price) / entry) * 100
        
        # Apply slippage and fees
        total_fees = SLIPPAGE_PERCENT + TRADING_FEE * 2  # Entry + exit
        pnl_pct -= total_fees
        
        # Position size based on tier
        position_size = self.position_size_diamond if tier == "DIAMOND" else self.position_size_gold
        pnl_usd = (pnl_pct / 100) * position_size * 50  # Assume 50x leverage
        
        # Get entry time
        try:
            entry_time = current.name if hasattr(current, 'name') else datetime.now()
        except:
            entry_time = datetime.now()
        
        return BacktestTradeV2(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            entry_time=entry_time,
            stoploss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            strategy=signal['strategy'],
            confidence_score=confidence,
            tier=tier,
            four_layer_passed=signal.get('four_layer_passed', 0),
            atr_value=atr,
            exit_price=exit_price,
            result=result,
            pnl_percent=pnl_pct,
            pnl_usd=pnl_usd,
            hit_level=hit_level,
            breakdown=signal.get('breakdown', [])
        )
    
    def _simulate_trade(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        future: pd.DataFrame
    ) -> Tuple[TradeResult, float, str]:
        """Simulate trade outcome using future candles with max 24h hold."""
        # Max hold = 96 candles (24 hours for 15m timeframe)
        max_hold_candles = min(96, len(future))
        
        for idx in range(max_hold_candles):
            candle = future.iloc[idx]
            high = candle['high']
            low = candle['low']
            
            if direction == "LONG":
                # Check SL hit first (worst case)
                if low <= sl:
                    return TradeResult.LOSS, sl, "SL"
                # Check TP hits
                if high >= tp3:
                    return TradeResult.WIN, tp3, "TP3"
                if high >= tp2:
                    return TradeResult.WIN, tp2, "TP2"
                if high >= tp1:
                    return TradeResult.WIN, tp1, "TP1"
            else:  # SHORT
                # Check SL hit first
                if high >= sl:
                    return TradeResult.LOSS, sl, "SL"
                # Check TP hits
                if low <= tp3:
                    return TradeResult.WIN, tp3, "TP3"
                if low <= tp2:
                    return TradeResult.WIN, tp2, "TP2"
                if low <= tp1:
                    return TradeResult.WIN, tp1, "TP1"
        
        # Trade still open after max hold - force close at market
        # Count as LOSS if negative PnL, WIN if positive
        exit_price = future.iloc[max_hold_candles - 1]['close'] if max_hold_candles > 0 else entry
        
        if direction == "LONG":
            pnl = exit_price - entry
        else:
            pnl = entry - exit_price
        
        if pnl > 0:
            return TradeResult.WIN, exit_price, "TIMEOUT_WIN"
        elif pnl < 0:
            return TradeResult.LOSS, exit_price, "TIMEOUT_LOSS"
        else:
            return TradeResult.BREAKEVEN, exit_price, "TIMEOUT"
    
    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime,
        total_candles: int
    ) -> BacktestResultV2:
        """Calculate comprehensive backtest statistics."""
        total = len(self.trades)
        
        if total == 0:
            return BacktestResultV2(
                start_date=start_date,
                end_date=end_date,
                total_candles=total_candles,
                total_trades=0,
                wins=0, losses=0, breakevens=0, winrate=0,
                diamond_trades=0, diamond_wins=0, diamond_winrate=0,
                gold_trades=0, gold_wins=0, gold_winrate=0,
                long_trades=0, long_winrate=0,
                short_trades=0, short_winrate=0,
                total_pnl_percent=0, total_pnl_usd=0,
                average_win_percent=0, average_loss_percent=0,
                profit_factor=0,
                max_drawdown_percent=0, max_drawdown_usd=0,
                blocked_by_btc_filter=self.blocked_btc,
                blocked_by_4layer_filter=self.blocked_4layer,
                strategy_stats={},
                trades=[]
            )
        
        # Count outcomes
        wins = sum(1 for t in self.trades if t.result == TradeResult.WIN)
        losses = sum(1 for t in self.trades if t.result == TradeResult.LOSS)
        breakevens = sum(1 for t in self.trades if t.result == TradeResult.BREAKEVEN)
        
        closed = wins + losses
        winrate = (wins / closed * 100) if closed > 0 else 0
        
        # By tier
        diamond = [t for t in self.trades if t.tier == "DIAMOND"]
        diamond_wins = sum(1 for t in diamond if t.result == TradeResult.WIN)
        diamond_closed = sum(1 for t in diamond if t.result in [TradeResult.WIN, TradeResult.LOSS])
        diamond_winrate = (diamond_wins / diamond_closed * 100) if diamond_closed > 0 else 0
        
        gold = [t for t in self.trades if t.tier == "GOLD"]
        gold_wins = sum(1 for t in gold if t.result == TradeResult.WIN)
        gold_closed = sum(1 for t in gold if t.result in [TradeResult.WIN, TradeResult.LOSS])
        gold_winrate = (gold_wins / gold_closed * 100) if gold_closed > 0 else 0
        
        # By direction
        longs = [t for t in self.trades if t.direction == "LONG"]
        long_wins = sum(1 for t in longs if t.result == TradeResult.WIN)
        long_closed = sum(1 for t in longs if t.result in [TradeResult.WIN, TradeResult.LOSS])
        long_winrate = (long_wins / long_closed * 100) if long_closed > 0 else 0
        
        shorts = [t for t in self.trades if t.direction == "SHORT"]
        short_wins = sum(1 for t in shorts if t.result == TradeResult.WIN)
        short_closed = sum(1 for t in shorts if t.result in [TradeResult.WIN, TradeResult.LOSS])
        short_winrate = (short_wins / short_closed * 100) if short_closed > 0 else 0
        
        # PnL stats
        win_pnls = [t.pnl_percent for t in self.trades if t.result == TradeResult.WIN]
        loss_pnls = [t.pnl_percent for t in self.trades if t.result == TradeResult.LOSS]
        
        total_pnl = sum(t.pnl_percent for t in self.trades)
        total_pnl_usd = sum(t.pnl_usd for t in self.trades)
        avg_win = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0
        avg_loss = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0
        
        total_wins_pnl = sum(win_pnls) if win_pnls else 0
        total_losses_pnl = abs(sum(loss_pnls)) if loss_pnls else 1
        profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0
        
        # Calculate drawdown
        max_dd, max_dd_usd = self._calculate_drawdown()
        
        # Strategy breakdown
        strategy_stats = {}
        strategies = set(t.strategy for t in self.trades)
        for strat in strategies:
            strat_trades = [t for t in self.trades if t.strategy == strat]
            strat_wins = sum(1 for t in strat_trades if t.result == TradeResult.WIN)
            strat_closed = sum(1 for t in strat_trades if t.result in [TradeResult.WIN, TradeResult.LOSS])
            strat_pnl = sum(t.pnl_percent for t in strat_trades)
            
            strategy_stats[strat] = {
                'count': len(strat_trades),
                'wins': strat_wins,
                'winrate': (strat_wins / strat_closed * 100) if strat_closed > 0 else 0,
                'avg_pnl': strat_pnl / len(strat_trades) if strat_trades else 0
            }
        
        return BacktestResultV2(
            start_date=start_date,
            end_date=end_date,
            total_candles=total_candles,
            total_trades=total,
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            winrate=winrate,
            diamond_trades=len(diamond),
            diamond_wins=diamond_wins,
            diamond_winrate=diamond_winrate,
            gold_trades=len(gold),
            gold_wins=gold_wins,
            gold_winrate=gold_winrate,
            long_trades=len(longs),
            long_winrate=long_winrate,
            short_trades=len(shorts),
            short_winrate=short_winrate,
            total_pnl_percent=total_pnl,
            total_pnl_usd=total_pnl_usd,
            average_win_percent=avg_win,
            average_loss_percent=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_percent=max_dd,
            max_drawdown_usd=max_dd_usd,
            blocked_by_btc_filter=self.blocked_btc,
            blocked_by_4layer_filter=self.blocked_4layer,
            blocked_by_cooldown=self.blocked_cooldown,
            strategy_stats=strategy_stats,
            trades=self.trades
        )
    
    def _calculate_drawdown(self) -> Tuple[float, float]:
        """Calculate maximum drawdown."""
        if not self.trades:
            return 0, 0
        
        # Sort trades by time
        sorted_trades = sorted(self.trades, key=lambda t: t.entry_time)
        
        balance = self.initial_balance
        peak = balance
        max_dd_pct = 0
        max_dd_usd = 0
        
        for trade in sorted_trades:
            balance += trade.pnl_usd
            
            if balance > peak:
                peak = balance
            
            dd_usd = peak - balance
            dd_pct = (dd_usd / peak * 100) if peak > 0 else 0
            
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
                max_dd_usd = dd_usd
        
        return max_dd_pct, max_dd_usd
    
    def print_report(self, result: BacktestResultV2):
        """Print formatted backtest report."""
        print("\n" + "="*70)
        print("📊 BACKTEST REPORT V2.0 - Full Signal Simulation")
        print("="*70)
        print(f"📅 Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"📈 Total Candles: {result.total_candles:,}")
        print(f"💱 Total Trades: {result.total_trades}")
        print()
        
        print("═══════════════════════════════════════════════════════════════════")
        print("📈 OVERALL PERFORMANCE")
        print("═══════════════════════════════════════════════════════════════════")
        print(f"   Win/Loss/BE: {result.wins}/{result.losses}/{result.breakevens}")
        print(f"   Winrate: {result.winrate:.1f}%")
        print(f"   Total PnL: {result.total_pnl_percent:+.2f}% (${result.total_pnl_usd:+.2f})")
        print(f"   Profit Factor: {result.profit_factor:.2f}")
        print(f"   Max Drawdown: {result.max_drawdown_percent:.2f}% (${result.max_drawdown_usd:.2f})")
        print()
        
        print("═══════════════════════════════════════════════════════════════════")
        print("💎 TIER BREAKDOWN")
        print("═══════════════════════════════════════════════════════════════════")
        d_status = "✅" if result.diamond_winrate >= 65 else "❌"
        g_status = "✅" if result.gold_winrate >= 55 else "❌"
        print(f"   {d_status} DIAMOND: {result.diamond_trades} trades, {result.diamond_wins} wins, {result.diamond_winrate:.1f}% (target: 65%+)")
        print(f"   {g_status} GOLD: {result.gold_trades} trades, {result.gold_wins} wins, {result.gold_winrate:.1f}% (target: 55%+)")
        print()
        
        print("═══════════════════════════════════════════════════════════════════")
        print("📊 DIRECTION BREAKDOWN")
        print("═══════════════════════════════════════════════════════════════════")
        print(f"   LONG: {result.long_trades} trades, {result.long_winrate:.1f}% winrate")
        print(f"   SHORT: {result.short_trades} trades, {result.short_winrate:.1f}% winrate")
        print()
        
        print("═══════════════════════════════════════════════════════════════════")
        print("🔒 FILTER STATS")
        print("═══════════════════════════════════════════════════════════════════")
        print(f"   Blocked by BTC Correlation: {result.blocked_by_btc_filter}")
        print(f"   Blocked by 4-Layer Filter: {result.blocked_by_4layer_filter}")
        print(f"   Blocked by Cooldown (12 candles): {result.blocked_by_cooldown}")
        
        # BTC change stats for debugging
        if hasattr(self, 'btc_changes') and self.btc_changes:
            btc_min = min(self.btc_changes)
            btc_max = max(self.btc_changes)
            btc_dumps = sum(1 for c in self.btc_changes if c < -0.5)
            print(f"   BTC Change Range: {btc_min:.2f}% to {btc_max:.2f}%")
            print(f"   BTC Dumps > 0.5%: {btc_dumps} times")
        print()
        
        print("═══════════════════════════════════════════════════════════════════")
        print("📋 STRATEGY BREAKDOWN")
        print("═══════════════════════════════════════════════════════════════════")
        for strat, stats in result.strategy_stats.items():
            wr = stats.get('winrate', 0)
            status = "✅" if wr >= 55 else "❌"
            print(f"   {status} {strat}:")
            print(f"      Trades: {stats.get('count', 0)}")
            print(f"      Winrate: {wr:.1f}%")
            print(f"      Avg PnL: {stats.get('avg_pnl', 0):+.2f}%")
        
        print("\n" + "="*70)
        
        # Sample trades
        if result.trades:
            print("\n📝 SAMPLE TRADES (Last 10):")
            for trade in result.trades[-10:]:
                emoji = "✅" if trade.result == TradeResult.WIN else "❌"
                print(f"   {emoji} {trade.symbol} {trade.direction} | "
                      f"Score: {trade.confidence_score} ({trade.tier}) | "
                      f"PnL: {trade.pnl_percent:+.2f}% | Hit: {trade.hit_level}")


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════

async def run_backtest_v2_cli():
    """CLI entry point for backtest v2."""
    import argparse
    from src.ingestion.rest_client import BingXRestClient
    
    parser = argparse.ArgumentParser(description="Run backtest v2 with full signals")
    parser.add_argument("--days", type=int, default=14, help="Days to backtest")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to test")
    
    args = parser.parse_args()
    
    # Default symbols
    symbols = args.symbols or [
        "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT",
        "ADA-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "MATIC-USDT",
        "UNI-USDT", "ATOM-USDT", "LTC-USDT", "FIL-USDT", "ARB-USDT",
        "OP-USDT", "APT-USDT", "SUI-USDT", "INJ-USDT", "TIA-USDT"
    ]
    
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=args.days)
    
    async with BingXRestClient() as client:
        engine = BacktestEngineV2(rest_client=client)
        result = await engine.run(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="15m"
        )
        
        engine.print_report(result)
        
        return result


if __name__ == "__main__":
    asyncio.run(run_backtest_v2_cli())
