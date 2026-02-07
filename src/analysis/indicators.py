"""
Advanced Technical Indicators Calculator (v2.0)
Adds: VWAP, OBV, Stochastic RSI, MACD, Multi-timeframe analysis
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Dict
from enum import Enum

import pandas as pd
import ta

from ..config import config

logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Trend classification."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class CoinIndicators:
    """All calculated indicators for a coin (Enhanced v3.0 - Full TradingView Style)."""
    symbol: str
    price: float
    
    # EMAs (Exponential Moving Averages)
    ema34_h1: float
    ema89_h1: float
    ema34_h4: float
    ema89_h4: float
    
    # RSI (Relative Strength Index)
    rsi_15m: float
    rsi_h1: float
    
    # Bollinger Bands
    bb_upper: float
    bb_lower: float
    bb_middle: float
    bb_bandwidth: float  # (upper - lower) / middle
    
    # ADX (Average Directional Index) - trend strength
    adx: float
    
    # ATR (Average True Range) - volatility
    atr: float
    
    # Volume
    current_volume: float
    avg_volume_20: float
    volume_ratio: float  # current / avg
    
    # Support/Resistance levels
    swing_high_20: float  # Highest high in last 20 candles
    swing_low_20: float   # Lowest low in last 20 candles
    
    # ========== v2.0 INDICATORS ==========
    
    # VWAP (Volume Weighted Average Price)
    vwap: float = 0.0
    price_vs_vwap: float = 0.0  # % above/below VWAP
    
    # OBV (On-Balance Volume)
    obv: float = 0.0
    obv_trend: str = "NEUTRAL"  # BULLISH, BEARISH, NEUTRAL
    
    # Stochastic RSI (more sensitive than RSI)
    stoch_rsi_k: float = 50.0
    stoch_rsi_d: float = 50.0
    
    # MACD
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    macd_trend: str = "NEUTRAL"
    
    # Multi-Timeframe Analysis
    trend_15m: str = "NEUTRAL"
    trend_h1: str = "NEUTRAL"
    trend_h4: str = "NEUTRAL"
    mtf_alignment_score: int = 0  # 0-3: how many timeframes aligned
    
    # Candle Analysis
    last_candle_bullish: bool = True
    candle_body_ratio: float = 0.5  # body / total range
    
    # Orderbook Imbalance (if available)
    bid_ask_ratio: float = 1.0  # bid_volume / ask_volume
    
    # ========== v3.0 INDICATORS ==========
    
    # MFI (Money Flow Index)
    mfi: float = 50.0
    mfi_status: str = "Neutral"  # Overbought, Oversold, Neutral
    
    # CCI (Commodity Channel Index)
    cci: float = 0.0
    
    # Bollinger Bands Status
    bb_status: str = "Normal Range"  # Above Upper, Below Lower, Normal Range
    
    # SMA (Simple Moving Average)
    sma_50: float = 0.0
    sma_status: str = "Neutral"  # Price above SMA, Price below SMA
    
    # Momentum
    momentum: float = 0.0
    momentum_status: str = "Neutral"  # Above 0 - Bullish, Below 0 - Bearish
    
    # ADX Signal (direction + strength)
    adx_signal: str = "N/A"  # Strong Bullish, Normal Bearish, etc.
    plus_di: float = 0.0
    minus_di: float = 0.0
    
    # Parabolic SAR
    psar: float = 0.0
    psar_direction: str = "Bullish"  # Bullish, Bearish
    
    # TD Sequential (Tom DeMark)
    td_count: int = 0
    td_direction: str = "None"  # Up, Down, None
    
    # RSI Divergence
    rsi_divergence: str = "None"  # Bullish, Bearish, None
    
    # Volatility %
    volatility_pct: float = 0.0
    
    # WaveTrend
    wt1: float = 0.0
    wt2: float = 0.0
    wt_signal: str = "Neutral"  # Bullish Cross, Bearish Cross, Overbought, Oversold
    
    # Squeeze Momentum
    squeeze_value: float = 0.0
    squeeze_on: bool = False  # True = squeeze is on (low volatility)
    squeeze_momentum: str = "Neutral"  # Bullish, Bearish, Neutral
    
    # ========== PROPERTIES ==========
    
    @property
    def is_above_ema34(self) -> bool:
        return self.price > self.ema34_h4
    
    @property
    def is_above_ema89(self) -> bool:
        return self.price > self.ema89_h4
    
    @property
    def is_uptrend(self) -> bool:
        return self.ema34_h4 > self.ema89_h4
    
    @property
    def is_overbought(self) -> bool:
        return self.rsi_15m > config.zone_detection.rsi_overbought
    
    @property
    def is_oversold(self) -> bool:
        return self.rsi_15m < config.zone_detection.rsi_oversold
    
    @property
    def has_strong_trend(self) -> bool:
        return self.adx > config.zone_detection.adx_trend_threshold
    
    @property
    def is_stoch_oversold(self) -> bool:
        return self.stoch_rsi_k < 20
    
    @property
    def is_stoch_overbought(self) -> bool:
        return self.stoch_rsi_k > 80
    
    @property
    def is_above_vwap(self) -> bool:
        return self.price > self.vwap if self.vwap > 0 else True
    
    @property
    def has_mtf_alignment(self) -> bool:
        """True if at least 2 timeframes are aligned."""
        return self.mtf_alignment_score >= 2
    
    @property
    def has_bullish_divergence(self) -> bool:
        """Price making lower lows but OBV making higher lows."""
        return self.obv_trend == "BULLISH" and self.price < self.swing_low_20 * 1.02
    
    @property
    def is_mfi_overbought(self) -> bool:
        return self.mfi > 80
    
    @property
    def is_mfi_oversold(self) -> bool:
        return self.mfi < 20
    
    @property
    def is_squeeze_ready(self) -> bool:
        """True if squeeze is on and momentum is building."""
        return self.squeeze_on and abs(self.squeeze_value) > 0


class IndicatorCalculator:
    """Calculate technical indicators from kline data (Enhanced v2.0)."""
    
    def __init__(self):
        self.ema_fast = config.indicator.ema_fast
        self.ema_slow = config.indicator.ema_slow
        self.rsi_period = config.indicator.rsi_period
        self.bb_period = config.indicator.bb_period
        self.bb_std = config.indicator.bb_std
        self.adx_period = config.indicator.adx_period
        self.atr_period = config.indicator.atr_period
        self.volume_lookback = config.indicator.volume_lookback
    
    def calculate(
        self,
        symbol: str,
        klines_15m: List,
        klines_h1: List,
        klines_h4: List
    ) -> Optional[CoinIndicators]:
        """
        Calculate all indicators for a coin (Enhanced v2.0).
        """
        try:
            # Convert to DataFrames
            df_15m = self._to_dataframe(klines_15m)
            df_h1 = self._to_dataframe(klines_h1)
            df_h4 = self._to_dataframe(klines_h4)
            
            if df_15m.empty or df_h1.empty or df_h4.empty:
                logger.warning(f"Empty data for {symbol}")
                return None
            
            # Current price & candle info
            price = float(df_15m['close'].iloc[-1])
            last_open = float(df_15m['open'].iloc[-1])
            last_high = float(df_15m['high'].iloc[-1])
            last_low = float(df_15m['low'].iloc[-1])
            last_candle_bullish = price > last_open
            candle_range = last_high - last_low
            candle_body = abs(price - last_open)
            candle_body_ratio = candle_body / candle_range if candle_range > 0 else 0.5
            
            # ========== BASIC INDICATORS ==========
            
            # EMAs
            ema34_h1 = self._calc_ema(df_h1['close'], self.ema_fast)
            ema89_h1 = self._calc_ema(df_h1['close'], self.ema_slow)
            ema34_h4 = self._calc_ema(df_h4['close'], self.ema_fast)
            ema89_h4 = self._calc_ema(df_h4['close'], self.ema_slow)
            
            # RSI
            rsi_15m = self._calc_rsi(df_15m['close'], self.rsi_period)
            rsi_h1 = self._calc_rsi(df_h1['close'], self.rsi_period)
            
            # Bollinger Bands (on H4)
            bb_upper, bb_middle, bb_lower = self._calc_bollinger(df_h4['close'])
            bb_bandwidth = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
            
            # ADX (on H4)
            adx = self._calc_adx(df_h4)
            
            # ATR (on H4)
            atr = self._calc_atr(df_h4)
            
            # Volume analysis (on 15m)
            current_volume = float(df_15m['volume'].iloc[-1])
            avg_volume_20 = float(df_15m['volume'].rolling(self.volume_lookback).mean().iloc[-1])
            volume_ratio = current_volume / avg_volume_20 if avg_volume_20 > 0 else 0
            
            # Swing High/Low (on H4)
            swing_high = float(df_h4['high'].rolling(20).max().iloc[-1])
            swing_low = float(df_h4['low'].rolling(20).min().iloc[-1])
            
            # ========== NEW v2.0 INDICATORS ==========
            
            # VWAP (on 15m - intraday)
            vwap = self._calc_vwap(df_15m)
            price_vs_vwap = ((price - vwap) / vwap * 100) if vwap > 0 else 0
            
            # OBV
            obv, obv_trend = self._calc_obv(df_h1)
            
            # Stochastic RSI
            stoch_rsi_k, stoch_rsi_d = self._calc_stoch_rsi(df_15m['close'])
            
            # MACD
            macd_line, macd_signal, macd_histogram, macd_trend = self._calc_macd(df_h1['close'])
            
            # Multi-timeframe trend analysis
            trend_15m = self._determine_trend(df_15m)
            trend_h1 = self._determine_trend(df_h1)
            trend_h4 = self._determine_trend(df_h4)
            
            # MTF alignment score
            mtf_alignment_score = self._calc_mtf_alignment(trend_15m, trend_h1, trend_h4)
            
            # ========== NEW v3.0 INDICATORS ==========
            
            # MFI (Money Flow Index)
            mfi = self._calc_mfi(df_h1)
            mfi_status = "Overbought" if mfi > 80 else "Oversold" if mfi < 20 else "Neutral"
            
            # CCI (Commodity Channel Index)
            cci = self._calc_cci(df_h1)
            
            # BB Status
            if price > bb_upper:
                bb_status = "Above Upper"
            elif price < bb_lower:
                bb_status = "Below Lower"
            else:
                bb_status = "Normal Range"
            
            # SMA 50
            sma_50 = self._calc_sma(df_h1['close'], 50)
            sma_status = "Price above SMA" if price > sma_50 else "Price below SMA"
            
            # Momentum
            momentum = self._calc_momentum(df_h1['close'], 10)
            momentum_status = "Bullish" if momentum > 0 else "Bearish"
            
            # ADX with direction
            adx, plus_di, minus_di, adx_signal = self._calc_adx_full(df_h4)
            
            # Parabolic SAR
            psar, psar_direction = self._calc_psar(df_h1)
            
            # TD Sequential
            td_count, td_direction = self._calc_td_sequential(df_h1)
            
            # RSI Divergence
            rsi_divergence = self._detect_rsi_divergence(df_h1)
            
            # Volatility %
            volatility_pct = self._calc_volatility_pct(df_h1)
            
            # WaveTrend
            wt1, wt2, wt_signal = self._calc_wavetrend(df_h1)
            
            # Squeeze Momentum
            squeeze_value, squeeze_on, squeeze_momentum = self._calc_squeeze(df_h1)
            
            return CoinIndicators(
                symbol=symbol,
                price=price,
                # Basic
                ema34_h1=ema34_h1,
                ema89_h1=ema89_h1,
                ema34_h4=ema34_h4,
                ema89_h4=ema89_h4,
                rsi_15m=rsi_15m,
                rsi_h1=rsi_h1,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
                bb_middle=bb_middle,
                bb_bandwidth=bb_bandwidth,
                adx=adx,
                atr=atr,
                current_volume=current_volume,
                avg_volume_20=avg_volume_20,
                volume_ratio=volume_ratio,
                swing_high_20=swing_high,
                swing_low_20=swing_low,
                # New v2.0
                vwap=vwap,
                price_vs_vwap=price_vs_vwap,
                obv=obv,
                obv_trend=obv_trend,
                stoch_rsi_k=stoch_rsi_k,
                stoch_rsi_d=stoch_rsi_d,
                macd_line=macd_line,
                macd_signal=macd_signal,
                macd_histogram=macd_histogram,
                macd_trend=macd_trend,
                trend_15m=trend_15m,
                trend_h1=trend_h1,
                trend_h4=trend_h4,
                mtf_alignment_score=mtf_alignment_score,
                last_candle_bullish=last_candle_bullish,
                candle_body_ratio=candle_body_ratio,
                # v3.0 indicators
                mfi=mfi,
                mfi_status=mfi_status,
                cci=cci,
                bb_status=bb_status,
                sma_50=sma_50,
                sma_status=sma_status,
                momentum=momentum,
                momentum_status=momentum_status,
                adx_signal=adx_signal,
                plus_di=plus_di,
                minus_di=minus_di,
                psar=psar,
                psar_direction=psar_direction,
                td_count=td_count,
                td_direction=td_direction,
                rsi_divergence=rsi_divergence,
                volatility_pct=volatility_pct,
                wt1=wt1,
                wt2=wt2,
                wt_signal=wt_signal,
                squeeze_value=squeeze_value,
                squeeze_on=squeeze_on,
                squeeze_momentum=squeeze_momentum
            )
            
        except Exception as e:
            logger.error(f"Error calculating indicators for {symbol}: {e}")
            return None
    
    def _to_dataframe(self, klines: List) -> pd.DataFrame:
        """Convert kline data to pandas DataFrame, sorted by time ascending."""
        if not klines:
            return pd.DataFrame()
        
        if isinstance(klines[0], list):
            data = []
            for k in klines:
                if len(k) >= 6:
                    data.append({
                        'time': k[0],
                        'open': float(k[1]),
                        'high': float(k[2]),
                        'low': float(k[3]),
                        'close': float(k[4]),
                        'volume': float(k[5])
                    })
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame(klines)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = df[col].astype(float)
            if 'time' in df.columns:
                df['time'] = df['time'].astype(int)
        
        # Sort by time ascending (oldest first, newest last)
        # This ensures iloc[-1] always gets the most recent data
        if 'time' in df.columns and not df.empty:
            df = df.sort_values('time', ascending=True).reset_index(drop=True)
        
        return df
    
    # --- Basic Indicator Methods ---
    
    def _calc_ema(self, series: pd.Series, period: int) -> float:
        ema = ta.trend.ema_indicator(series, window=period)
        return float(ema.iloc[-1]) if not ema.empty else 0.0
    
    def _calc_rsi(self, series: pd.Series, period: int) -> float:
        rsi = ta.momentum.rsi(series, window=period)
        return float(rsi.iloc[-1]) if not rsi.empty and not pd.isna(rsi.iloc[-1]) else 50.0
    
    def _calc_bollinger(self, series: pd.Series) -> tuple:
        bb = ta.volatility.BollingerBands(series, window=self.bb_period, window_dev=self.bb_std)
        upper = float(bb.bollinger_hband().iloc[-1])
        middle = float(bb.bollinger_mavg().iloc[-1])
        lower = float(bb.bollinger_lband().iloc[-1])
        return upper, middle, lower
    
    def _calc_adx(self, df: pd.DataFrame) -> float:
        adx = ta.trend.adx(df['high'], df['low'], df['close'], window=self.adx_period)
        return float(adx.iloc[-1]) if not adx.empty and not pd.isna(adx.iloc[-1]) else 0.0
    
    def _calc_atr(self, df: pd.DataFrame) -> float:
        atr = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=self.atr_period)
        return float(atr.iloc[-1]) if not atr.empty and not pd.isna(atr.iloc[-1]) else 0.0
    
    # --- v2.0 Indicator Methods ---
    
    def _calc_vwap(self, df: pd.DataFrame) -> float:
        """Calculate VWAP (Volume Weighted Average Price)."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return float(vwap.iloc[-1]) if not vwap.empty else 0.0
        except Exception:
            return 0.0
    
    def _calc_obv(self, df: pd.DataFrame) -> tuple:
        """Calculate OBV and its trend."""
        try:
            obv = ta.volume.on_balance_volume(df['close'], df['volume'])
            obv_val = float(obv.iloc[-1])
            
            # Determine OBV trend (compare last 5 values)
            if len(obv) >= 5:
                obv_sma = obv.rolling(5).mean()
                if obv.iloc[-1] > obv_sma.iloc[-1]:
                    trend = "BULLISH"
                elif obv.iloc[-1] < obv_sma.iloc[-1]:
                    trend = "BEARISH"
                else:
                    trend = "NEUTRAL"
            else:
                trend = "NEUTRAL"
            
            return obv_val, trend
        except Exception:
            return 0.0, "NEUTRAL"
    
    def _calc_stoch_rsi(self, series: pd.Series) -> tuple:
        """Calculate Stochastic RSI."""
        try:
            stoch_rsi = ta.momentum.stochrsi(series, window=14, smooth1=3, smooth2=3)
            stoch_rsi_k = ta.momentum.stochrsi_k(series, window=14, smooth1=3, smooth2=3)
            stoch_rsi_d = ta.momentum.stochrsi_d(series, window=14, smooth1=3, smooth2=3)
            
            k_val = float(stoch_rsi_k.iloc[-1] * 100) if not pd.isna(stoch_rsi_k.iloc[-1]) else 50.0
            d_val = float(stoch_rsi_d.iloc[-1] * 100) if not pd.isna(stoch_rsi_d.iloc[-1]) else 50.0
            
            return k_val, d_val
        except Exception:
            return 50.0, 50.0
    
    def _calc_macd(self, series: pd.Series) -> tuple:
        """Calculate MACD line, signal, histogram, and trend."""
        try:
            macd = ta.trend.MACD(series, window_slow=26, window_fast=12, window_sign=9)
            macd_line = float(macd.macd().iloc[-1])
            macd_signal = float(macd.macd_signal().iloc[-1])
            macd_histogram = float(macd.macd_diff().iloc[-1])
            
            # Determine trend
            if macd_line > macd_signal and macd_histogram > 0:
                trend = "BULLISH"
            elif macd_line < macd_signal and macd_histogram < 0:
                trend = "BEARISH"
            else:
                trend = "NEUTRAL"
            
            return macd_line, macd_signal, macd_histogram, trend
        except Exception:
            return 0.0, 0.0, 0.0, "NEUTRAL"
    
    def _determine_trend(self, df: pd.DataFrame) -> str:
        """Determine trend based on EMA crossover."""
        try:
            ema_fast = ta.trend.ema_indicator(df['close'], window=self.ema_fast)
            ema_slow = ta.trend.ema_indicator(df['close'], window=self.ema_slow)
            
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                return "BULLISH"
            elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                return "BEARISH"
            return "NEUTRAL"
        except Exception:
            return "NEUTRAL"
    
    def _calc_mtf_alignment(self, trend_15m: str, trend_h1: str, trend_h4: str) -> int:
        """Calculate multi-timeframe alignment score (0-3)."""
        score = 0
        
        # Count bullish alignment
        bullish_count = sum(1 for t in [trend_15m, trend_h1, trend_h4] if t == "BULLISH")
        bearish_count = sum(1 for t in [trend_15m, trend_h1, trend_h4] if t == "BEARISH")
        
        # Perfect alignment = 3, partial = 2, mixed = 0-1
        if bullish_count == 3 or bearish_count == 3:
            score = 3
        elif bullish_count == 2 or bearish_count == 2:
            score = 2
        elif bullish_count == 1 or bearish_count == 1:
            score = 1
        
        return score
    
    # --- v3.0 Indicator Methods ---
    
    def _calc_mfi(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Money Flow Index."""
        try:
            mfi = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'], window=period)
            return float(mfi.iloc[-1]) if not pd.isna(mfi.iloc[-1]) else 50.0
        except Exception:
            return 50.0
    
    def _calc_cci(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate Commodity Channel Index."""
        try:
            cci = ta.trend.cci(df['high'], df['low'], df['close'], window=period)
            return float(cci.iloc[-1]) if not pd.isna(cci.iloc[-1]) else 0.0
        except Exception:
            return 0.0
    
    def _calc_sma(self, series: pd.Series, period: int) -> float:
        """Calculate Simple Moving Average."""
        try:
            sma = ta.trend.sma_indicator(series, window=period)
            return float(sma.iloc[-1]) if not pd.isna(sma.iloc[-1]) else 0.0
        except Exception:
            return 0.0
    
    def _calc_momentum(self, series: pd.Series, period: int = 10) -> float:
        """Calculate momentum (price change over period)."""
        try:
            if len(series) < period + 1:
                return 0.0
            return float(series.iloc[-1] - series.iloc[-period-1])
        except Exception:
            return 0.0
    
    def _calc_adx_full(self, df: pd.DataFrame) -> tuple:
        """Calculate ADX with +DI, -DI and signal."""
        try:
            adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=self.adx_period)
            adx_val = float(adx_indicator.adx().iloc[-1])
            plus_di = float(adx_indicator.adx_pos().iloc[-1])
            minus_di = float(adx_indicator.adx_neg().iloc[-1])
            
            # Determine signal
            if plus_di > minus_di:
                direction = "Bullish"
            else:
                direction = "Bearish"
            
            if adx_val > 25:
                strength = "Strong"
            elif adx_val > 20:
                strength = "Normal"
            else:
                strength = "Weak"
            
            signal = f"{strength} {direction} Trend"
            
            return adx_val, plus_di, minus_di, signal
        except Exception:
            return 0.0, 0.0, 0.0, "N/A"
    
    def _calc_psar(self, df: pd.DataFrame) -> tuple:
        """Calculate Parabolic SAR."""
        try:
            psar = ta.trend.PSARIndicator(df['high'], df['low'], df['close'])
            psar_val = psar.psar().iloc[-1]
            current_price = df['close'].iloc[-1]
            direction = "Bullish" if current_price > psar_val else "Bearish"
            return float(psar_val), direction
        except Exception:
            return 0.0, "Neutral"
    
    def _calc_td_sequential(self, df: pd.DataFrame) -> tuple:
        """Calculate TD Sequential count."""
        try:
            closes = df['close'].values
            if len(closes) < 5:
                return 0, "None"
            
            count = 0
            direction = "None"
            
            # Count consecutive closes above/below 4 bars ago
            for i in range(len(closes) - 1, max(3, len(closes) - 15), -1):
                if closes[i] > closes[i - 4]:
                    if direction == "Up" or direction == "None":
                        direction = "Up"
                        count += 1
                    else:
                        break
                elif closes[i] < closes[i - 4]:
                    if direction == "Down" or direction == "None":
                        direction = "Down"
                        count += 1
                    else:
                        break
                else:
                    break
            
            return min(count, 13), direction
        except Exception:
            return 0, "None"
    
    def _detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        """Detect RSI divergence."""
        try:
            if len(df) < 20:
                return "None"
            
            closes = df['close'].values
            rsi = ta.momentum.rsi(df['close'], window=14).values
            
            # Find recent swing lows/highs in price and RSI
            # Bullish divergence: price lower low, RSI higher low
            price_low1 = min(closes[-10:-5])
            price_low2 = min(closes[-5:])
            rsi_low1 = min(rsi[-10:-5])
            rsi_low2 = min(rsi[-5:])
            
            if price_low2 < price_low1 and rsi_low2 > rsi_low1:
                return "Bullish"
            
            # Bearish divergence: price higher high, RSI lower high
            price_high1 = max(closes[-10:-5])
            price_high2 = max(closes[-5:])
            rsi_high1 = max(rsi[-10:-5])
            rsi_high2 = max(rsi[-5:])
            
            if price_high2 > price_high1 and rsi_high2 < rsi_high1:
                return "Bearish"
            
            return "None"
        except Exception:
            return "None"
    
    def _calc_volatility_pct(self, df: pd.DataFrame, period: int = 14) -> float:
        """Calculate volatility as percentage."""
        try:
            highs = df['high'].values[-period:]
            lows = df['low'].values[-period:]
            avg_price = sum((h + l) / 2 for h, l in zip(highs, lows)) / period
            avg_range = sum(h - l for h, l in zip(highs, lows)) / period
            return (avg_range / avg_price * 100) if avg_price > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calc_wavetrend(self, df: pd.DataFrame, n1: int = 10, n2: int = 21) -> tuple:
        """Calculate WaveTrend oscillator."""
        try:
            hlc3 = (df['high'] + df['low'] + df['close']) / 3
            esa = ta.trend.ema_indicator(hlc3, window=n1)
            diff = abs(hlc3 - esa)
            d = ta.trend.ema_indicator(diff, window=n1)
            
            # Avoid division by zero
            ci = pd.Series([0.0] * len(hlc3))
            for i in range(len(hlc3)):
                if d.iloc[i] != 0:
                    ci.iloc[i] = (hlc3.iloc[i] - esa.iloc[i]) / (0.015 * d.iloc[i])
            
            wt1 = ta.trend.ema_indicator(ci, window=n2)
            wt2 = ta.trend.sma_indicator(wt1, window=4)
            
            wt1_val = float(wt1.iloc[-1]) if not pd.isna(wt1.iloc[-1]) else 0.0
            wt2_val = float(wt2.iloc[-1]) if not pd.isna(wt2.iloc[-1]) else 0.0
            
            # Determine signal
            if wt1_val > 60:
                signal = "Overbought"
            elif wt1_val < -60:
                signal = "Oversold"
            elif wt1_val > wt2_val and wt1.iloc[-2] <= wt2.iloc[-2]:
                signal = "Bullish Cross"
            elif wt1_val < wt2_val and wt1.iloc[-2] >= wt2.iloc[-2]:
                signal = "Bearish Cross"
            else:
                signal = "Neutral"
            
            return wt1_val, wt2_val, signal
        except Exception:
            return 0.0, 0.0, "Neutral"
    
    def _calc_squeeze(self, df: pd.DataFrame, length: int = 20, mult: float = 2.0, 
                      lengthKC: int = 20, multKC: float = 1.5) -> tuple:
        """Calculate Squeeze Momentum indicator."""
        try:
            closes = df['close']
            highs = df['high']
            lows = df['low']
            
            # Bollinger Bands
            basis = ta.trend.sma_indicator(closes, window=length)
            dev = closes.rolling(window=length).std() * mult
            upperBB = basis + dev
            lowerBB = basis - dev
            
            # Keltner Channels
            ma = ta.trend.sma_indicator(closes, window=lengthKC)
            tr = ta.volatility.average_true_range(highs, lows, closes, window=lengthKC)
            upperKC = ma + tr * multKC
            lowerKC = ma - tr * multKC
            
            # Squeeze detection
            sqz_on = (lowerBB.iloc[-1] > lowerKC.iloc[-1]) and (upperBB.iloc[-1] < upperKC.iloc[-1])
            
            # Momentum value
            highest = highs.rolling(window=lengthKC).max()
            lowest = lows.rolling(window=lengthKC).min()
            sma_c = ta.trend.sma_indicator(closes, window=lengthKC)
            val = closes - ((highest + lowest) / 2 + sma_c) / 2
            
            squeeze_val = float(val.iloc[-1]) if not pd.isna(val.iloc[-1]) else 0.0
            
            # Determine momentum direction
            if squeeze_val > 0:
                momentum = "Bullish"
            elif squeeze_val < 0:
                momentum = "Bearish"
            else:
                momentum = "Neutral"
            
            return squeeze_val, sqz_on, momentum
        except Exception:
            return 0.0, False, "Neutral"


# Singleton test removed - use pytest instead
