"""
Chart Generator v3.0 - TradingView-Style Charts

Features:
- Larger, clearer candlestick display
- TP/SL zones with shaded regions (like TradingView)
- Clean indicator panels
- Professional color scheme
"""

import io
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

logger = logging.getLogger(__name__)

# TradingView-like color scheme
COLORS = {
    'bg': '#131722',           # Dark background
    'grid': '#1e222d',         # Grid lines
    'text': '#d1d4dc',         # Text color
    'green': '#26a69a',        # Bullish candles
    'red': '#ef5350',          # Bearish candles
    'green_bright': '#00ff88', # Bright green for highlights
    'red_bright': '#ff4444',   # Bright red for highlights
    'blue': '#2962ff',         # EMA lines
    'orange': '#ff9800',       # Entry line
    'yellow': '#f0c14b',       # TP lines
    'purple': '#ab47bc',       # SL line
    'cyan': '#00bcd4',         # VWAP/Other
    'gray': '#787b86',         # Neutral
    'white': '#ffffff',        # Current price
    # Zone colors (semi-transparent)
    'tp_zone': '#26a69a',      # Green for TP zone
    'sl_zone': '#ef5350',      # Red for SL zone
}


class ChartGenerator:
    """
    Generates professional TradingView-style candlestick chart images.
    Features large candles, clear TP/SL zones, and clean indicators.
    """
    
    def __init__(self):
        self.width = 16          # Wider chart
        self.height = 12         # Taller chart
        self.dpi = 150           # Higher resolution
        plt.style.use('dark_background')
    
    def generate_chart(
        self,
        symbol: str,
        klines: List,
        interval: str = "1h",
        num_candles: int = 60,   # Fewer candles = larger display
        # Technical settings
        ema_fast: int = 34,
        ema_slow: int = 89,
        # Trade levels
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit_1: Optional[float] = None,
        take_profit_2: Optional[float] = None,
        take_profit_3: Optional[float] = None,
        trade_direction: str = "LONG",
        # SMC levels
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        # Additional info
        trend: str = "NEUTRAL",
        zone: str = "EQUILIBRIUM",
        strategy_name: str = ""
    ) -> bytes:
        """
        Generate a professional TradingView-style chart with clear TP/SL zones.
        """
        try:
            # Parse klines data and sort by time ascending
            klines_list = list(klines)
            
            # Sort by time ascending if dict format
            if klines_list and isinstance(klines_list[0], dict):
                klines_list = sorted(klines_list, key=lambda x: int(x.get('time', 0)))
            
            # Take last num_candles
            klines_list = klines_list[-num_candles:] if len(klines_list) > num_candles else klines_list
            
            times = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            for k in klines_list:
                if isinstance(k, list) and len(k) >= 6:
                    times.append(datetime.fromtimestamp(k[0] / 1000))
                    opens.append(float(k[1]))
                    highs.append(float(k[2]))
                    lows.append(float(k[3]))
                    closes.append(float(k[4]))
                    volumes.append(float(k[5]))
                elif isinstance(k, dict):
                    times.append(datetime.fromtimestamp(int(k.get('time', 0)) / 1000))
                    opens.append(float(k.get('open', 0)))
                    highs.append(float(k.get('high', 0)))
                    lows.append(float(k.get('low', 0)))
                    closes.append(float(k.get('close', 0)))
                    volumes.append(float(k.get('volume', 0)))
            
            if not closes or len(closes) < 10:
                return b''
            
            # Calculate indicators
            wt1, wt2 = self._calc_wavetrend(closes, highs, lows)
            sqz_val, sqz_on = self._calc_squeeze(closes, highs, lows)
            ema_fast_vals = self._calc_ema(closes, ema_fast)
            ema_slow_vals = self._calc_ema(closes, ema_slow)
            
            # Create figure with subplots - Larger main chart
            fig = plt.figure(figsize=(self.width, self.height))
            fig.patch.set_facecolor(COLORS['bg'])
            gs = GridSpec(4, 1, height_ratios=[4, 0.8, 0.8, 0.8], hspace=0.08)
            
            # ========== Main Chart (Candlesticks) ==========
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor(COLORS['bg'])
            
            # Draw candlesticks - LARGER
            candle_width = 0.7
            for i in range(len(closes)):
                color = COLORS['green'] if closes[i] >= opens[i] else COLORS['red']
                
                # Wick - thicker
                ax1.plot([i, i], [lows[i], highs[i]], color=color, linewidth=1.2)
                
                # Body
                body_low = min(opens[i], closes[i])
                body_high = max(opens[i], closes[i])
                body_height = body_high - body_low
                if body_height == 0:
                    body_height = (highs[i] - lows[i]) * 0.01
                
                rect = Rectangle(
                    (i - candle_width/2, body_low), 
                    candle_width, 
                    body_height,
                    facecolor=color, 
                    edgecolor=color
                )
                ax1.add_patch(rect)
            
            # Draw EMAs - MUCH THICKER and BRIGHTER
            ema_fast_line = None
            ema_slow_line = None
            
            if len(ema_slow_vals) > 0:
                start_idx = len(closes) - len(ema_slow_vals)
                ema_slow_line, = ax1.plot(
                    range(start_idx, len(closes)), 
                    ema_slow_vals, 
                    color='#ff6600',  # Bright orange
                    linewidth=3.5,
                    label=f'EMA{ema_slow}',
                    alpha=1.0,
                    zorder=5
                )
                # Label on right side
                if len(ema_slow_vals) > 0:
                    ax1.annotate(
                        f'EMA{ema_slow}', 
                        xy=(len(closes) - 1, ema_slow_vals[-1]),
                        xytext=(len(closes) + 0.5, ema_slow_vals[-1]),
                        fontsize=9, 
                        color='#ff6600',
                        fontweight='bold',
                        va='center'
                    )
            
            if len(ema_fast_vals) > 0:
                start_idx = len(closes) - len(ema_fast_vals)
                ema_fast_line, = ax1.plot(
                    range(start_idx, len(closes)), 
                    ema_fast_vals, 
                    color='#00ffff',  # Bright cyan
                    linewidth=3.5, 
                    label=f'EMA{ema_fast}',
                    alpha=1.0,
                    zorder=6
                )
                # Label on right side
                if len(ema_fast_vals) > 0:
                    ax1.annotate(
                        f'EMA{ema_fast}', 
                        xy=(len(closes) - 1, ema_fast_vals[-1]),
                        xytext=(len(closes) + 0.5, ema_fast_vals[-1]),
                        fontsize=9, 
                        color='#00ffff',
                        fontweight='bold',
                        va='center'
                    )
            
            # ========== DRAW TP/SL ZONES (TradingView Style) ==========
            # Calculate zone boundaries for future projection
            zone_start = len(closes) - 1
            zone_end = len(closes) + 15  # Extend into future
            
            current_price = closes[-1]
            
            if entry_price and stop_loss and take_profit_3:
                # Draw zones as rectangles extending into future
                
                if trade_direction == "LONG":
                    # SL Zone (below entry) - Red shaded area
                    sl_rect = Rectangle(
                        (zone_start, stop_loss),
                        zone_end - zone_start,
                        entry_price - stop_loss,
                        facecolor=COLORS['sl_zone'],
                        alpha=0.15,
                        edgecolor='none'
                    )
                    ax1.add_patch(sl_rect)
                    
                    # TP Zone (above entry) - Green shaded area
                    tp_rect = Rectangle(
                        (zone_start, entry_price),
                        zone_end - zone_start,
                        take_profit_3 - entry_price,
                        facecolor=COLORS['tp_zone'],
                        alpha=0.15,
                        edgecolor='none'
                    )
                    ax1.add_patch(tp_rect)
                    
                else:  # SHORT
                    # SL Zone (above entry) - Red shaded area
                    sl_rect = Rectangle(
                        (zone_start, entry_price),
                        zone_end - zone_start,
                        stop_loss - entry_price,
                        facecolor=COLORS['sl_zone'],
                        alpha=0.15,
                        edgecolor='none'
                    )
                    ax1.add_patch(sl_rect)
                    
                    # TP Zone (below entry) - Green shaded area
                    tp_rect = Rectangle(
                        (zone_start, take_profit_3),
                        zone_end - zone_start,
                        entry_price - take_profit_3,
                        facecolor=COLORS['tp_zone'],
                        alpha=0.15,
                        edgecolor='none'
                    )
                    ax1.add_patch(tp_rect)
            
            # Draw trade level lines
            if entry_price:
                ax1.axhline(
                    y=entry_price, 
                    color=COLORS['orange'], 
                    linestyle='-', 
                    linewidth=2.5,
                    xmin=0.7,
                    label=f'Entry: {self._format_price(entry_price)}'
                )
                # Entry price label on right
                ax1.annotate(
                    f'ENTRY {self._format_price(entry_price)}', 
                    xy=(len(closes) + 2, entry_price),
                    fontsize=10, 
                    color=COLORS['orange'],
                    fontweight='bold',
                    va='center'
                )
            
            if stop_loss:
                ax1.axhline(
                    y=stop_loss, 
                    color=COLORS['red_bright'], 
                    linestyle='-', 
                    linewidth=2.5,
                    xmin=0.7,
                    label=f'SL: {self._format_price(stop_loss)}'
                )
                # SL price label
                ax1.annotate(
                    f'SL {self._format_price(stop_loss)}', 
                    xy=(len(closes) + 2, stop_loss),
                    fontsize=10, 
                    color=COLORS['red_bright'],
                    fontweight='bold',
                    va='center'
                )
            
            if take_profit_1:
                ax1.axhline(
                    y=take_profit_1, 
                    color=COLORS['green'], 
                    linestyle='--', 
                    linewidth=1.5, 
                    alpha=0.8,
                    xmin=0.7,
                    label=f'TP1: {self._format_price(take_profit_1)}'
                )
                ax1.annotate(
                    f'TP1 {self._format_price(take_profit_1)}', 
                    xy=(len(closes) + 2, take_profit_1),
                    fontsize=9, 
                    color=COLORS['green'],
                    va='center'
                )
            
            if take_profit_2:
                ax1.axhline(
                    y=take_profit_2, 
                    color=COLORS['green_bright'], 
                    linestyle='--', 
                    linewidth=1.8,
                    xmin=0.7,
                    alpha=0.9,
                    label=f'TP2: {self._format_price(take_profit_2)}'
                )
                ax1.annotate(
                    f'TP2 {self._format_price(take_profit_2)}', 
                    xy=(len(closes) + 2, take_profit_2),
                    fontsize=9, 
                    color=COLORS['green_bright'],
                    va='center'
                )
            
            if take_profit_3:
                ax1.axhline(
                    y=take_profit_3, 
                    color=COLORS['green_bright'], 
                    linestyle='-', 
                    linewidth=2.5,
                    xmin=0.7,
                    label=f'TP3: {self._format_price(take_profit_3)}'
                )
                ax1.annotate(
                    f'TP3 {self._format_price(take_profit_3)}', 
                    xy=(len(closes) + 2, take_profit_3),
                    fontsize=10, 
                    color=COLORS['green_bright'],
                    fontweight='bold',
                    va='center'
                )
            
            # Current price line and label
            ax1.axhline(y=current_price, color=COLORS['white'], linestyle='-', linewidth=1, alpha=0.5)
            
            # Current price box
            ax1.annotate(
                f'{self._format_price(current_price)}', 
                xy=(len(closes) - 1, current_price),
                xytext=(len(closes) + 0.5, current_price),
                fontsize=11, 
                color=COLORS['white'],
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.4', 
                    facecolor=COLORS['green'] if closes[-1] >= opens[-1] else COLORS['red'], 
                    edgecolor='none',
                    alpha=0.9
                ),
                va='center'
            )
            
            # Title with strategy info
            direction_color = COLORS['green'] if trade_direction == "LONG" else COLORS['red']
            direction_text = "LONG" if trade_direction == "LONG" else "SHORT"
            direction_arrow = "+" if trade_direction == "LONG" else "-"
            
            title = f"{symbol} | {interval} | {strategy_name} | [{direction_arrow}] {direction_text}"
            ax1.set_title(
                title, 
                fontsize=14, 
                color=COLORS['text'], 
                fontweight='bold',
                pad=15
            )
            
            # Legend - smaller, top left
            ax1.legend(loc='upper left', fontsize=8, framealpha=0.7, ncol=2)
            
            # Styling
            ax1.tick_params(colors=COLORS['gray'], labelsize=9)
            ax1.grid(True, alpha=0.15, color=COLORS['grid'])
            ax1.set_xlim(-2, len(closes) + 18)  # More space on right for labels
            ax1.set_xticklabels([])
            ax1.set_ylabel('Price (USDT)', fontsize=10, color=COLORS['gray'])
            
            # Format y-axis for small prices
            if current_price < 0.001:
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.8f}'))
            elif current_price < 1:
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.6f}'))
            
            # ========== Volume Chart ==========
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.set_facecolor(COLORS['bg'])
            
            vol_colors = [COLORS['green'] if closes[i] >= opens[i] else COLORS['red'] for i in range(len(closes))]
            ax2.bar(range(len(volumes)), volumes, color=vol_colors, alpha=0.7, width=0.7)
            
            ax2.set_ylabel('Vol', fontsize=9, color=COLORS['gray'])
            ax2.tick_params(colors=COLORS['gray'], labelsize=8)
            ax2.grid(True, alpha=0.1, color=COLORS['grid'])
            ax2.set_xticklabels([])
            
            # ========== WaveTrend Oscillator ==========
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.set_facecolor(COLORS['bg'])
            
            ax3.axhline(y=0, color=COLORS['gray'], linewidth=0.8)
            ax3.axhline(y=60, color=COLORS['red'], linewidth=0.8, linestyle='--', alpha=0.6)
            ax3.axhline(y=-60, color=COLORS['green'], linewidth=0.8, linestyle='--', alpha=0.6)
            
            if len(wt1) > 0 and len(wt2) > 0:
                ax3.fill_between(range(len(wt1)), wt1, wt2, alpha=0.3, color=COLORS['cyan'])
                ax3.plot(wt1, color=COLORS['green'], linewidth=1.2)
                ax3.plot(wt2, color=COLORS['red'], linewidth=1.2)
            
            ax3.set_ylabel('WT', fontsize=9, color=COLORS['gray'])
            ax3.tick_params(colors=COLORS['gray'], labelsize=8)
            ax3.set_ylim(-100, 100)
            ax3.grid(True, alpha=0.1, color=COLORS['grid'])
            ax3.set_xticklabels([])
            
            # ========== Squeeze Momentum ==========
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            ax4.set_facecolor(COLORS['bg'])
            
            ax4.axhline(y=0, color=COLORS['gray'], linewidth=0.8)
            
            if len(sqz_val) > 0:
                # Color bars based on momentum direction and change
                bar_colors = []
                for i in range(len(sqz_val)):
                    if sqz_val[i] > 0:
                        if i == 0 or sqz_val[i] > sqz_val[i-1]:
                            bar_colors.append(COLORS['green_bright'])
                        else:
                            bar_colors.append('#006400')
                    else:
                        if i == 0 or sqz_val[i] < sqz_val[i-1]:
                            bar_colors.append(COLORS['red_bright'])
                        else:
                            bar_colors.append('#8b0000')
                
                ax4.bar(range(len(sqz_val)), sqz_val, color=bar_colors, width=0.7)
                
                # Squeeze dots
                for i in range(len(sqz_on)):
                    dot_color = COLORS['gray'] if sqz_on[i] else COLORS['cyan']
                    ax4.scatter(i, 0, s=15, c=dot_color, marker='o')
            
            ax4.set_ylabel('SQZ', fontsize=9, color=COLORS['gray'])
            ax4.tick_params(colors=COLORS['gray'], labelsize=8)
            ax4.grid(True, alpha=0.1, color=COLORS['grid'])
            
            # Timestamp
            fig.text(
                0.99, 0.01, 
                f'Generated: {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}',
                ha='right', va='bottom', fontsize=9, color=COLORS['gray']
            )
            
            # Adjust layout
            plt.subplots_adjust(left=0.08, right=0.88, top=0.95, bottom=0.05)
            
            # Save to bytes
            buf = io.BytesIO()
            plt.savefig(
                buf, 
                format='png', 
                dpi=self.dpi, 
                facecolor=COLORS['bg'],
                edgecolor='none', 
                bbox_inches='tight'
            )
            plt.close(fig)
            buf.seek(0)
            
            return buf.getvalue()
            
        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            import traceback
            traceback.print_exc()
            return b''
    
    def _format_price(self, price: float) -> str:
        """Format price based on magnitude."""
        if price < 0.0001:
            return f"{price:.8f}"
        elif price < 0.01:
            return f"{price:.6f}"
        elif price < 1:
            return f"{price:.4f}"
        else:
            return f"{price:.4f}"
    
    def _calc_ema(self, data: List[float], period: int) -> List[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema = [0.0] * len(data)
        ema[period - 1] = sum(data[:period]) / period
        
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * multiplier + ema[i - 1]
        
        return ema[period - 1:]
    
    def _sma(self, data: List[float], period: int) -> List[float]:
        """Calculate Simple Moving Average."""
        result = [0.0] * len(data)
        for i in range(len(data)):
            if i >= period - 1:
                result[i] = sum(data[i - period + 1:i + 1]) / period
        return result
    
    def _calc_wavetrend(
        self, 
        closes: List[float], 
        highs: List[float], 
        lows: List[float], 
        n1: int = 10, 
        n2: int = 21
    ) -> Tuple[List[float], List[float]]:
        """Calculate WaveTrend oscillator."""
        try:
            hlc3 = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
            esa = self._calc_ema(hlc3, n1)
            
            if not esa:
                return [], []
            
            # Align hlc3 with esa
            start = len(hlc3) - len(esa)
            aligned_hlc3 = hlc3[start:]
            
            diff = [abs(aligned_hlc3[i] - esa[i]) for i in range(len(esa))]
            d = self._calc_ema(diff, n1)
            
            if not d:
                return [], []
            
            # Align again
            offset = len(esa) - len(d)
            aligned_esa = esa[offset:]
            aligned_hlc3_2 = aligned_hlc3[offset:]
            
            # Calculate CI
            ci = []
            for i in range(len(d)):
                if d[i] != 0:
                    ci.append((aligned_hlc3_2[i] - aligned_esa[i]) / (0.015 * d[i]))
                else:
                    ci.append(0)
            
            wt1 = self._calc_ema(ci, n2)
            
            if not wt1:
                return [], []
            
            wt2 = self._sma(wt1, 4)
            
            # Align wt1 and wt2
            offset2 = len(wt1) - len(wt2)
            if offset2 > 0:
                wt1 = wt1[offset2:]
            
            return wt1, wt2[-len(wt1):] if len(wt2) > len(wt1) else wt2
            
        except Exception as e:
            logger.debug(f"WaveTrend calc error: {e}")
            return [], []
    
    def _calc_squeeze(
        self, 
        closes: List[float], 
        highs: List[float], 
        lows: List[float],
        bb_length: int = 20,
        bb_mult: float = 2.0,
        kc_length: int = 20,
        kc_mult: float = 1.5
    ) -> Tuple[List[float], List[bool]]:
        """Calculate Squeeze Momentum indicator."""
        try:
            n = len(closes)
            if n < bb_length:
                return [], []
            
            # Calculate Bollinger Bands
            sma = self._sma(closes, bb_length)
            std = []
            for i in range(n):
                if i >= bb_length - 1:
                    window = closes[i - bb_length + 1:i + 1]
                    mean = sum(window) / len(window)
                    variance = sum((x - mean) ** 2 for x in window) / len(window)
                    std.append(variance ** 0.5)
                else:
                    std.append(0)
            
            bb_upper = [sma[i] + bb_mult * std[i] for i in range(n)]
            bb_lower = [sma[i] - bb_mult * std[i] for i in range(n)]
            
            # Calculate Keltner Channels
            tr = []
            for i in range(n):
                if i == 0:
                    tr.append(highs[i] - lows[i])
                else:
                    tr.append(max(
                        highs[i] - lows[i],
                        abs(highs[i] - closes[i-1]),
                        abs(lows[i] - closes[i-1])
                    ))
            
            atr = self._sma(tr, kc_length)
            kc_upper = [sma[i] + kc_mult * atr[i] for i in range(n)]
            kc_lower = [sma[i] - kc_mult * atr[i] for i in range(n)]
            
            # Squeeze on = BB inside KC
            sqz_on = []
            for i in range(n):
                sqz_on.append(bb_lower[i] > kc_lower[i] and bb_upper[i] < kc_upper[i])
            
            # Momentum
            highest = []
            lowest = []
            for i in range(n):
                if i >= kc_length - 1:
                    highest.append(max(highs[i - kc_length + 1:i + 1]))
                    lowest.append(min(lows[i - kc_length + 1:i + 1]))
                else:
                    highest.append(highs[i])
                    lowest.append(lows[i])
            
            avg_hl = [(highest[i] + lowest[i]) / 2 for i in range(n)]
            avg_all = [(avg_hl[i] + sma[i]) / 2 for i in range(n)]
            sqz_val = [closes[i] - avg_all[i] for i in range(n)]
            
            return sqz_val, sqz_on
            
        except Exception as e:
            logger.debug(f"Squeeze calc error: {e}")
            return [], []


# Test function
if __name__ == "__main__":
    print("Testing ChartGenerator...")
    generator = ChartGenerator()
    print("Chart generator initialized successfully!")
