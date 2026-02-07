"""
Chart Generator v4.0 - Professional TradingView-Style Charts

Improvements over v3.0:
- RSI panel with overbought/oversold zones
- Bollinger Bands overlay on main chart
- Swing high/low level markers
- Separate TP1/TP2/TP3 gradient bands
- R:R ratio display + direction badge
- Volume SMA reference line
- Better Y-axis auto-padding
- Cleaner label positioning
"""

import io
import logging
from datetime import datetime
from typing import List, Optional, Dict, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
import numpy as np

logger = logging.getLogger(__name__)

COLORS = {
    'bg': '#131722',
    'panel_bg': '#1a1e2e',
    'grid': '#1e222d',
    'text': '#d1d4dc',
    'text_dim': '#787b86',
    'green': '#26a69a',
    'red': '#ef5350',
    'green_bright': '#00e676',
    'red_bright': '#ff1744',
    'green_dark': '#1b5e20',
    'red_dark': '#b71c1c',
    'blue': '#2962ff',
    'orange': '#ff9800',
    'yellow': '#ffeb3b',
    'purple': '#ab47bc',
    'cyan': '#00bcd4',
    'white': '#ffffff',
    'gray': '#787b86',
    'tp1_zone': '#1b5e20',
    'tp2_zone': '#2e7d32',
    'tp3_zone': '#388e3c',
    'sl_zone': '#b71c1c',
    'bb_fill': '#2962ff',
    'bb_line': '#5c85d6',
    'swing_high': '#ff9800',
    'swing_low': '#00bcd4',
}

STRATEGY_ICONS = {
    'SFP': '\U0001f504',
    'LIQ_SWEEP': '\U0001f30a',
    'SILVER_BULLET': '\U0001f3af',
    'UNICORN': '\U0001f984',
    'TURTLE_SOUP': '\U0001f422',
    'IE': '\U0001f4d0',
}


class ChartGenerator:
    def __init__(self):
        self.width = 18
        self.height = 14
        self.dpi = 150
        plt.style.use('dark_background')

    def generate_chart(
        self,
        symbol: str,
        klines: List,
        interval: str = "1h",
        num_candles: int = 60,
        ema_fast: int = 34,
        ema_slow: int = 89,
        entry_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit_1: Optional[float] = None,
        take_profit_2: Optional[float] = None,
        take_profit_3: Optional[float] = None,
        trade_direction: str = "LONG",
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        trend: str = "NEUTRAL",
        zone: str = "EQUILIBRIUM",
        strategy_name: str = "",
    ) -> bytes:
        try:
            klines_list = list(klines)
            if klines_list and isinstance(klines_list[0], dict):
                klines_list = sorted(klines_list, key=lambda x: int(x.get('time', 0)))
            klines_list = klines_list[-num_candles:] if len(klines_list) > num_candles else klines_list

            times, opens, highs, lows, closes, volumes = [], [], [], [], [], []
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

            n = len(closes)
            current_price = closes[-1]

            ema_f = self._calc_ema(closes, ema_fast)
            ema_s = self._calc_ema(closes, ema_slow)
            bb_mid, bb_upper, bb_lower = self._calc_bb(closes, 20, 2.0)
            rsi_vals = self._calc_rsi(closes, 14)
            wt1, wt2 = self._calc_wavetrend(closes, highs, lows)
            sqz_val, sqz_on = self._calc_squeeze(closes, highs, lows)
            vol_sma = self._sma(volumes, 20)

            fig = plt.figure(figsize=(self.width, self.height))
            fig.patch.set_facecolor(COLORS['bg'])
            gs = GridSpec(5, 1, height_ratios=[5.5, 1.2, 1.2, 1.2, 0.9], hspace=0.06)

            # ============ 1. MAIN CHART ============
            ax1 = fig.add_subplot(gs[0])
            ax1.set_facecolor(COLORS['bg'])
            cw = 0.65

            if len(bb_upper) > 0:
                bb_off = n - len(bb_upper)
                x_bb = range(bb_off, n)
                ax1.fill_between(x_bb, bb_lower, bb_upper, alpha=0.06, color=COLORS['bb_fill'])
                ax1.plot(x_bb, bb_upper, color=COLORS['bb_line'], linewidth=0.7, alpha=0.4, linestyle='--')
                ax1.plot(x_bb, bb_lower, color=COLORS['bb_line'], linewidth=0.7, alpha=0.4, linestyle='--')
                ax1.plot(x_bb, bb_mid, color=COLORS['bb_line'], linewidth=0.5, alpha=0.25, linestyle=':')

            for i in range(n):
                c = COLORS['green'] if closes[i] >= opens[i] else COLORS['red']
                ax1.plot([i, i], [lows[i], highs[i]], color=c, linewidth=1.0)
                blo = min(opens[i], closes[i])
                bhi = max(opens[i], closes[i])
                bh = bhi - blo or (highs[i] - lows[i]) * 0.01
                ax1.add_patch(Rectangle((i - cw / 2, blo), cw, bh, facecolor=c, edgecolor=c))

            if ema_s:
                off = n - len(ema_s)
                ax1.plot(range(off, n), ema_s, color='#ff6600', linewidth=2.0, label=f'EMA{ema_slow}', zorder=5)
                ax1.annotate(f'EMA{ema_slow}', xy=(n - 1, ema_s[-1]), xytext=(n + 0.5, ema_s[-1]),
                             fontsize=8, color='#ff6600', fontweight='bold', va='center')
            if ema_f:
                off = n - len(ema_f)
                ax1.plot(range(off, n), ema_f, color='#00e5ff', linewidth=2.0, label=f'EMA{ema_fast}', zorder=6)
                ax1.annotate(f'EMA{ema_fast}', xy=(n - 1, ema_f[-1]), xytext=(n + 0.5, ema_f[-1]),
                             fontsize=8, color='#00e5ff', fontweight='bold', va='center')

            if swing_high and swing_high > 0:
                ax1.axhline(y=swing_high, color=COLORS['swing_high'], linestyle=':', linewidth=1.0, alpha=0.6)
                ax1.annotate(f'Swing H {self._fp(swing_high)}', xy=(2, swing_high),
                             fontsize=7, color=COLORS['swing_high'], alpha=0.8, va='bottom')
            if swing_low and swing_low > 0:
                ax1.axhline(y=swing_low, color=COLORS['swing_low'], linestyle=':', linewidth=1.0, alpha=0.6)
                ax1.annotate(f'Swing L {self._fp(swing_low)}', xy=(2, swing_low),
                             fontsize=7, color=COLORS['swing_low'], alpha=0.8, va='top')

            zs = n - 1
            ze = n + 16
            if entry_price and stop_loss:
                if trade_direction == "LONG":
                    ax1.add_patch(Rectangle((zs, stop_loss), ze - zs, entry_price - stop_loss,
                                            facecolor=COLORS['sl_zone'], alpha=0.18))
                else:
                    ax1.add_patch(Rectangle((zs, entry_price), ze - zs, stop_loss - entry_price,
                                            facecolor=COLORS['sl_zone'], alpha=0.18))
                if take_profit_1:
                    if trade_direction == "LONG":
                        ax1.add_patch(Rectangle((zs, entry_price), ze - zs, take_profit_1 - entry_price,
                                                facecolor=COLORS['tp1_zone'], alpha=0.15))
                    else:
                        ax1.add_patch(Rectangle((zs, take_profit_1), ze - zs, entry_price - take_profit_1,
                                                facecolor=COLORS['tp1_zone'], alpha=0.15))
                if take_profit_1 and take_profit_2:
                    if trade_direction == "LONG":
                        ax1.add_patch(Rectangle((zs, take_profit_1), ze - zs, take_profit_2 - take_profit_1,
                                                facecolor=COLORS['tp2_zone'], alpha=0.15))
                    else:
                        ax1.add_patch(Rectangle((zs, take_profit_2), ze - zs, take_profit_1 - take_profit_2,
                                                facecolor=COLORS['tp2_zone'], alpha=0.15))
                if take_profit_2 and take_profit_3:
                    if trade_direction == "LONG":
                        ax1.add_patch(Rectangle((zs, take_profit_2), ze - zs, take_profit_3 - take_profit_2,
                                                facecolor=COLORS['tp3_zone'], alpha=0.15))
                    else:
                        ax1.add_patch(Rectangle((zs, take_profit_3), ze - zs, take_profit_2 - take_profit_3,
                                                facecolor=COLORS['tp3_zone'], alpha=0.15))

            label_x = n + 2
            if entry_price:
                ax1.axhline(y=entry_price, color=COLORS['orange'], linewidth=2.0, xmin=0.65, zorder=7)
                ax1.annotate(f'  ENTRY {self._fp(entry_price)}', xy=(label_x, entry_price),
                             fontsize=9, color=COLORS['orange'], fontweight='bold', va='center')
            if stop_loss:
                ax1.axhline(y=stop_loss, color=COLORS['red_bright'], linewidth=2.0, xmin=0.65, zorder=7)
                ax1.annotate(f'  SL {self._fp(stop_loss)}', xy=(label_x, stop_loss),
                             fontsize=9, color=COLORS['red_bright'], fontweight='bold', va='center')
            if take_profit_1:
                ax1.axhline(y=take_profit_1, color=COLORS['green'], linewidth=1.2, linestyle='--', xmin=0.65, alpha=0.8)
                ax1.annotate(f'TP1 {self._fp(take_profit_1)}', xy=(label_x, take_profit_1),
                             fontsize=8, color=COLORS['green'], va='center')
            if take_profit_2:
                ax1.axhline(y=take_profit_2, color=COLORS['green_bright'], linewidth=1.5, linestyle='--', xmin=0.65, alpha=0.9)
                ax1.annotate(f'TP2 {self._fp(take_profit_2)}', xy=(label_x, take_profit_2),
                             fontsize=8, color=COLORS['green_bright'], va='center')
            if take_profit_3:
                ax1.axhline(y=take_profit_3, color=COLORS['green_bright'], linewidth=2.0, xmin=0.65)
                ax1.annotate(f'TP3 {self._fp(take_profit_3)}', xy=(label_x, take_profit_3),
                             fontsize=9, color=COLORS['green_bright'], fontweight='bold', va='center')

            ax1.axhline(y=current_price, color=COLORS['white'], linewidth=0.8, alpha=0.4)
            box_color = COLORS['green'] if closes[-1] >= opens[-1] else COLORS['red']
            ax1.annotate(self._fp(current_price), xy=(n - 1, current_price),
                         xytext=(n + 0.3, current_price), fontsize=10, color='white', fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.35', facecolor=box_color, edgecolor='none', alpha=0.9),
                         va='center', zorder=10)

            if entry_price and stop_loss and take_profit_1:
                risk = abs(entry_price - stop_loss)
                reward = abs(take_profit_1 - entry_price)
                if risk > 0:
                    rr = reward / risk
                    rr_text = f'R:R = {rr:.1f}:1'
                    ax1.text(0.98, 0.03, rr_text, transform=ax1.transAxes, fontsize=11,
                             color=COLORS['yellow'], fontweight='bold', ha='right', va='bottom',
                             bbox=dict(boxstyle='round,pad=0.4', facecolor='#1a1a2e',
                                       edgecolor=COLORS['yellow'], alpha=0.85))

            icon = STRATEGY_ICONS.get(strategy_name, '\U0001f4ca')
            dir_color = COLORS['green'] if trade_direction == "LONG" else COLORS['red']
            arrow = '\u25b2' if trade_direction == "LONG" else '\u25bc'
            badge_text = f'{icon} {strategy_name}  {arrow} {trade_direction}'
            ax1.text(0.01, 0.97, badge_text, transform=ax1.transAxes, fontsize=12,
                     color='white', fontweight='bold', ha='left', va='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor=dir_color, edgecolor='none', alpha=0.85))

            title = f'{symbol}  \u2022  {interval}  \u2022  {trend}'
            ax1.set_title(title, fontsize=13, color=COLORS['text'], fontweight='bold', pad=12)

            if current_price < 0.001:
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.8f}'))
            elif current_price < 1:
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.6f}'))
            elif current_price < 100:
                ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

            all_prices = highs + lows
            if entry_price: all_prices.append(entry_price)
            if stop_loss: all_prices.append(stop_loss)
            if take_profit_3: all_prices.append(take_profit_3)
            elif take_profit_2: all_prices.append(take_profit_2)
            elif take_profit_1: all_prices.append(take_profit_1)
            ymin_v, ymax_v = min(all_prices), max(all_prices)
            pad = (ymax_v - ymin_v) * 0.08
            ax1.set_ylim(ymin_v - pad, ymax_v + pad)
            ax1.set_xlim(-2, n + 18)
            ax1.set_xticklabels([])
            ax1.tick_params(colors=COLORS['gray'], labelsize=8)
            ax1.grid(True, alpha=0.12, color=COLORS['grid'])
            ax1.set_ylabel('Price', fontsize=9, color=COLORS['gray'])

            # ============ 2. VOLUME ============
            ax2 = fig.add_subplot(gs[1], sharex=ax1)
            ax2.set_facecolor(COLORS['bg'])
            vc = [COLORS['green'] if closes[i] >= opens[i] else COLORS['red'] for i in range(n)]
            ax2.bar(range(n), volumes, color=vc, alpha=0.65, width=0.7)
            if vol_sma:
                off = n - len(vol_sma)
                ax2.plot(range(off, n), vol_sma, color=COLORS['yellow'], linewidth=1.0, alpha=0.7)
            ax2.set_ylabel('Vol', fontsize=8, color=COLORS['gray'])
            ax2.tick_params(colors=COLORS['gray'], labelsize=7)
            ax2.grid(True, alpha=0.08, color=COLORS['grid'])
            ax2.set_xticklabels([])

            # ============ 3. RSI ============
            ax3 = fig.add_subplot(gs[2], sharex=ax1)
            ax3.set_facecolor(COLORS['bg'])
            ax3.axhline(y=70, color=COLORS['red'], linewidth=0.7, linestyle='--', alpha=0.5)
            ax3.axhline(y=30, color=COLORS['green'], linewidth=0.7, linestyle='--', alpha=0.5)
            ax3.axhline(y=50, color=COLORS['gray'], linewidth=0.5, alpha=0.3)
            ax3.axhspan(70, 100, alpha=0.05, color=COLORS['red'])
            ax3.axhspan(0, 30, alpha=0.05, color=COLORS['green'])
            if rsi_vals:
                rsi_off = n - len(rsi_vals)
                rsi_x = list(range(rsi_off, n))
                ax3.plot(rsi_x, rsi_vals, color='#e040fb', linewidth=1.3)
                ax3.annotate(f'{rsi_vals[-1]:.0f}', xy=(n - 1, rsi_vals[-1]),
                             xytext=(n + 0.5, rsi_vals[-1]), fontsize=8,
                             color='#e040fb', fontweight='bold', va='center')
            ax3.set_ylabel('RSI', fontsize=8, color=COLORS['gray'])
            ax3.set_ylim(0, 100)
            ax3.tick_params(colors=COLORS['gray'], labelsize=7)
            ax3.grid(True, alpha=0.08, color=COLORS['grid'])
            ax3.set_xticklabels([])

            # ============ 4. WAVETREND ============
            ax4 = fig.add_subplot(gs[3], sharex=ax1)
            ax4.set_facecolor(COLORS['bg'])
            ax4.axhline(y=0, color=COLORS['gray'], linewidth=0.7)
            ax4.axhline(y=60, color=COLORS['red'], linewidth=0.7, linestyle='--', alpha=0.5)
            ax4.axhline(y=-60, color=COLORS['green'], linewidth=0.7, linestyle='--', alpha=0.5)
            if wt1 and wt2:
                wt_x = range(len(wt1))
                ax4.fill_between(wt_x, wt1, wt2, alpha=0.25, color=COLORS['cyan'])
                ax4.plot(wt_x, wt1, color=COLORS['green'], linewidth=1.1)
                ax4.plot(wt_x, wt2, color=COLORS['red'], linewidth=1.1)
            ax4.set_ylabel('WT', fontsize=8, color=COLORS['gray'])
            ax4.set_ylim(-100, 100)
            ax4.tick_params(colors=COLORS['gray'], labelsize=7)
            ax4.grid(True, alpha=0.08, color=COLORS['grid'])
            ax4.set_xticklabels([])

            # ============ 5. SQUEEZE MOMENTUM ============
            ax5 = fig.add_subplot(gs[4], sharex=ax1)
            ax5.set_facecolor(COLORS['bg'])
            ax5.axhline(y=0, color=COLORS['gray'], linewidth=0.7)
            if sqz_val:
                bar_c = []
                for i in range(len(sqz_val)):
                    if sqz_val[i] > 0:
                        bar_c.append(COLORS['green_bright'] if i == 0 or sqz_val[i] > sqz_val[i - 1] else '#006400')
                    else:
                        bar_c.append(COLORS['red_bright'] if i == 0 or sqz_val[i] < sqz_val[i - 1] else '#8b0000')
                ax5.bar(range(len(sqz_val)), sqz_val, color=bar_c, width=0.7)
                for i in range(len(sqz_on)):
                    ax5.scatter(i, 0, s=12, c=(COLORS['gray'] if sqz_on[i] else COLORS['cyan']), marker='o', zorder=5)
            ax5.set_ylabel('SQZ', fontsize=8, color=COLORS['gray'])
            ax5.tick_params(colors=COLORS['gray'], labelsize=7)
            ax5.grid(True, alpha=0.08, color=COLORS['grid'])

            fig.text(0.99, 0.005,
                     f'BingX Alert Bot \u2022 {datetime.now().strftime("%H:%M:%S %d/%m/%Y")}',
                     ha='right', va='bottom', fontsize=8, color=COLORS['text_dim'])

            plt.subplots_adjust(left=0.07, right=0.87, top=0.96, bottom=0.03)

            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=self.dpi, facecolor=COLORS['bg'], edgecolor='none', bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return buf.getvalue()

        except Exception as e:
            logger.error(f"Chart generation error: {e}")
            import traceback
            traceback.print_exc()
            return b''

    def _fp(self, price: float) -> str:
        if price < 0.0001:
            return f"{price:.8f}"
        elif price < 0.01:
            return f"{price:.6f}"
        elif price < 1:
            return f"{price:.4f}"
        elif price < 100:
            return f"{price:.4f}"
        elif price < 10000:
            return f"{price:.2f}"
        else:
            return f"{price:.1f}"

    def _calc_ema(self, data: List[float], period: int) -> List[float]:
        if len(data) < period:
            return []
        mult = 2 / (period + 1)
        ema = [0.0] * len(data)
        ema[period - 1] = sum(data[:period]) / period
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1]
        return ema[period - 1:]

    def _sma(self, data: List[float], period: int) -> List[float]:
        r = []
        for i in range(len(data)):
            if i >= period - 1:
                r.append(sum(data[i - period + 1:i + 1]) / period)
        return r

    def _calc_bb(self, closes, period=20, mult=2.0):
        n = len(closes)
        if n < period:
            return [], [], []
        sma, upper, lower = [], [], []
        for i in range(n):
            if i >= period - 1:
                w = closes[i - period + 1:i + 1]
                m = sum(w) / len(w)
                std = (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5
                sma.append(m)
                upper.append(m + mult * std)
                lower.append(m - mult * std)
        return sma, upper, lower

    def _calc_rsi(self, closes: List[float], period: int = 14) -> List[float]:
        if len(closes) < period + 1:
            return []
        deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0.0 for d in deltas]
        losses = [-d if d < 0 else 0.0 for d in deltas]
        avg_g = sum(gains[:period]) / period
        avg_l = sum(losses[:period]) / period
        rsi = []
        for i in range(period, len(deltas)):
            avg_g = (avg_g * (period - 1) + gains[i]) / period
            avg_l = (avg_l * (period - 1) + losses[i]) / period
            rs = avg_g / avg_l if avg_l != 0 else 100
            rsi.append(100 - 100 / (1 + rs))
        return rsi

    def _calc_wavetrend(self, closes, highs, lows, n1=10, n2=21):
        try:
            hlc3 = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
            esa = self._calc_ema(hlc3, n1)
            if not esa:
                return [], []
            start = len(hlc3) - len(esa)
            aligned = hlc3[start:]
            diff = [abs(aligned[i] - esa[i]) for i in range(len(esa))]
            d = self._calc_ema(diff, n1)
            if not d:
                return [], []
            off = len(esa) - len(d)
            a_esa = esa[off:]
            a_hlc = aligned[off:]
            ci = [(a_hlc[i] - a_esa[i]) / (0.015 * d[i]) if d[i] != 0 else 0 for i in range(len(d))]
            wt1 = self._calc_ema(ci, n2)
            if not wt1:
                return [], []
            sma_raw = []
            for i in range(len(wt1)):
                if i >= 3:
                    sma_raw.append(sum(wt1[i - 3:i + 1]) / 4)
                else:
                    sma_raw.append(wt1[i])
            return wt1, sma_raw
        except Exception as e:
            logger.debug(f"WT error: {e}")
            return [], []

    def _calc_squeeze(self, closes, highs, lows, bb_len=20, bb_m=2.0, kc_len=20, kc_m=1.5):
        try:
            n = len(closes)
            if n < bb_len:
                return [], []
            sma_raw = [0.0] * n
            std_raw = [0.0] * n
            for i in range(n):
                if i >= bb_len - 1:
                    w = closes[i - bb_len + 1:i + 1]
                    m = sum(w) / len(w)
                    sma_raw[i] = m
                    std_raw[i] = (sum((x - m) ** 2 for x in w) / len(w)) ** 0.5
            bb_u = [sma_raw[i] + bb_m * std_raw[i] for i in range(n)]
            bb_l = [sma_raw[i] - bb_m * std_raw[i] for i in range(n)]
            tr = [highs[0] - lows[0]]
            for i in range(1, n):
                tr.append(max(highs[i] - lows[i], abs(highs[i] - closes[i - 1]), abs(lows[i] - closes[i - 1])))
            atr_raw = [0.0] * n
            for i in range(n):
                if i >= kc_len - 1:
                    atr_raw[i] = sum(tr[i - kc_len + 1:i + 1]) / kc_len
            kc_u = [sma_raw[i] + kc_m * atr_raw[i] for i in range(n)]
            kc_l = [sma_raw[i] - kc_m * atr_raw[i] for i in range(n)]
            sqz_on = [bb_l[i] > kc_l[i] and bb_u[i] < kc_u[i] for i in range(n)]
            hh = [0.0] * n
            ll = [0.0] * n
            for i in range(n):
                if i >= kc_len - 1:
                    hh[i] = max(highs[i - kc_len + 1:i + 1])
                    ll[i] = min(lows[i - kc_len + 1:i + 1])
                else:
                    hh[i] = highs[i]
                    ll[i] = lows[i]
            avg_hl = [(hh[i] + ll[i]) / 2 for i in range(n)]
            avg_all = [(avg_hl[i] + sma_raw[i]) / 2 for i in range(n)]
            sqz_val = [closes[i] - avg_all[i] for i in range(n)]
            return sqz_val, sqz_on
        except Exception as e:
            logger.debug(f"SQZ error: {e}")
            return [], []


if __name__ == "__main__":
    print("ChartGenerator v4.0 ready")
