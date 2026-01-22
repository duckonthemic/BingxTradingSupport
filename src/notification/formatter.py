"""
Alert Formatter v3.0 - Modern, Clean UI Design

Features:
- Clean visual hierarchy with better spacing
- Risk:Reward visualization with progress bars
- Separated trade execution section
- Mobile-optimized formatting
"""

from typing import Optional, Dict, List
from datetime import datetime

from telegram import InlineKeyboardButton

from ..analysis.indicators import CoinIndicators
from ..analysis.strategy_detector import TradeSetup
from ..analysis.trade_filter import OptimizedLevels


class AlertFormatter:
    """Modern alert formatter with clean UI."""
    
    @staticmethod
    def format_price(price: float) -> str:
        """Format price based on magnitude."""
        if price < 0.0001:
            return f"{price:.8f}"
        elif price < 0.01:
            return f"{price:.6f}"
        elif price < 1:
            return f"{price:.4f}"
        elif price < 100:
            return f"{price:.4f}"
        else:
            return f"{price:,.2f}"
    
    @staticmethod
    def _make_rr_bar(rr: float, max_rr: float = 8.0) -> str:
        """Create visual R:R progress bar."""
        filled = min(int((rr / max_rr) * 10), 10)
        empty = 10 - filled
        return "▓" * filled + "░" * empty
    
    @staticmethod
    def _make_confidence_bar(confidence: float) -> str:
        """Create confidence visual bar."""
        pct = confidence * 100
        if pct >= 80:
            return "🟢🟢🟢🟢🟢"
        elif pct >= 70:
            return "🟢🟢🟢🟢⚪"
        elif pct >= 60:
            return "🟢🟢🟢⚪⚪"
        elif pct >= 50:
            return "🟡🟡⚪⚪⚪"
        else:
            return "🔴⚪⚪⚪⚪"
    
    @staticmethod
    def _get_rsi_status(rsi: float) -> str:
        """Get RSI status emoji."""
        if rsi >= 70:
            return "🔴 Quá mua"
        elif rsi >= 60:
            return "🟡 Cao"
        elif rsi <= 30:
            return "🟢 Quá bán"
        elif rsi <= 40:
            return "🟡 Thấp"
        else:
            return "⚪ Trung tính"
    
    @staticmethod
    def _get_mfi_status(mfi: float) -> str:
        """Get MFI status emoji."""
        if mfi >= 80:
            return "🔴 Quá mua"
        elif mfi <= 20:
            return "🟢 Quá bán"
        else:
            return "⚪ Trung tính"
    
    @staticmethod
    def _get_cci_status(cci: float) -> str:
        """Get CCI status emoji."""
        if cci >= 100:
            return "🔴 Quá mua"
        elif cci <= -100:
            return "🟢 Quá bán"
        else:
            return "⚪ Trung tính"
    
    @staticmethod
    def format_signal_v2(
        symbol: str,
        indicators: CoinIndicators,
        setup: TradeSetup,
        levels: OptimizedLevels,
        funding_rate: Optional[float] = None,
        btc_change_15m: float = 0.0
    ) -> str:
        """
        Format a modern, clean signal with optimized layout.
        """
        sym = symbol.replace("-USDT", "")
        direction_emoji = "🟢" if setup.direction == "LONG" else "🔴"
        direction_word = "LONG ↗" if setup.direction == "LONG" else "SHORT ↘"
        
        # Confidence visualization
        conf_pct = setup.confidence * 100
        conf_bar = AlertFormatter._make_confidence_bar(setup.confidence)
        
        # R:R visualization
        rr_bar = AlertFormatter._make_rr_bar(levels.risk_reward)
        
        # Golden badge
        golden_badge = "🏆 GOLDEN SETUP" if setup.is_golden_setup else ""
        
        # Market mood
        btc_emoji = "🔴" if btc_change_15m < -0.3 else "🟢" if btc_change_15m > 0.3 else "⚪"
        
        # Funding badge
        funding_text = ""
        if funding_rate is not None:
            if funding_rate < -0.01:
                funding_text = "💰 Funding giảm (Short trả phí)"
            elif funding_rate > 0.03:
                funding_text = "⚠️ Funding cao (Cảnh báo Long)"
        
        msg = f"""
{direction_emoji} <b>${sym} {direction_word}</b>
{conf_bar} <code>{conf_pct:.0f}%</code> {golden_badge}
{'─' * 30}

<b>🎯 CHIẾN LƯỢC</b>
{setup.icon} {setup.name}
📍 Zone: <i>{setup.zone_type}</i>

<b>📋 LÝ DO SETUP</b>"""
        
        # Add detailed reasons
        reason_icons = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣"]
        for i, reason in enumerate(setup.reasons[:5]):
            icon = reason_icons[i] if i < len(reason_icons) else "•"
            msg += f"\n{icon} {reason}"
        
        # Confluence signals
        confluence_list = []
        if setup.has_rsi_divergence:
            confluence_list.append("📈 RSI Divergence")
        if setup.has_wavetrend_cross:
            confluence_list.append("🌊 WaveTrend Cross")
        if setup.has_volume_spike:
            confluence_list.append("📊 Volume Spike")
        if setup.has_ob_confluence:
            confluence_list.append("🧱 Order Block Confluence")
        
        if confluence_list:
            msg += f"\n\n<b>✨ CONFLUENCE ({len(confluence_list)}/4)</b>"
            for c in confluence_list:
                msg += f"\n• {c}"
        
        msg += f"""

{'─' * 30}

<b>💹 ENTRY</b>
┌──────────────────────────┐
│ 📍 <b>Entry:</b> <code>{AlertFormatter.format_price(levels.entry)}</code>
│ 
│ 🛑 <b>SL:</b>    <code>{AlertFormatter.format_price(levels.stop_loss)}</code>
│ 
│ ✅ TP1:   <code>{AlertFormatter.format_price(levels.take_profit_1)}</code> ▸ 2R
│ ✅ TP2:   <code>{AlertFormatter.format_price(levels.take_profit_2)}</code> ▸ 4R
│ 🎯 TP3:   <code>{AlertFormatter.format_price(levels.take_profit_3)}</code> ▸ 6R
└──────────────────────────┘

<b>📊 RISK MANAGEMENT</b>
• Leverage: <code>{levels.leverage}x</code>
• Size: <code>${levels.position_size:,.0f}</code>
• Risk: <code>-${levels.risk_usd:.2f}</code>
• Reward: <code>+${levels.reward_usd:.2f}</code>
{rr_bar} <b>R:R = 1:{levels.risk_reward}</b>
• Liquidation: <code>{AlertFormatter.format_price(levels.liquidation_price)}</code>

{'─' * 30}

<b>📈 CHỈ BÁO ĐẦY ĐỦ</b>
┌─────────────────────────────┐
│ <b>Momentum</b>
│ RSI: <code>{indicators.rsi_h1:.0f}</code> {AlertFormatter._get_rsi_status(indicators.rsi_h1)}
│ MFI: <code>{indicators.mfi:.0f}</code> {AlertFormatter._get_mfi_status(indicators.mfi)}
│ CCI: <code>{indicators.cci:.0f}</code> {AlertFormatter._get_cci_status(indicators.cci)}
├─────────────────────────────┤
│ <b>Trend</b>
│ ADX: <code>{indicators.adx:.0f}</code> ({indicators.adx_signal})
│ MACD: {indicators.macd_trend}
│ EMA: {"🟢 Bullish" if indicators.ema34_h1 > indicators.ema89_h1 else "🔴 Bearish"}
├─────────────────────────────┤
│ <b>WaveTrend</b>
│ WT1: <code>{indicators.wt1:.0f}</code> | WT2: <code>{indicators.wt2:.0f}</code>
│ Signal: {indicators.wt_signal}
├─────────────────────────────┤
│ <b>Bollinger Bands</b>
│ Upper: <code>{AlertFormatter.format_price(indicators.bb_upper)}</code>
│ Middle: <code>{AlertFormatter.format_price(indicators.bb_middle)}</code>
│ Lower: <code>{AlertFormatter.format_price(indicators.bb_lower)}</code>
│ Status: {indicators.bb_status}
├─────────────────────────────┤
│ <b>Volume</b>
│ Ratio: <code>{indicators.volume_ratio:.1f}x</code> {"🔥" if indicators.volume_ratio > 1.5 else ""}
│ ATR: <code>{AlertFormatter.format_price(indicators.atr)}</code>
└─────────────────────────────┘

<b>🌐 MARKET CONTEXT</b>
{btc_emoji} BTC 15m: <code>{btc_change_15m:+.2f}%</code>
📊 Trend H1: {indicators.trend_h1} | H4: {indicators.trend_h4}"""

        if funding_rate is not None:
            fr_emoji = "📉" if funding_rate < 0 else "📈"
            msg += f"\n{fr_emoji} Funding: <code>{funding_rate:+.4f}%</code>"
            if funding_text:
                msg += f"\n{funding_text}"
        
        # Add warnings section if any
        if setup.warnings:
            msg += f"""

<b>⚠️ CẢNH BÁO</b>"""
            for warning in setup.warnings[:2]:
                msg += f"\n• {warning}"
        
        msg += f"""

⏰ <i>{datetime.now().strftime('%H:%M:%S │ %d/%m/%Y')}</i>"""
        
        return msg.strip()
    
    @staticmethod
    def format_quick_analysis(
        symbol: str,
        indicators: CoinIndicators,
        btc_change_15m: float = 0.0
    ) -> str:
        """Format quick analysis (for /ana command)."""
        sym = symbol.replace("-USDT", "")
        
        # Determine bias
        above_ema89 = indicators.price > indicators.ema89_h1
        bias_emoji = "🟢" if above_ema89 else "🔴"
        bias_text = "BULLISH" if above_ema89 else "BEARISH"
        
        # RSI zone
        if indicators.rsi_h1 > 70:
            rsi_zone = "🔴 Quá mua"
        elif indicators.rsi_h1 < 30:
            rsi_zone = "🟢 Quá bán"
        else:
            rsi_zone = "⚪ Trung tính"
        
        # BTC status
        btc_emoji = "🔴" if btc_change_15m < -0.3 else "🟢" if btc_change_15m > 0.3 else "⚪"
        
        msg = f"""
📊 <b>PHÂN TÍCH ${sym}</b>
{'═' * 30}

<b>💰 GIÁ HIỆN TẠI</b>
<code>{AlertFormatter.format_price(indicators.price)}</code>

<b>📈 XU HƯỚNG</b>
• EMA89 H1: <code>{AlertFormatter.format_price(indicators.ema89_h1)}</code>
• Vị trí: {'Trên EMA ↗' if above_ema89 else 'Dưới EMA ↘'}
• Bias: {bias_emoji} <b>{bias_text}</b>

<b>📊 CHỈ BÁO</b>
┌─────────────────────┐
│ RSI:  <code>{indicators.rsi_h1:.0f}</code> {rsi_zone}
│ MFI:  <code>{indicators.mfi:.0f}</code> {indicators.mfi_status}
│ MACD: {indicators.macd_trend}
│ ADX:  <code>{indicators.adx:.0f}</code>
│ WT:   {indicators.wt_signal}
│ Vol:  <code>{indicators.volume_ratio:.1f}x</code>
└─────────────────────┘

<b>📍 MỨC GIÁ QUAN TRỌNG</b>
• Swing High: <code>{AlertFormatter.format_price(indicators.swing_high_20)}</code>
• Swing Low: <code>{AlertFormatter.format_price(indicators.swing_low_20)}</code>

<b>🌐 CONTEXT</b>
• Trend H1: {indicators.trend_h1}
• Trend H4: {indicators.trend_h4}
{btc_emoji} BTC 15m: <code>{btc_change_15m:+.2f}%</code>

<b>💡 GỢI Ý</b>
{"✅ LONG OK - Giá trên EMA89" if above_ema89 and btc_change_15m >= -0.5 else ""}
{"✅ SHORT OK - Giá dưới EMA89" if not above_ema89 else ""}
{"⚠️ TRÁNH LONG - BTC đang dump" if btc_change_15m < -0.5 else ""}

⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        return msg.strip()
    
    @staticmethod
    def get_inline_buttons(symbol: str) -> List[List[InlineKeyboardButton]]:
        """Get inline keyboard buttons for the alert - Fixed to use correct symbol."""
        # Clean symbol format
        sym = symbol.replace("-USDT", "").replace("USDT", "").upper()
        
        buttons = [
            [
                InlineKeyboardButton(
                    text="📈 Chart TradingView",
                    url=f"https://www.tradingview.com/chart/?symbol=BINANCE:{sym}USDT.P"
                ),
                InlineKeyboardButton(
                    text="💹 Trade BingX",
                    url=f"https://bingx.com/vi-vn/perpetual/{sym}-USDT"
                )
            ]
        ]
        
        return buttons
    
    @staticmethod
    def format_btc_status(
        current_price: float,
        change_15m: float,
        change_1h: float,
        change_4h: float,
        change_24h: float
    ) -> str:
        """Format BTC status message."""
        if change_15m < -0.5:
            mood = "🔴 DUMPING"
            mood_desc = "Altcoin có thể bị kéo theo"
        elif change_15m > 0.5:
            mood = "🟢 BULLISH"
            mood_desc = "Thị trường tích cực"
        else:
            mood = "⚪ NEUTRAL"
            mood_desc = "Thị trường ổn định"
        
        msg = f"""
₿ <b>TÌNH TRẠNG BTC</b>
{'═' * 30}

<b>💰 Giá:</b> <code>${current_price:,.2f}</code>

<b>📊 Biến động:</b>
┌──────────────────┐
│ 15m:  <code>{change_15m:+.2f}%</code>
│ 1h:   <code>{change_1h:+.2f}%</code>  
│ 4h:   <code>{change_4h:+.2f}%</code>
│ 24h:  <code>{change_24h:+.2f}%</code>
└──────────────────┘

<b>🌡️ TÂM LÝ:</b> {mood}
<i>{mood_desc}</i>

<b>💡 Khuyến nghị:</b>
{"⚠️ Cẩn thận với lệnh LONG" if change_15m < -0.5 else "✅ Giao dịch bình thường"}

⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        return msg.strip()
    
    @staticmethod
    def format_status(
        running: bool,
        paused: bool,
        btc_dumping: bool,
        btc_change: str,
        circuit_state: str,
        scans: int,
        coins_scanned: int,
        setups_found: int,
        alerts_sent: int,
        filtered_btc: int,
        filtered_mtf: int,
        filtered_rr: int
    ) -> str:
        """Format bot status message."""
        running_emoji = "🟢" if running else "🔴"
        paused_emoji = "⏸️" if paused else "▶️"
        btc_emoji = "🔴" if btc_dumping else "🟢"
        circuit_emoji = "🟢" if circuit_state == "closed" else "🔴"
        
        msg = f"""
📊 <b>TRẠNG THÁI BOT</b>
{'═' * 30}

<b>🔧 HỆ THỐNG</b>
{running_emoji} Running: {'Có' if running else 'Không'}
{paused_emoji} Paused: {'Có' if paused else 'Không'}
{circuit_emoji} Circuit: {circuit_state}

<b>₿ BITCOIN</b>
{btc_emoji} {"DUMP MODE - Block LONG" if btc_dumping else "Bình thường"}
Thay đổi 15m: {btc_change}

<b>📈 PHIÊN LÀM VIỆC</b>
┌──────────────────┐
│ Scans:   <code>{scans}</code>
│ Coins:   <code>{coins_scanned}</code>
│ Setups:  <code>{setups_found}</code>
│ Alerts:  <code>{alerts_sent}</code>
└──────────────────┘

<b>🚫 ĐÃ LỌC</b>
• BTC Dump: {filtered_btc}
• MTF Trend: {filtered_mtf}
• R:R thấp: {filtered_rr}

⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
        return msg.strip()
