"""
Telegram Command Handler - Interactive commands for bot control.

Commands:
- /status - Current bot status
- /stats - Statistics summary
- /pause - Pause scanning
- /resume - Resume scanning
- /ana [symbol] - Quick analysis of a coin
- /btc - BTC mood check
- /coins - Top coins by volume
- /news - Economic calendar
- /help - Help message
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional, Callable, Awaitable

import pytz

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from ..config import config
from ..ingestion.rest_client import BingXRestClient
from ..analysis.indicators import IndicatorCalculator
from ..analysis.trade_filter import TradeFilter, TradeDirection

logger = logging.getLogger(__name__)


class TelegramCommandHandler:
    """Handles Telegram bot commands."""
    
    def __init__(
        self,
        alert_manager,  # Reference to AlertManager
        rest_client: BingXRestClient,
        trade_filter: TradeFilter
    ):
        self.alert_manager = alert_manager
        self.rest_client = rest_client
        self.trade_filter = trade_filter
        self.indicator_calc = IndicatorCalculator()
        
        self._app: Optional[Application] = None
        self._chat_id = config.telegram.chat_id
    
    async def setup(self, app: Application):
        """Setup command handlers."""
        self._app = app
        
        # Register commands
        app.add_handler(CommandHandler("status", self.cmd_status))
        app.add_handler(CommandHandler("stats", self.cmd_stats))
        app.add_handler(CommandHandler("pause", self.cmd_pause))
        app.add_handler(CommandHandler("resume", self.cmd_resume))
        app.add_handler(CommandHandler("ana", self.cmd_analyze))
        app.add_handler(CommandHandler("btc", self.cmd_btc))
        app.add_handler(CommandHandler("coins", self.cmd_coins))
        app.add_handler(CommandHandler("news", self.cmd_news))
        app.add_handler(CommandHandler("session", self.cmd_session))
        app.add_handler(CommandHandler("help", self.cmd_help))
        app.add_handler(CommandHandler("start", self.cmd_help))
        
        # Set commands for menu
        commands = [
            BotCommand("status", "Current bot status"),
            BotCommand("stats", "Statistics summary"),
            BotCommand("pause", "Pause scanning"),
            BotCommand("resume", "Resume scanning"),
            BotCommand("ana", "Quick analysis: /ana BTC"),
            BotCommand("btc", "BTC mood check"),
            BotCommand("coins", "Top coins by volume"),
            BotCommand("news", "Economic calendar"),
            BotCommand("session", "Trading sessions status"),
            BotCommand("help", "Show help"),
        ]
        
        try:
            await app.bot.set_my_commands(commands)
        except Exception as e:
            logger.warning(f"Failed to set commands: {e}")
    
    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show current bot status."""
        try:
            stats = self.alert_manager.stats
            
            btc_status = "🔴 DUMP MODE" if stats["btc_dumping"] else "🟢 Normal"
            running_emoji = "🟢" if stats["running"] else "🔴"
            paused_emoji = "⏸️" if stats["paused"] else "▶️"
            circuit_emoji = "🟢" if stats["circuit_state"] == "closed" else "🔴"
            
            msg = f"""
📊 <b>BOT STATUS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

{running_emoji} Running: {'Yes' if stats["running"] else 'No'}
{paused_emoji} Paused: {'Yes' if stats["paused"] else 'No'}
{circuit_emoji} Circuit: {stats["circuit_state"]}

<b>🔗 BTC Status:</b>
{btc_status}
Change 15m: {stats["btc_change_15m"]}

<b>📈 Session Stats:</b>
Scans: {stats["scans"]}
Coins Scanned: {stats["coins_scanned"]}
Setups Found: {stats["setups_detected"]}
Alerts Sent: {stats["alerts_sent"]}

<b>🚫 Filtered:</b>
BTC Dump: {stats["filtered_by_btc"]}
MTF Trend: {stats["filtered_by_trend"]}
Low Score: {stats["filtered_by_score"]}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Status command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed statistics."""
        try:
            stats = self.alert_manager.stats
            
            # Strategy breakdown
            strategy_lines = []
            for strategy, count in sorted(stats["strategies"].items(), 
                                          key=lambda x: x[1], reverse=True):
                strategy_lines.append(f"• {strategy}: {count}")
            
            strategy_text = "\n".join(strategy_lines) if strategy_lines else "No detections yet"
            
            msg = f"""
📈 <b>DETAILED STATISTICS</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>Total Scans:</b> {stats["scans"]}
<b>Total Coins:</b> {stats["coins_scanned"]}
<b>Setups Found:</b> {stats["setups_detected"]}
<b>Alerts Sent:</b> {stats["alerts_sent"]}

<b>Strategy Detections:</b>
{strategy_text}

<b>Filter Statistics:</b>
• BTC Dump Filter: {stats["filtered_by_btc"]}
• MTF Trend Filter: {stats["filtered_by_trend"]}
• Low Score Filter: {stats["filtered_by_score"]}

<b>Efficiency:</b>
• Detection Rate: {stats["setups_detected"]/max(1, stats["coins_scanned"])*100:.1f}%
• Alert Rate: {stats["alerts_sent"]/max(1, stats["setups_detected"])*100:.1f}%

⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Stats command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause bot scanning."""
        try:
            self.alert_manager.pause()
            await update.message.reply_text(
                "⏸️ <b>Scanning Paused</b>\n\nUse /resume to continue.",
                parse_mode='HTML'
            )
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume bot scanning."""
        try:
            self.alert_manager.resume()
            await update.message.reply_text(
                "▶️ <b>Scanning Resumed</b>\n\nBot is now active.",
                parse_mode='HTML'
            )
        except Exception as e:
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_analyze(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Quick analysis of a coin."""
        try:
            # Get symbol from args
            if not context.args:
                await update.message.reply_text(
                    "Usage: /ana <symbol>\nExample: /ana BTC or /ana DOGE",
                    parse_mode='HTML'
                )
                return
            
            symbol = context.args[0].upper()
            if not symbol.endswith("-USDT"):
                symbol = f"{symbol}-USDT"
            
            await update.message.reply_text(f"🔍 Analyzing <code>{symbol}</code>...", parse_mode='HTML')
            
            # Fetch klines
            klines_15m = await self.rest_client.get_futures_klines(symbol, "15m", 100)
            klines_h1 = await self.rest_client.get_futures_klines(symbol, "1h", 200)
            klines_h4 = await self.rest_client.get_futures_klines(symbol, "4h", 100)
            
            if not all([klines_15m, klines_h1, klines_h4]):
                await update.message.reply_text(f"❌ Could not fetch data for {symbol}")
                return
            
            # Calculate indicators
            indicators = self.indicator_calc.calculate(
                symbol, klines_15m, klines_h1, klines_h4
            )
            
            if not indicators:
                await update.message.reply_text(f"❌ Analysis failed for {symbol}")
                return
            
            # Check BTC mood
            btc_status = "🔴 DUMP" if self.trade_filter.is_btc_dumping() else "🟢 OK"
            
            # MTF trend
            price = indicators.price
            ema89 = indicators.ema89_h1
            above_ema = price > ema89
            mtf_bias = "🟢 BULLISH" if above_ema else "🔴 BEARISH"
            
            # Swing levels
            swing_range = indicators.swing_high_20 - indicators.swing_low_20
            swing_pct = (swing_range / price) * 100
            
            msg = f"""
📊 <b>QUICK ANALYSIS: {symbol}</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>💰 Price:</b> <code>{indicators.price:,.8g}</code>
<b>📊 ATR:</b> <code>{indicators.atr:,.8g}</code>

<b>📈 Indicators:</b>
• RSI H1: {indicators.rsi_h1:.0f}
• MFI: {indicators.mfi:.0f}
• WaveTrend: {indicators.wt_signal}
• MACD: {indicators.macd_trend}
• Volume: {indicators.volume_ratio:.1f}x

<b>🔄 MTF Trend:</b>
• EMA89 H1: <code>{ema89:,.8g}</code>
• Price vs EMA: {'Above ↗️' if above_ema else 'Below ↘️'}
• Bias: {mtf_bias}

<b>📍 Levels:</b>
• Swing High: <code>{indicators.swing_high_20:,.8g}</code>
• Swing Low: <code>{indicators.swing_low_20:,.8g}</code>
• Range: {swing_pct:.1f}%

<b>🌐 Context:</b>
• Trend H1: {indicators.trend_h1}
• BTC Status: {btc_status}

<b>💡 Recommendation:</b>
{"✅ LONG OK" if above_ema and not self.trade_filter.is_btc_dumping() else ""}
{"✅ SHORT OK" if not above_ema else ""}
{"⚠️ BTC dumping - avoid LONG" if self.trade_filter.is_btc_dumping() else ""}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Analyze command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_btc(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Check BTC mood."""
        try:
            # Get BTC data
            klines_15m = await self.rest_client.get_futures_klines("BTC-USDT", "15m", 4)
            klines_h1 = await self.rest_client.get_futures_klines("BTC-USDT", "1h", 48)
            
            if not klines_15m or not klines_h1:
                await update.message.reply_text("❌ Could not fetch BTC data")
                return
            
            # Parse data
            import pandas as pd
            
            df_15m = pd.DataFrame([{
                'time': int(k.get('time', 0)),
                'open': float(k.get('open', 0)),
                'high': float(k.get('high', 0)),
                'low': float(k.get('low', 0)),
                'close': float(k.get('close', 0)),
            } for k in klines_15m])
            df_15m = df_15m.sort_values('time', ascending=True)
            
            df_h1 = pd.DataFrame([{
                'time': int(k.get('time', 0)),
                'close': float(k.get('close', 0)),
            } for k in klines_h1])
            df_h1 = df_h1.sort_values('time', ascending=True)
            
            current_price = df_15m['close'].iloc[-1]
            price_15m_ago = df_15m['close'].iloc[-2] if len(df_15m) >= 2 else current_price
            price_1h_ago = df_h1['close'].iloc[-2] if len(df_h1) >= 2 else current_price
            price_4h_ago = df_h1['close'].iloc[-4] if len(df_h1) >= 4 else current_price
            price_24h_ago = df_h1['close'].iloc[-24] if len(df_h1) >= 24 else current_price
            
            change_15m = ((current_price - price_15m_ago) / price_15m_ago) * 100
            change_1h = ((current_price - price_1h_ago) / price_1h_ago) * 100
            change_4h = ((current_price - price_4h_ago) / price_4h_ago) * 100
            change_24h = ((current_price - price_24h_ago) / price_24h_ago) * 100
            
            # Determine mood
            is_dumping = change_15m < -0.5
            mood_emoji = "🔴" if is_dumping else "🟢" if change_15m > 0.5 else "🟡"
            mood_text = "DUMPING" if is_dumping else "BULLISH" if change_15m > 0.5 else "NEUTRAL"
            
            # Session high/low
            session_high = df_15m['high'].max()
            session_low = df_15m['low'].min()
            
            msg = f"""
₿ <b>BTC MOOD CHECK</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>💰 Price:</b> <code>${current_price:,.2f}</code>

<b>📊 Changes:</b>
• 15m: <code>{change_15m:+.2f}%</code> {mood_emoji}
• 1h: <code>{change_1h:+.2f}%</code>
• 4h: <code>{change_4h:+.2f}%</code>
• 24h: <code>{change_24h:+.2f}%</code>

<b>📍 Session Range:</b>
• High: <code>${session_high:,.2f}</code>
• Low: <code>${session_low:,.2f}</code>

<b>🌡️ MOOD: {mood_text}</b>

<b>💡 Impact on Altcoins:</b>
{"⚠️ Avoid LONG on altcoins when BTC dumps" if is_dumping else "✅ Altcoin LONG signals OK"}
{"✅ SHORT signals are fine" if is_dumping else ""}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"BTC command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_coins(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show top coins by volume."""
        try:
            tickers = await self.rest_client.get_futures_tickers()
            if not tickers:
                await update.message.reply_text("❌ Could not fetch tickers")
                return
            
            # Sort by volume
            sorted_tickers = sorted(
                tickers,
                key=lambda x: float(x.get('quoteVolume', 0)),
                reverse=True
            )[:15]
            
            lines = []
            for i, t in enumerate(sorted_tickers, 1):
                symbol = t.get('symbol', '').replace('-USDT', '')
                price = float(t.get('lastPrice', 0))
                change = float(t.get('priceChangePercent', 0))
                volume = float(t.get('quoteVolume', 0))
                
                emoji = "🟢" if change > 0 else "🔴" if change < 0 else "⚪"
                vol_str = f"${volume/1e6:.1f}M" if volume >= 1e6 else f"${volume/1e3:.0f}K"
                
                lines.append(f"{i}. {symbol}: ${price:,.4g} {emoji} {change:+.1f}% | {vol_str}")
            
            msg = f"""
📊 <b>TOP 15 BY VOLUME</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

""" + "\n".join(lines) + f"""

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"Coins command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show economic calendar."""
        try:
            # Get news manager from alert manager
            news_manager = self.alert_manager.news_manager
            if not news_manager:
                await update.message.reply_text("❌ News Manager not initialized")
                return
            
            # Get upcoming events (next 48 hours)
            events = await news_manager.get_upcoming_events(hours=48)
            
            if not events:
                msg = """
📅 <b>ECONOMIC CALENDAR</b>
━━━━━━━━━━━━━━━━━━━━━━

✅ No high-impact news in next 48 hours

🚀 Trading is safe!

⏰ {now}
""".format(now=datetime.now().strftime('%H:%M:%S'))
                await update.message.reply_text(msg.strip(), parse_mode='HTML')
                return
            
            # Format events
            now = datetime.now(pytz.utc)
            lines = []
            
            def get_sort_date(e):
                """Get timezone-aware date for sorting."""
                if e.date.tzinfo is None:
                    return pytz.utc.localize(e.date)
                return e.date.astimezone(pytz.utc)
            
            # Group by day
            current_day = None
            for event in sorted(events, key=get_sort_date):
                day = event.date.strftime("%A, %b %d")
                if day != current_day:
                    if current_day:
                        lines.append("")
                    lines.append(f"📍 <b>{day}</b>")
                    current_day = day
                
                # Time and details
                time_str = event.date.strftime("%H:%M")
                impact_emoji = "🔥" if event.impact.value == "High" else "⚡"
                
                # Time until
                time_until = event.time_until
                if time_until and time_until.total_seconds() > 0:
                    hours = int(time_until.total_seconds() // 3600)
                    mins = int((time_until.total_seconds() % 3600) // 60)
                    if hours > 0:
                        until_str = f"({hours}h {mins}m)"
                    else:
                        until_str = f"({mins}m)"
                else:
                    until_str = "(passed)"
                
                lines.append(f"  {impact_emoji} {time_str} UTC - <b>{event.title}</b> {until_str}")
            
            # News manager status
            poller_mode = news_manager.poller.mode.value if news_manager.poller else "N/A"
            is_paused = "🛑 PAUSED" if news_manager.is_trading_paused else "🟢 ACTIVE"
            
            msg = f"""
📅 <b>ECONOMIC CALENDAR - USD HIGH IMPACT</b>
━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(lines)}

━━━━━━━━━━━━━━━━━━━━━━
📊 Events tracked: {len(events)}
⚡ Poller Mode: {poller_mode}
🤖 Trading: {is_paused}

⚠️ Bot pauses 5min before 🔥 news
📈 Straddle alert at 3min before

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
            await update.message.reply_text(msg.strip(), parse_mode='HTML')
            
        except Exception as e:
            logger.error(f"News command error: {e}")
            import traceback
            traceback.print_exc()
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_session(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show trading sessions status."""
        try:
            # Get session status from alert manager
            status_msg = await self.alert_manager.get_session_status()
            await update.message.reply_text(status_msg, parse_mode='HTML')
        except Exception as e:
            logger.error(f"Session command error: {e}")
            await update.message.reply_text(f"Error: {e}")
    
    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show help message."""
        msg = """
🤖 <b>BingX Alert Bot v3.5</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>📊 Status & Info:</b>
/status - Bot status & circuit breaker
/stats - Detailed statistics
/btc - BTC mood check (dump/pump)
/coins - Top 15 by volume
/news - Economic calendar (next 48h)
/session - Trading sessions (Asia/EU/US)

<b>🔍 Analysis:</b>
/ana <symbol> - Quick analysis
Example: /ana BTC or /ana DOGE

<b>⚙️ Control:</b>
/pause - Pause scanning
/resume - Resume scanning

<b>ℹ️ Help:</b>
/help - This message

<b>📌 FEATURES:</b>
• Scoring System v2.1 (0-100 points)
• Tier System: 💎 DIAMOND / 🥇 GOLD
• Google Sheets Trade Journal (18 cols)
• Auto Session Notifications
• BTC Correlation Filter
• News Awareness (auto pause)
• Circuit Breaker Protection

<b>🎯 SCORING:</b>
Strategy Points: PUMP_FADE +30, SFP +25, etc.
Confirmation: RSI Div +15, Fib Golden +15
Position Size: Diamond $2, Gold $1

<b>📊 GOOGLE SHEETS:</b>
• Auto-logs all trades
• Real-time PnL updates
• Track Grade, Layers, Checklist

<b>🌐 SESSIONS:</b>
• 🌏 Asia: 00:00 UTC
• 🌍 Europe: 07:00 UTC
• 🌎 US: 13:30 UTC

⏰ Bot v3.5 | Scoring v2.1 | WebSocket Mode
"""
        await update.message.reply_text(msg.strip(), parse_mode='HTML')
