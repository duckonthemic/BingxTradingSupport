"""
IE Trade Telegram Commands

Commands:
- /dbias B - Set daily bias to LONG (Buy)
- /dbias S - Set daily bias to SHORT (Sell)
- /dbias - Show current bias status
- /iestatus - Show IE Trade module status
- /iestop - Stop IE Trade scanning
- /iestart - Start IE Trade scanning
"""

import logging
from datetime import datetime
from typing import Optional

from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes

from .config import IETradeConfig, DEFAULT_CONFIG
from .bias_manager import BiasManager, DailyBias, BiasScheduler
from .scanner import IEScanner

logger = logging.getLogger(__name__)


class IETradeCommandHandler:
    """Handles IE Trade Telegram commands."""
    
    def __init__(
        self,
        scanner: IEScanner,
        bias_manager: BiasManager,
        config: IETradeConfig = DEFAULT_CONFIG
    ):
        self.scanner = scanner
        self.bias_manager = bias_manager
        self.config = config
        self._app: Optional[Application] = None
    
    async def setup(self, app: Application):
        """Setup command handlers."""
        self._app = app
        
        # Register IE Trade commands
        app.add_handler(CommandHandler("dbias", self.cmd_dbias))
        app.add_handler(CommandHandler("iestatus", self.cmd_iestatus))
        app.add_handler(CommandHandler("iestop", self.cmd_iestop))
        app.add_handler(CommandHandler("iestart", self.cmd_iestart))
        app.add_handler(CommandHandler("iereset", self.cmd_iereset))
        
        logger.info("🎯 IE Trade commands registered")
    
    def get_commands(self) -> list:
        """Get list of BotCommand for menu."""
        return [
            BotCommand("dbias", "Set daily bias: /dbias B or /dbias S"),
            BotCommand("iestatus", "IE Trade module status"),
            BotCommand("iestart", "Start IE Trade scanning"),
            BotCommand("iestop", "Stop IE Trade scanning"),
            BotCommand("iereset", "Reset IE Trade states"),
        ]
    
    async def cmd_dbias(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Handle /dbias command.
        
        Usage:
        - /dbias B - Set LONG bias
        - /dbias S - Set SHORT bias
        - /dbias - Show current bias
        """
        try:
            args = context.args
            
            if not args:
                # Show current bias status
                msg = self.bias_manager.get_status_message()
                await update.message.reply_text(msg, parse_mode='Markdown')
                return
            
            # Parse bias direction
            bias_str = args[0].upper()
            
            if bias_str in ('B', 'BUY', 'LONG', 'L'):
                bias = DailyBias.LONG
            elif bias_str in ('S', 'SELL', 'SHORT'):
                bias = DailyBias.SHORT
            else:
                await update.message.reply_text(
                    "❌ Invalid bias. Use:\n"
                    "• `/dbias B` for LONG (Bullish)\n"
                    "• `/dbias S` for SHORT (Bearish)",
                    parse_mode='Markdown'
                )
                return
            
            # Set bias
            user = update.effective_user
            username = user.username or user.first_name or str(user.id)
            
            self.bias_manager.set_bias(bias, set_by=username)
            
            # Send confirmation
            msg = self.bias_manager.get_bias_confirmed_message(bias)
            await update.message.reply_text(msg, parse_mode='Markdown')
            
            logger.info(f"🎯 IE Bias set to {bias.value} by {username}")
            
        except Exception as e:
            logger.error(f"Error in /dbias: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_iestatus(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show IE Trade module status."""
        try:
            status = self.scanner.get_status()
            bias_state = self.bias_manager.state
            
            # Kill zone info
            kz_info = status['kill_zone']
            kz_emoji = "🟢" if "London" in kz_info or "York" in kz_info else "⚪"
            
            # Running status
            running_emoji = "🟢" if status['running'] else "🔴"
            
            # Bias status
            if bias_state.is_active:
                bias_emoji = "🟢" if bias_state.bias == DailyBias.LONG else "🔴"
                bias_text = f"{bias_emoji} {bias_state.bias.value}"
            else:
                bias_text = "⚪ Not set"
            
            # Phase breakdown
            phases = status['phases']
            
            msg = f"""
🎯 **IE Trade Status**
━━━━━━━━━━━━━━━━━━━━━━━━

{running_emoji} **Scanner:** {'Running' if status['running'] else 'Stopped'}
📊 **Scans:** {status['scan_count']}

**Daily Bias:** {bias_text}
⏱️ **Expires:** {status['bias_expires_in']}

{kz_emoji} **Kill Zone:** {kz_info}
🔢 **Coins Monitored:** {status['coins_monitored']}

**Scanning Phases:**
• Waiting for FVG: {phases.get('SCANNING_FVG', 0)}
• Monitoring FVG: {phases.get('MONITORING_FVG', 0)}
• In FVG Zone: {phases.get('IN_FVG_ZONE', 0)}
• MSS Detected: {phases.get('MSS_DETECTED', 0)}
• Setup Ready: {phases.get('SETUP_READY', 0)}
• Alerted: {phases.get('ALERTED', 0)}

📈 **Active Positions:** {status['active_positions']}/{self.config.MAX_OPEN_POSITIONS}
⏳ **Pending Setups:** {status['pending_setups']}

━━━━━━━━━━━━━━━━━━━━━━━━
💡 Commands:
• `/dbias B` - Set LONG bias
• `/dbias S` - Set SHORT bias
• `/iestart` - Start scanning
• `/iestop` - Stop scanning
"""
            await update.message.reply_text(msg, parse_mode='Markdown')
            
        except Exception as e:
            logger.error(f"Error in /iestatus: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_iestart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Start IE Trade scanning."""
        try:
            if not self.bias_manager.is_bias_set:
                await update.message.reply_text(
                    "⚠️ **Cannot start - No bias set!**\n\n"
                    "First set your daily bias:\n"
                    "• `/dbias B` for LONG\n"
                    "• `/dbias S` for SHORT",
                    parse_mode='Markdown'
                )
                return
            
            await self.scanner.start()
            
            bias = self.bias_manager.current_bias.value
            await update.message.reply_text(
                f"✅ **IE Trade Started!**\n\n"
                f"📊 Bias: {bias}\n"
                f"🔍 Scanning {len(self.config.TOP_COINS)} coins\n"
                f"⏰ Kill Zones: London 14-17h, NY 19-23h\n\n"
                f"_Alerts will be sent when setups are found_",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in /iestart: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_iestop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Stop IE Trade scanning."""
        try:
            await self.scanner.stop()
            
            await update.message.reply_text(
                "⏹️ **IE Trade Stopped**\n\n"
                "_Use `/iestart` to resume scanning_",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in /iestop: {e}")
            await update.message.reply_text(f"❌ Error: {e}")
    
    async def cmd_iereset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Reset all IE Trade states."""
        try:
            self.scanner.reset_all_states()
            self.bias_manager.clear_bias()
            
            await update.message.reply_text(
                "🔄 **IE Trade Reset**\n\n"
                "• All coin states cleared\n"
                "• Daily bias cleared\n"
                "• Pending setups cleared\n\n"
                "_Use `/dbias` to set new bias_",
                parse_mode='Markdown'
            )
            
        except Exception as e:
            logger.error(f"Error in /iereset: {e}")
            await update.message.reply_text(f"❌ Error: {e}")


def setup_ie_trade_commands(
    app: Application,
    scanner: IEScanner,
    bias_manager: BiasManager,
    config: IETradeConfig = DEFAULT_CONFIG
) -> IETradeCommandHandler:
    """
    Factory function to setup IE Trade commands.
    
    Args:
        app: Telegram Application
        scanner: IE Scanner instance
        bias_manager: Bias manager instance
        config: IE Trade config
        
    Returns:
        Configured command handler
    """
    handler = IETradeCommandHandler(scanner, bias_manager, config)
    # Note: setup() is async, needs to be called separately
    return handler
