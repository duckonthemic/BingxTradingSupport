"""
IE Trade Runner - Main Entry Point

This module provides the main entry point for running the IE Trade scanner
as part of the main application or standalone.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from .config import IETradeConfig
from .scanner import create_ie_scanner, IEScanner
from .commands import setup_ie_trade_commands
from .bias_manager import BiasScheduler

logger = logging.getLogger(__name__)


class IETradeRunner:
    """
    Main runner for the IE Trade module.
    
    Manages the lifecycle of all IE Trade components:
    - Scanner for detecting setups
    - Bias scheduler for 7AM reminders
    - Telegram command handlers
    """
    
    def __init__(
        self,
        rest_client,
        telegram_bot,
        sheets_client,
        redis_client=None,
        config: Optional[IETradeConfig] = None
    ):
        """
        Initialize the IE Trade runner.
        
        Args:
            rest_client: BingX REST API client
            telegram_bot: Telegram bot instance
            sheets_client: Google Sheets client (optional)
            redis_client: Redis client for state persistence (optional)
            config: Custom configuration (uses default if not provided)
        """
        self.config = config or IETradeConfig()
        self.rest_client = rest_client
        self.telegram_bot = telegram_bot
        self.sheets_client = sheets_client
        self.redis_client = redis_client
        
        # Components (initialized in start())
        self.scanner: Optional[IEScanner] = None
        self.bias_scheduler: Optional[BiasScheduler] = None
        self.command_handler = None
        
        # State
        self._running = False
        self._tasks: list[asyncio.Task] = []
        
    async def start(self):
        """Start the IE Trade module."""
        if self._running:
            logger.warning("IE Trade runner already running")
            return
            
        logger.info("=" * 50)
        logger.info("🚀 Starting IE Trade Module v1.0.0")
        logger.info("=" * 50)
        
        try:
            # Create scanner with Redis for bias persistence
            self.scanner = create_ie_scanner(
                rest_client=self.rest_client,
                telegram_bot=self.telegram_bot,
                sheets_client=self.sheets_client,
                redis_client=self.redis_client,  # For bias persistence
                config=self.config
            )
            
            # Create send callback for scheduler
            async def send_reminder(message: str):
                """Send reminder via Telegram"""
                from src.notification.telegram_bot import TelegramBot
                bot = TelegramBot()
                await bot.send_message(message, parse_mode='Markdown')
            
            # Setup bias scheduler for 7AM reminders
            self.bias_scheduler = BiasScheduler(
                bias_manager=self.scanner.bias_manager,
                send_callback=send_reminder,
                config=self.config
            )
            
            # Setup command handlers
            self.command_handler = setup_ie_trade_commands(
                app=self.telegram_bot,
                scanner=self.scanner,
                bias_manager=self.scanner.bias_manager,
                config=self.config
            )
            # Call async setup
            await self.command_handler.setup(self.telegram_bot)
            
            # Start background tasks
            self._tasks.append(
                asyncio.create_task(
                    self.bias_scheduler.start(),
                    name="ie_bias_scheduler"
                )
            )
            
            # START THE SCANNER (was missing!)
            self._tasks.append(
                asyncio.create_task(
                    self.scanner.start(),
                    name="ie_scanner"
                )
            )
            
            self._running = True
            
            # Log startup info
            logger.info(f"📊 Monitoring {len(self.config.TOP_COINS)} coins")
            logger.info(f"⏰ Kill Zones: London 14:00-17:00, NY 19:00-23:00 (VN)")
            logger.info(f"📈 Premium threshold: {self.config.PREMIUM_THRESHOLD * 100:.1f}%")
            logger.info(f"⏱️ FVG max age: {self.config.FVG_MAX_AGE_HOURS}h")
            logger.info("✅ IE Trade module started successfully")
            logger.info("")
            logger.info("📝 Commands available:")
            logger.info("  /dbias B  - Set LONG bias for today")
            logger.info("  /dbias S  - Set SHORT bias for today")
            logger.info("  /iestatus - Show module status")
            logger.info("  /iestart  - Start scanning")
            logger.info("  /iestop   - Stop scanning")
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"Failed to start IE Trade module: {e}")
            await self.stop()
            raise
            
    async def stop(self):
        """Stop the IE Trade module."""
        logger.info("Stopping IE Trade module...")
        
        self._running = False
        
        # Cancel all background tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self._tasks.clear()
        
        # Stop scanner
        if self.scanner:
            await self.scanner.stop()
            self.scanner = None
            
        logger.info("IE Trade module stopped")
        
    async def get_status(self) -> dict:
        """Get current status of the IE Trade module."""
        status = {
            "running": self._running,
            "module": "IE Trade",
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat()
        }
        
        if self.scanner:
            scanner_status = await self.scanner.get_status()
            status.update(scanner_status)
            
        return status


# Global runner instance
_runner: Optional[IETradeRunner] = None


async def get_ie_trade_runner() -> Optional[IETradeRunner]:
    """Get the global IE Trade runner instance."""
    return _runner


async def start_ie_trade(
    rest_client,
    telegram_bot,
    sheets_client=None,
    redis_client=None,
    config: Optional[IETradeConfig] = None
) -> IETradeRunner:
    """
    Start the IE Trade module.
    
    This is the main entry point for integrating IE Trade into
    the main application.
    
    Args:
        rest_client: BingX REST API client
        telegram_bot: Telegram bot instance  
        sheets_client: Google Sheets client (optional)
        redis_client: Redis client (optional)
        config: Custom configuration
        
    Returns:
        IETradeRunner instance
    """
    global _runner
    
    if _runner and _runner._running:
        logger.warning("IE Trade module already running")
        return _runner
        
    _runner = IETradeRunner(
        rest_client=rest_client,
        telegram_bot=telegram_bot,
        sheets_client=sheets_client,
        redis_client=redis_client,
        config=config
    )
    
    await _runner.start()
    return _runner


async def stop_ie_trade():
    """Stop the IE Trade module."""
    global _runner
    
    if _runner:
        await _runner.stop()
        _runner = None
