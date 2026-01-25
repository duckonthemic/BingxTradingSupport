"""
Main entry point for BingX Alert Bot v3.5

SCORING SYSTEM v2.0:
- Confidence Matrix (0-100 points)
- Pump Fade / Shooting Star detection
- Kill Shot combo: SFP + Shooting Star = 10/10 confidence
- Tier System: Diamond (>=80), Gold (>=60)
- Rate Limiter: 10 alerts/hour with sliding window
- Dynamic Stoploss for counter-trend trades
- TOP 5 coins by confidence score per cycle
- 15-minute scan interval

NEW IN v3.5:
- RealTimeSignalEngine for instant signal detection
- WebSocket-based price streaming
- Event-driven instead of polling

NEW IN v3.6:
- IE Trade Module: Independent ICT 4-step entry strategy
- Daily Bias confirmation before scanning
- H1 FVG + M5 MSS detection
- Kill Zone filtering (London/NY sessions)
"""

import asyncio
import logging
import signal
import os
from datetime import datetime

from src.config import config
from src.notification.alert_manager import AlertManager
from src.notification.command_handler import TelegramCommandHandler
from src.ingestion.rest_client import BingXRestClient
from src.analysis.trade_filter import TradeFilter
from src.analysis.realtime_engine import RealTimeSignalEngine, SignalTrigger

from telegram import Update
from telegram.ext import Application, CommandHandler

# IE Trade module
from src.ie_trade import start_ie_trade, stop_ie_trade

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/bot.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)


class Bot:
    """Main bot class that integrates all components."""
    
    def __init__(self, use_realtime: bool = False, enable_ie_trade: bool = True):
        """
        Initialize bot.
        
        Args:
            use_realtime: If True, use WebSocket RealTimeSignalEngine instead of polling
            enable_ie_trade: If True, enable the IE Trade module for ICT entry strategy
        """
        self.alert_manager = AlertManager()
        self.telegram_app: Application = None
        self.command_handler: TelegramCommandHandler = None
        self._running = False
        
        # Real-time mode
        self.use_realtime = use_realtime
        self.realtime_engine: RealTimeSignalEngine = None
        
        # IE Trade module
        self.enable_ie_trade = enable_ie_trade
        self.ie_trade_runner = None
        
    async def _on_signal_trigger(self, trigger: SignalTrigger):
        """
        Callback when RealTimeSignalEngine detects a trigger.
        Run full analysis only on triggered coin.
        """
        logger.info(f"⚡ REALTIME TRIGGER: {trigger.symbol} | {trigger.trigger_type} | "
                   f"vol_ratio={trigger.volume_ratio:.1f}x | price=${trigger.price:.4f}")
        
        # Run full analysis on just this coin
        try:
            await self.alert_manager.scan_single_coin(trigger.symbol)
        except Exception as e:
            logger.error(f"Error processing triggered coin {trigger.symbol}: {e}")
    
    async def start(self):
        """Start the bot."""
        logger.info("="*50)
        mode = "REALTIME (WebSocket)" if self.use_realtime else "POLLING (90s)"
        logger.info(f"🤖 BingX Alert Bot v3.5 - Mode: {mode}")
        logger.info("="*50)
        
        # Initialize Telegram bot with commands
        self.telegram_app = Application.builder().token(
            config.telegram.bot_token
        ).build()
        
        # Setup command handler with alert manager reference
        self.command_handler = TelegramCommandHandler(
            alert_manager=self.alert_manager,
            rest_client=self.alert_manager.rest_client,
            trade_filter=self.alert_manager.trade_filter
        )
        await self.command_handler.setup(self.telegram_app)
        
        # Start Telegram polling in background
        await self.telegram_app.initialize()
        await self.telegram_app.start()
        
        # Start updater
        asyncio.create_task(
            self.telegram_app.updater.start_polling(
                allowed_updates=Update.ALL_TYPES,
                drop_pending_updates=True
            )
        )
        
        logger.info("✅ Telegram commands initialized")
        
        # Start IE Trade module if enabled
        if self.enable_ie_trade:
            try:
                logger.info("🎯 Starting IE Trade Module...")
                self.ie_trade_runner = await start_ie_trade(
                    rest_client=self.alert_manager.rest_client,
                    telegram_bot=self.telegram_app,
                    sheets_client=getattr(self.alert_manager, 'sheets_client', None),
                    redis_client=getattr(self.alert_manager, 'redis', None)  # For bias persistence
                )
                logger.info("✅ IE Trade module initialized")
            except Exception as e:
                logger.error(f"Failed to start IE Trade module: {e}")
                logger.warning("Continuing without IE Trade module...")
        
        # Start alert manager
        self._running = True
        
        if self.use_realtime:
            # Initialize real-time engine with trigger callback
            self.realtime_engine = RealTimeSignalEngine(
                on_signal_trigger=self._on_signal_trigger
            )
            
            # Start WebSocket streaming + polling as fallback
            logger.info("🌐 Starting RealTimeSignalEngine...")
            await asyncio.gather(
                self.alert_manager.start(),  # Background polling fallback
                self._run_realtime_loop()    # Real-time trigger detection
            )
        else:
            # Traditional polling mode
            await self.alert_manager.start()
    
    async def _run_realtime_loop(self):
        """
        Run real-time signal detection loop.
        Subscribes to WebSocket and processes price ticks.
        """
        try:
            # Get filtered coins from prefilter
            from src.ingestion.prefilter import PreFilter
            prefilter = PreFilter()
            
            # Subscribe to top coins
            async with BingXRestClient() as client:
                tickers = await client.get_futures_tickers()
                
            if tickers:
                filtered = prefilter.filter_batch(tickers)[:50]  # Top 50 coins
                symbols = [c.symbol for c in filtered]  # FilteredTicker objects
                logger.info(f"🔗 Subscribing to {len(symbols)} coins for real-time...")
                
                # Connect to WebSocket and stream prices
                from src.ingestion.futures_websocket import FuturesWebSocketClient
                
                async with FuturesWebSocketClient() as ws:
                    await ws.subscribe_symbols(symbols)
                    
                    async for msg in ws.receive():
                        if not self._running:
                            break
                            
                        # Process tick through real-time engine
                        if msg and 'data' in msg:
                            data = msg['data']
                            if isinstance(data, dict):
                                tick = {
                                    'symbol': data.get('s', ''),
                                    'price': float(data.get('c', 0)),
                                    'volume': float(data.get('v', 0)),
                                    'timestamp': datetime.now()
                                }
                                await self.realtime_engine.process_tick(tick)
                            
        except Exception as e:
            logger.error(f"Real-time loop error: {e}. Falling back to polling.")
            import traceback
            traceback.print_exc()
            # Fallback is already running via alert_manager.start()
    
    async def stop(self):
        """Stop the bot."""
        logger.info("Stopping bot...")
        self._running = False
        
        # Stop IE Trade module
        if self.ie_trade_runner:
            try:
                await stop_ie_trade()
                logger.info("IE Trade module stopped")
            except Exception as e:
                logger.error(f"Error stopping IE Trade: {e}")
        
        if self.telegram_app:
            await self.telegram_app.updater.stop()
            await self.telegram_app.stop()
            await self.telegram_app.shutdown()
        
        await self.alert_manager.stop()
        logger.info("Bot stopped")


async def main():
    """Main entry point."""
    # Check for real-time mode via env var
    use_realtime = os.getenv("REALTIME_MODE", "false").lower() == "true"
    
    # Check for IE Trade module via env var (enabled by default)
    enable_ie_trade = os.getenv("IE_TRADE_ENABLED", "true").lower() == "true"
    
    bot = Bot(use_realtime=use_realtime, enable_ie_trade=enable_ie_trade)
    
    # Handle signals
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(bot.stop())
    
    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, signal_handler)
    except NotImplementedError:
        # Windows doesn't support add_signal_handler
        pass
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        await bot.stop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        await bot.stop()
        raise


if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          BingX Alert Bot v3.6 - SCORING SYSTEM v2.0          ║
    ╠══════════════════════════════════════════════════════════════╣
    ║  📋 THE CHECKLIST (Score >= 55 = Send Signal)                ║
    ║  ────────────────────────────────────────────────────────────║
    ║  ✅ Nhóm 1: Context                                          ║
    ║     • EMA Trend (H1): Price vs EMA34/89                      ║
    ║     • Market Structure: HH/HL or LL/LH                       ║
    ║                                                              ║
    ║  ✅ Nhóm 2: Trigger (Bắt buộc >= 1)                          ║
    ║     • SFP/Sweep: Quét râu rồi rút chân                       ║
    ║     • Retest Zone: OB hoặc Fib 0.5-0.618                     ║
    ║                                                              ║
    ║  ✅ Nhóm 3: Momentum                                         ║
    ║     • RSI/WaveTrend + Volume Spike                           ║
    ║  ────────────────────────────────────────────────────────────║
    ║  💎 DIAMOND (>=75) = Full confidence                         ║
    ║  🥇 GOLD (>=55) = Strong signal                              ║
    ║  ────────────────────────────────────────────────────────────║
    ║  🎯 IE TRADE MODULE (ICT 4-Step Entry):                      ║
    ║     Step 1: Daily Bias (/dbias B or /dbias S)                ║
    ║     Step 2: H1 FVG in Premium/Discount zone                  ║
    ║     Step 3: M5 MSS confirmation                              ║
    ║     Step 4: M5 FVG entry with SL/TP                          ║
    ║     Commands: /iestatus, /iestart, /iestop                   ║
    ║  ────────────────────────────────────────────────────────────║
    ║  🌐 MODES:                                                   ║
    ║     REALTIME_MODE=true   → WebSocket                         ║
    ║     REALTIME_MODE=false  → 90s polling (default)             ║
    ║     IE_TRADE_ENABLED=true → ICT module ON (default)          ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    asyncio.run(main())
