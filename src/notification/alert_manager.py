"""
Alert Manager v3.4 - Scoring v2.0 + Risk Management + Trade Journal

KEY FEATURES:
- Confidence Matrix Scoring (0-100 points)
- Pump Fade / Shooting Star detection
- Kill Shot combo (SFP + Shooting Star = 10/10 confidence)
- Tier System: Diamond (>=75), Gold (>=55)
- Rate Limiter: 10 alerts/hour with sliding window
- Dynamic Stoploss for counter-trend trades
- BTC Correlation Filter (block LONG when BTC dump)
- Fixed Position Size ($2) with Dynamic Leverage
- Circuit Breaker (daily/weekly max loss)
- Signal Invalidation Tracker
- Google Sheets Trade Journal
- Auto TP/SL tracking with Telegram reply
"""

import asyncio
import logging
import uuid
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import pandas as pd

from ..config import config
from ..storage.redis_client import RedisClient
from ..storage.sheets_client import GoogleSheetsClient, TradeRecord, get_sheets_client
from ..ingestion.rest_client import BingXRestClient
from ..ingestion.futures_websocket import FuturesWebSocketClient, CircuitBreaker as WSCircuitBreaker, RetryHandler
from ..ingestion.prefilter import PreFilter, FilteredTicker
from ..analysis.indicators import IndicatorCalculator, CoinIndicators
from ..analysis.strategy_detector import StrategyDetector, TradeSetup, StrategyType
from ..analysis.trade_filter import TradeFilter, TradeDirection, FilterResult, OptimizedLevels
from ..analysis.scoring_system import (
    ScoringSystem, ChecklistScore, FourLayerResult, SignalGrade,
    ConfidenceScore, SignalTier, THRESHOLD_DIAMOND, THRESHOLD_GOLD
)
from ..context.context_manager import ContextManager
from ..notification.telegram_bot import TelegramNotifier
from ..notification.formatter import AlertFormatter
from ..notification.chart_generator import ChartGenerator
from ..notification.session_scheduler import SessionScheduler
from ..risk.risk_manager import RiskManager, BTCCorrelationResult
from ..news import get_news_manager, NewsManager
from ..tracking import TradeTracker, ActiveTrade, get_trade_tracker

logger = logging.getLogger(__name__)


@dataclass
class ScanResult:
    """Result of scanning a single coin."""
    symbol: str
    success: bool
    indicators: Optional[CoinIndicators] = None
    setup: Optional[TradeSetup] = None
    filter_result: Optional[FilterResult] = None
    filter_reason: Optional[str] = None
    optimized_levels: Optional[OptimizedLevels] = None
    error: Optional[str] = None
    
    # Scoring System v2.0
    checklist_score: Optional[ChecklistScore] = None
    four_layer_result: Optional[FourLayerResult] = None
    signal_grade: SignalGrade = SignalGrade.D_REJECT
    grade_reasons: List[str] = None
    
    # NEW v2.0 fields
    confidence_score: Optional[ConfidenceScore] = None
    signal_tier: SignalTier = SignalTier.REJECT
    confidence_points: int = 0
    is_pump_fade: bool = False
    is_kill_shot: bool = False
    dynamic_sl_price: float = 0.0
    
    def __post_init__(self):
        if self.grade_reasons is None:
            self.grade_reasons = []
    
    @property
    def is_golden(self) -> bool:
        """True if this is a golden setup (SFP + RSI Divergence) or Kill Shot."""
        if self.is_kill_shot:
            return True
        return self.setup and self.setup.is_golden_setup if self.setup else False
    
    @property
    def confluence_score(self) -> int:
        """Get confluence score."""
        return self.setup.confluence_score if self.setup else 0
    
    @property
    def is_tradeable(self) -> bool:
        """
        Check if signal is tradeable based on tier.
        
        Basic tier check only - Adaptive SHORT filter is applied in AlertManager.
        - LONG: Diamond or Gold tier allowed
        - SHORT: Diamond or Gold allowed here, AlertManager applies regime filter
        """
        return self.signal_tier in [SignalTier.DIAMOND, SignalTier.GOLD]
    
    @property
    def volume_weight(self) -> float:
        """Get volume weight based on tier."""
        if self.signal_tier == SignalTier.DIAMOND:
            return 1.0
        elif self.signal_tier == SignalTier.GOLD:
            return 0.7
        return 0.0


class AlertManager:
    """
    High Win Rate Alert Manager v3.3.
    
    Features:
    - Confidence Matrix scoring (0-100 points)
    - Pump Fade / Kill Shot detection
    - Tier System: Diamond (>=75), Gold (>=55)
    - Rate Limiter: 10 alerts/hour with sliding window
    - BTC Correlation Filter (block LONG when BTC dump)
    - Fixed Position Size with Dynamic Leverage
    - Circuit Breaker (daily/weekly max loss)
    - Signal Invalidation Tracker
    - Google Sheets Trade Journal
    - Auto TP/SL tracking with Telegram reply
    """
    
    def __init__(self):
        # Core components
        self.redis = RedisClient()
        self.rest_client = BingXRestClient()
        self.prefilter = PreFilter()
        self.indicator_calc = IndicatorCalculator()
        self.strategy_detector = StrategyDetector()
        self.trade_filter = TradeFilter()
        self.chart_generator = ChartGenerator()
        self.scoring_system = ScoringSystem()  # Scoring v2.0
        self.context_manager: Optional[ContextManager] = None
        self.notifier: Optional[TelegramNotifier] = None
        
        # Trade Journal & Tracking
        self.sheets_client = get_sheets_client()
        self.trade_tracker = get_trade_tracker()
        
        # Risk Management Suite
        self.risk_manager = RiskManager(
            account_balance=config.risk.account_balance,
            risk_per_trade_pct=config.risk.risk_per_trade_pct
        )
        
        # WebSocket for BTC monitoring
        self.ws_client: Optional[FuturesWebSocketClient] = None
        
        # Circuit breaker for WS
        self.ws_circuit_breaker = WSCircuitBreaker()
        self.retry_handler = RetryHandler(max_retries=3)
        
        # Config - SCALP MODE (faster scanning)
        self.scan_interval = config.timing.scan_interval  # 90s for scalp
        self.min_confidence = 50  # Lower for more signals (was 60)
        self.min_rr = 1.5  # 1.5:1 R:R for scalping (was 2.0)
        self.max_concurrent_scans = 30
        self.max_coins_per_cycle = 8  # TOP 8 (was 5)
        
        # State
        self._running = False
        self._paused = False
        self._paused_for_news = False  # Track news-related pause separately
        self._btc_15m_prices: List[Tuple[datetime, float]] = []
        
        # News Manager
        self.news_manager: Optional[NewsManager] = None
        
        # Session Scheduler
        self.session_scheduler: Optional[SessionScheduler] = None
        
        # Stats
        self._stats = {
            "scans": 0,
            "coins_scanned": 0,
            "setups_detected": 0,
            "diamond_setups": 0,
            "gold_setups": 0,
            "kill_shots": 0,
            "pump_fades": 0,
            "filtered_by_btc": 0,
            "filtered_by_btc_correlation": 0,
            "filtered_by_trend": 0,
            "filtered_by_score": 0,
            "filtered_by_rate_limit": 0,
            "filtered_by_circuit_breaker": 0,
            "filtered_by_news": 0,  # New: filtered by news events
            "alerts_sent": 0,
            "alerts_invalidated": 0,
            "strategies": defaultdict(int),
        }
    
    async def start(self):
        """Start the alert manager."""
        logger.info("="*50)
        logger.info("🚀 Starting Alert Manager v3.4 - Trade Journal Mode")
        logger.info("="*50)
        
        # Initialize
        await self.redis.connect()
        await self.rest_client.connect()
        
        self.context_manager = ContextManager(self.redis)
        self.notifier = TelegramNotifier(self.redis)
        await self.notifier.connect()
        
        asyncio.create_task(self.context_manager.start())
        
        # Initialize Google Sheets
        sheets_connected = await self.sheets_client.connect()
        if sheets_connected:
            logger.info("📊 Google Sheets Trade Journal connected")
        else:
            logger.warning("⚠️ Google Sheets not available - trades won't be logged")
        
        # Setup trade tracker callbacks
        self.trade_tracker.on_tp_hit = self._on_tp_hit
        self.trade_tracker.on_sl_hit = self._on_sl_hit
        
        # WebSocket
        self.ws_client = FuturesWebSocketClient(on_btc_update=self._on_btc_ws_update)
        asyncio.create_task(self._run_ws_client())
        
        # Start invalidation checker
        asyncio.create_task(self._run_invalidation_checker())
        
        # Start trade price monitor
        asyncio.create_task(self._run_trade_monitor())
        
        # Start Google Sheets trade monitor
        asyncio.create_task(self._run_sheets_monitor())
        
        # Initialize News Manager
        await self._init_news_manager()
        
        # Initialize Session Scheduler (auto Asia/EU/US notifications)
        await self._init_session_scheduler()
        
        await self._send_startup_message()
        
        self._running = True
        interval_text = f"{self.scan_interval//60} min" if self.scan_interval >= 60 else f"{self.scan_interval}s"
        logger.info(f"✅ Alert Manager v3.4 started - scan every {interval_text}")
        
        while self._running:
            if not self._paused and not self._paused_for_news:
                # Check circuit breaker first
                if not self.risk_manager.circuit_breaker.is_trading_allowed():
                    cb_status = self.risk_manager.circuit_breaker.get_status()
                    logger.warning(f"🔴 Circuit Breaker OPEN: {cb_status.reason}")
                    self._stats["filtered_by_circuit_breaker"] += 1
                # Check news pause
                elif self.news_manager and self.news_manager.is_trading_paused:
                    logger.info("📰 Trading paused for news event")
                    self._stats["filtered_by_news"] += 1
                else:
                    try:
                        await self._run_scan_cycle()
                    except Exception as e:
                        logger.error(f"❌ Scan error: {e}")
                        import traceback
                        traceback.print_exc()
            
            await asyncio.sleep(self.scan_interval)
    
    async def _init_news_manager(self):
        """Initialize and configure the News Manager."""
        try:
            self.news_manager = get_news_manager()
            
            # Configure callbacks
            self.news_manager.configure(
                send_message=self._send_news_message,
                send_pinned_message=self._send_pinned_news_message,
                pause_trading=self._pause_for_news,
                resume_trading=self._resume_from_news
            )
            
            # Start news manager
            await self.news_manager.start()
            logger.info("📰 News Manager initialized and started")
            
            # Send startup news summary
            await self._send_startup_news_summary()
        except Exception as e:
            logger.error(f"Failed to initialize News Manager: {e}")
            self.news_manager = None
    
    async def _init_session_scheduler(self):
        """Initialize and start the Session Scheduler."""
        try:
            async def send_session_msg(msg: str) -> bool:
                if self.notifier:
                    return await self.notifier._send_message(msg)
                return False
            
            self.session_scheduler = SessionScheduler(send_message_fn=send_session_msg)
            await self.session_scheduler.start()
            logger.info("⏰ Session Scheduler started")
        except Exception as e:
            logger.error(f"Failed to initialize Session Scheduler: {e}")
            self.session_scheduler = None
    
    async def get_session_status(self) -> str:
        """Get current trading session status message."""
        if self.session_scheduler:
            return await self.session_scheduler.send_manual_session_status()
        return "Session scheduler not initialized"
    
    async def _send_news_message(self, message: str):
        """Send a news-related message via Telegram."""
        if self.notifier:
            await self.notifier.send_message(message)
    
    async def _send_pinned_news_message(self, message: str):
        """Send and pin a news-related message."""
        if self.notifier:
            # Send and attempt to pin
            try:
                msg = await self.notifier.send_message(message)
                if msg and hasattr(msg, 'message_id'):
                    await self.notifier.pin_message(msg.message_id)
            except Exception as e:
                logger.error(f"Failed to pin message: {e}")
                await self.notifier.send_message(message)
    
    async def _pause_for_news(self):
        """Pause trading due to news event."""
        self._paused_for_news = True
        logger.info("🛑 Trading PAUSED for news event")
    
    async def _resume_from_news(self):
        """Resume trading after news event."""
        self._paused_for_news = False
        logger.info("✅ Trading RESUMED after news event")
    
    async def _send_startup_news_summary(self):
        """Send news summary on startup."""
        try:
            if not self.news_manager:
                return
            
            # Get upcoming events
            events = await self.news_manager.get_upcoming_events(hours=24)
            
            if not events:
                return  # No news to report
            
            import pytz
            from datetime import datetime
            
            now = datetime.now(pytz.utc)
            lines = []
            
            def get_sort_date(e):
                """Get timezone-aware date for sorting."""
                if e.date.tzinfo is None:
                    return pytz.utc.localize(e.date)
                return e.date.astimezone(pytz.utc)
            
            for event in sorted(events, key=get_sort_date)[:5]:  # Top 5
                time_str = event.date.strftime("%H:%M")
                impact_emoji = "🔥" if event.impact.value == "High" else "⚡"
                
                time_until = event.time_until
                if time_until and time_until.total_seconds() > 0:
                    hours = int(time_until.total_seconds() // 3600)
                    mins = int((time_until.total_seconds() % 3600) // 60)
                    if hours > 0:
                        until_str = f"in {hours}h {mins}m"
                    else:
                        until_str = f"in {mins}m"
                else:
                    until_str = "passed"
                
                lines.append(f"{impact_emoji} {time_str} - {event.title} ({until_str})")
            
            msg = f"""
📰 <b>TODAY'S HIGH-IMPACT NEWS</b>
━━━━━━━━━━━━━━━━━━━━━━

{chr(10).join(lines)}

⚠️ Bot will PAUSE 5min before 🔥 events
📈 Use /news for full calendar
"""
            await self.notifier.send_message(msg.strip())
            logger.info(f"📰 Sent startup news summary ({len(events)} events)")
        except Exception as e:
            logger.error(f"Failed to send startup news: {e}")
    
    async def _run_invalidation_checker(self):
        """Background task to check signal invalidations."""
        while self._running:
            try:
                # Get all active tracked symbols
                active_signals = list(self.risk_manager.invalidation_tracker.active_signals.keys())
                
                for signal_id in active_signals:
                    signal = self.risk_manager.invalidation_tracker.active_signals.get(signal_id)
                    if not signal or signal.is_invalidated:
                        continue
                    
                    # Get current price
                    ticker = await self.rest_client.get_ticker(signal.symbol)
                    if not ticker:
                        continue
                    
                    current_price = float(ticker.get("lastPrice", 0))
                    
                    # Check invalidation
                    invalidated = self.risk_manager.check_invalidations(signal.symbol, current_price)
                    
                    for sig_id, sig in invalidated:
                        # Send cancel message
                        await self._send_cancel_alert(sig)
                        self._stats["alerts_invalidated"] += 1
                        logger.info(f"❌ Signal INVALIDATED: {sig.symbol} - {sig.invalidation_reason}")
                
                # Cleanup expired signals
                self.risk_manager.invalidation_tracker.cleanup_expired()
                
            except Exception as e:
                logger.error(f"Invalidation checker error: {e}")
            
            await asyncio.sleep(30)  # Check every 30 seconds
    
    async def _send_cancel_alert(self, signal):
        """Send cancel alert when signal is invalidated."""
        try:
            msg = f"""
❌ <b>ALERT CANCELLED</b>

🪙 <b>{signal.symbol}</b>
📍 Direction: {signal.direction}

<b>Reason:</b> {signal.invalidation_reason}

⚠️ <i>Setup đã không còn hợp lệ, KHÔNG vào lệnh</i>

⏰ <i>{datetime.now().strftime('%H:%M:%S')}</i>
"""
            await self.notifier.send_message(msg.strip())
        except Exception as e:
            logger.error(f"Failed to send cancel alert: {e}")
    
    async def _run_trade_monitor(self):
        """Background task to monitor active trades for TP/SL."""
        while self._running:
            try:
                # Get all symbols being tracked
                summary = self.trade_tracker.get_summary()
                if summary["active_count"] == 0:
                    await asyncio.sleep(10)
                    continue
                
                # Fetch current prices for tracked symbols
                prices = {}
                for symbol in summary["symbols"]:
                    try:
                        ticker = await self.rest_client.get_ticker(symbol)
                        if ticker:
                            prices[symbol] = float(ticker.get("lastPrice", 0))
                    except Exception:
                        pass
                
                # Update trade tracker with current prices
                await self.trade_tracker.update_prices(prices)
                
            except Exception as e:
                logger.error(f"Trade monitor error: {e}")
            
            await asyncio.sleep(5)  # Check every 5 seconds
    
    async def _on_tp_hit(self, trade: ActiveTrade, tp_level: str, pnl: float):
        """Callback when Take Profit is hit."""
        try:
            # Update Google Sheets
            note = f"{tp_level} hit"
            await self.sheets_client.update_trade_result(
                row=trade.sheet_row,
                pnl_percent=pnl,
                note=note
            )
            
            # Send Telegram notification (reply to original message)
            emoji = "🎯" if pnl > 50 else "✅"
            msg = f"""
{emoji} <b>{tp_level} HIT!</b>

🪙 <b>{trade.symbol.replace('-USDT', '')}</b> {trade.direction}
💰 PnL: <b>{pnl:+.2f}%</b>

📊 Entry: {trade.entry_price:,.8g}
🎯 Exit: {tp_level}

<i>Trade updated in journal ✅</i>
⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
"""
            # Reply to original message if we have message_id
            if trade.message_id:
                await self.notifier.reply_to_message(trade.message_id, msg.strip())
            else:
                await self.notifier.send_message(msg.strip())
            
        except Exception as e:
            logger.error(f"TP hit callback error: {e}")
    
    async def _on_sl_hit(self, trade: ActiveTrade, pnl: float):
        """Callback when Stop Loss is hit."""
        try:
            # Update Google Sheets
            await self.sheets_client.update_trade_result(
                row=trade.sheet_row,
                pnl_percent=pnl,
                note="Stoploss hit"
            )
            
            # Send Telegram notification (reply to original message)
            msg = f"""
💀 <b>STOPLOSS HIT</b>

🪙 <b>{trade.symbol.replace('-USDT', '')}</b> {trade.direction}
📉 PnL: <b>{pnl:.2f}%</b>

📊 Entry: {trade.entry_price:,.8g}
💀 SL: {trade.stop_loss:,.8g}

<i>Trade updated in journal ✅</i>
⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}
"""
            # Reply to original message if we have message_id
            if trade.message_id:
                await self.notifier.reply_to_message(trade.message_id, msg.strip())
            else:
                await self.notifier.send_message(msg.strip())
            
            # Record for circuit breaker
            self.risk_manager.record_trade_result(pnl)
            
        except Exception as e:
            logger.error(f"SL hit callback error: {e}")
    
    async def _run_sheets_monitor(self):
        """Background task to monitor Google Sheets trades and update status."""
        logger.info("📊 Google Sheets monitor started")
        
        # Wait for initial connection
        await asyncio.sleep(30)
        
        while self._running:
            try:
                if self.sheets_client._connected:
                    # Check and update all open trades
                    updated = await self.sheets_client.check_and_update_trades(self.rest_client)
                    if updated > 0:
                        logger.info(f"📊 Updated {updated} trades in Google Sheets")
                        
                        # Check for auto-closed trades and send Telegram notification
                        closed_trades = self.sheets_client.get_closed_trades()
                        for trade in closed_trades:
                            await self._send_trade_close_notification(trade)
                        
                        # Get and log stats
                        stats = await self.sheets_client.get_stats()
                        logger.info(
                            f"📈 Stats: {stats['total']} trades, "
                            f"{stats['wins']} wins, {stats['losses']} losses, "
                            f"Winrate: {stats['winrate']:.1f}%"
                        )
                
            except Exception as e:
                logger.error(f"Sheets monitor error: {e}")
            
            # Check every 5 minutes (reduced from 2 min to avoid quota)
            await asyncio.sleep(300)
        
        while self._running:
            if not self._paused:
                try:
                    await self._run_scan_cycle()
                except Exception as e:
                    logger.error(f"❌ Scan error: {e}")
                    import traceback
                    traceback.print_exc()
            
            await asyncio.sleep(self.scan_interval)
    
    async def _run_ws_client(self):
        """Run WebSocket in background."""
        try:
            await self.ws_client.connect()
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
    
    async def _on_btc_ws_update(self, ticker: dict):
        """Handle BTC update."""
        try:
            price = float(ticker.get("c", 0))
            now = datetime.now()
            
            self._btc_15m_prices.append((now, price))
            cutoff = now - timedelta(minutes=15)
            self._btc_15m_prices = [(t, p) for t, p in self._btc_15m_prices if t > cutoff]
            
            if len(self._btc_15m_prices) >= 2:
                oldest = self._btc_15m_prices[0][1]
                self.trade_filter.update_btc_state(price, oldest)
        except:
            pass
    
    async def _run_scan_cycle(self):
        """Run scan cycle - TOP 5 by R:R."""
        start_time = datetime.now()
        self._stats["scans"] += 1
        
        # Update BTC
        await self._update_btc_from_rest()
        
        # Check BTC mood first
        if self.trade_filter.is_btc_dumping():
            logger.info("🔴 BTC dumping - skipping LONG signals this cycle")
        
        # Fetch tickers
        tickers = await self._fetch_with_retry(self.rest_client.get_futures_tickers)
        if not tickers:
            return
        
        # Filter by volume
        filtered = self.prefilter.filter_batch(tickers)
        logger.info(f"📊 Scanning {len(filtered[:50])} coins...")
        
        # Scan concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent_scans)
        
        async def scan_with_limit(coin: FilteredTicker) -> ScanResult:
            async with semaphore:
                return await self._scan_coin(coin)
        
        tasks = [scan_with_limit(coin) for coin in filtered[:50]]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect valid setups
        valid_setups: List[ScanResult] = []
        
        for result in results:
            if isinstance(result, Exception):
                continue
            if not isinstance(result, ScanResult) or not result.success:
                continue
            
            self._stats["coins_scanned"] += 1
            
            if result.filter_result == FilterResult.PASS and result.setup and result.optimized_levels:
                valid_setups.append(result)
                self._stats["setups_detected"] += 1
                self._stats["strategies"][result.setup.strategy.value] += 1
                
                # Track tiers
                if result.signal_tier == SignalTier.DIAMOND:
                    self._stats["diamond_setups"] += 1
                elif result.signal_tier == SignalTier.GOLD:
                    self._stats["gold_setups"] += 1
                
                # Track special setups
                if result.is_kill_shot:
                    self._stats["kill_shots"] += 1
                if result.is_pump_fade:
                    self._stats["pump_fades"] += 1
                    
            elif result.filter_result == FilterResult.REJECT_BTC_DUMP:
                self._stats["filtered_by_btc"] += 1
                logger.debug(f"❌ {result.symbol}: BTC Dump")
            elif result.filter_result == FilterResult.REJECT_MTF_TREND:
                self._stats["filtered_by_trend"] += 1
                logger.debug(f"❌ {result.symbol}: MTF Trend - {result.filter_reason}")
            elif result.filter_result == FilterResult.REJECT_LOW_RR:
                self._stats["filtered_by_score"] += 1
                logger.debug(f"❌ {result.symbol}: Low Score - {result.filter_reason}")
        
        # Debug: log rejection stats
        logger.info(f"📈 Found {len(valid_setups)} valid | Rejected: trend={self._stats.get('filtered_by_trend',0)}, score={self._stats.get('filtered_by_score',0)}, btc={self._stats.get('filtered_by_btc',0)}")
        
        # ========== SELECTION LOGIC ==========
        # 1. Pick best setup per strategy
        # 2. Add top tier setup (Diamond with most confluence)
        
        from collections import defaultdict
        
        # Group by strategy
        strategy_groups = defaultdict(list)
        for result in valid_setups:
            if result.setup:
                strategy_groups[result.setup.strategy.value].append(result)
        
        # Sort each group by confidence and select best from each strategy
        best_per_strategy = []
        for strategy, results in strategy_groups.items():
            # Sort by: tier -> confidence -> confluence -> R:R
            results.sort(
                key=lambda x: (
                    x.signal_tier == SignalTier.DIAMOND,
                    x.confidence_points,
                    x.confluence_score,
                    x.optimized_levels.risk_reward if x.optimized_levels else 0
                ),
                reverse=True
            )
            if results:
                best_per_strategy.append(results[0])
                logger.debug(f"📌 Best {strategy}: {results[0].symbol} (score={results[0].confidence_points})")
        
        # Find TOP TIER signal (Diamond with most confluence)
        all_sorted = sorted(
            valid_setups,
            key=lambda x: (
                x.is_kill_shot,  # Kill Shot first
                x.signal_tier == SignalTier.DIAMOND,
                x.confidence_points,
                x.confluence_score,
                x.is_golden
            ),
            reverse=True
        )
        
        top_tier_signal = all_sorted[0] if all_sorted else None
        
        # Combine: Best per strategy + Top Tier (avoid duplicates)
        selected_symbols = set()
        final_selection = []
        
        # Add top tier first (highest priority)
        if top_tier_signal and top_tier_signal.symbol not in selected_symbols:
            final_selection.append(top_tier_signal)
            selected_symbols.add(top_tier_signal.symbol)
            logger.info(f"🏆 TOP TIER: {top_tier_signal.symbol} ({top_tier_signal.signal_tier.value}, score={top_tier_signal.confidence_points})")
        
        # Add best from each strategy
        for result in best_per_strategy:
            if result.symbol not in selected_symbols:
                final_selection.append(result)
                selected_symbols.add(result.symbol)
        
        # Log selection summary
        strategy_count = len([r for r in final_selection if r != top_tier_signal])
        logger.info(f"📋 Selection: 1 Top Tier + {strategy_count} Strategy Best = {len(final_selection)} total")
        
        # Get rate limit status
        rate_status = await self.redis.get_rate_limit_status()
        
        # Send alerts with rate limiting
        sent_count = 0
        strategies_sent = set()
        
        for result in final_selection:
            # Check rate limit before sending
            can_send, mode, alerts_count = await self.redis.can_send_alert(
                result.confidence_points
            )
            
            if not can_send:
                self._stats["filtered_by_rate_limit"] += 1
                logger.info(f"⏸️ Rate limit [{mode}]: {result.symbol} (score={result.confidence_points})")
                continue
            
            await self._send_alert(result)
            
            # Record alert for rate limiting
            await self.redis.record_alert_sent()
            
            self._stats["alerts_sent"] += 1
            sent_count += 1
            if result.setup:
                strategies_sent.add(result.setup.strategy.value)
            await asyncio.sleep(3)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        btc_status = "🔴 DUMP" if self.trade_filter.is_btc_dumping() else "🟢 OK"
        diamond_count = sum(1 for r in final_selection if r.signal_tier == SignalTier.DIAMOND)
        kill_count = sum(1 for r in final_selection if r.is_kill_shot)
        
        logger.info(
            f"✅ Scan #{self._stats['scans']} in {elapsed:.1f}s | "
            f"Sent {sent_count}/{len(final_selection)} | 💎{diamond_count} 🔥{kill_count} | "
            f"Strategies: {len(strategies_sent)} | Rate: {rate_status['message']} | BTC: {btc_status}"
        )
        
        # Note: Sheets trade updates moved to dedicated _run_sheets_monitor task
        # to avoid quota exceeded errors (runs every 5 min instead of every scan)
    
    async def scan_single_coin(self, symbol: str):
        """
        Scan a single coin triggered by RealTimeSignalEngine.
        Used for event-driven analysis instead of polling.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
        """
        logger.info(f"⚡ Real-time scan: {symbol}")
        
        try:
            # Create FilteredTicker for the coin
            ticker = await self.rest_client.get_futures_ticker(symbol)
            if not ticker:
                logger.warning(f"⚠️ Could not fetch ticker for {symbol}")
                return
            
            coin = FilteredTicker(
                symbol=symbol,
                price=float(ticker.get('lastPrice', 0)),
                volume_24h=float(ticker.get('quoteVolume', 0)),
                change_24h=float(ticker.get('priceChangePercent', '0').replace('%', ''))
            )
            
            # Run analysis
            result = await self._scan_coin(coin)
            
            if not result.success:
                return
            
            if result.filter_result != FilterResult.PASS:
                logger.info(f"📊 {symbol}: Filtered - {result.filter_reason}")
                return
            
            if not result.setup or not result.optimized_levels:
                return
            
            # Check rate limit
            can_send, mode, alerts_count = await self.redis.can_send_alert(
                result.confidence_points
            )
            
            if not can_send:
                logger.info(f"⏸️ Rate limit [{mode}]: {symbol} (score={result.confidence_points})")
                return
            
            # Send alert
            await self._send_alert(result)
            await self.redis.record_alert_sent()
            
            self._stats["alerts_sent"] += 1
            self._stats["realtime_triggers"] = self._stats.get("realtime_triggers", 0) + 1
            
            logger.info(f"✅ Real-time alert sent: {symbol} | score={result.confidence_points}")
            
        except Exception as e:
            logger.error(f"Error in scan_single_coin for {symbol}: {e}")
    
    async def _scan_coin(self, coin: FilteredTicker) -> ScanResult:
        """Scan single coin with v3 logic."""
        try:
            # Fetch klines
            klines_tasks = [
                self._fetch_klines_safe(coin.symbol, "5m", 100),
                self._fetch_klines_safe(coin.symbol, "15m", 100),
                self._fetch_klines_safe(coin.symbol, "1h", 200),
            ]
            klines_5m, klines_15m, klines_h1 = await asyncio.gather(*klines_tasks)
            
            if not all([klines_5m, klines_15m, klines_h1]):
                return ScanResult(symbol=coin.symbol, success=False, error="Missing klines")
            
            # Convert to DataFrames
            df_m5 = self._to_df(klines_5m)
            df_m15 = self._to_df(klines_15m)
            df_h1 = self._to_df(klines_h1)
            
            if df_h1.empty:
                return ScanResult(symbol=coin.symbol, success=False, error="Empty H1 data")
            
            # Calculate indicators
            indicators = self.indicator_calc.calculate(
                coin.symbol, klines_15m, klines_h1, klines_h1  # Use H1 for H4 too
            )
            if not indicators:
                return ScanResult(symbol=coin.symbol, success=False, error="Indicator error")
            
            # Detect strategies with v2 detector
            setups = self.strategy_detector.analyze(
                symbol=coin.symbol,
                df_m5=df_m5,
                df_m15=df_m15,
                df_h1=df_h1,
                current_price=indicators.price,
                atr=indicators.atr,
                wt1=indicators.wt1,
                wt2=indicators.wt2,
                rsi=indicators.rsi_h1,
                volume_ratio=indicators.volume_ratio
            )
            
            if not setups:
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    filter_result=FilterResult.REJECT_MTF_TREND,
                    filter_reason="No valid setup in current trend"
                )
            
            # Get best setup
            best_setup = setups[0]  # Already sorted by detector
            
            # Check actionability
            if not best_setup.is_actionable:
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_LOW_RR,
                    filter_reason=f"Price {best_setup.distance_to_entry_pct:.1f}% from entry"
                )
            
            # Apply BTC Correlation Filter via RiskManager
            direction = TradeDirection.LONG if best_setup.direction == "LONG" else TradeDirection.SHORT
            
            # Check BTC correlation for trade permission via RiskManager
            can_trade_allowed, can_trade_reason = self.risk_manager.can_trade(coin.symbol, direction.value)
            if not can_trade_allowed:
                self._stats["filtered_by_btc_correlation"] += 1
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_BTC_DUMP,
                    filter_reason=f"Risk: {can_trade_reason}"
                )
            
            # ========== STRATEGY-DIRECTION VALIDATION v3.0 ==========
            # Based on TradeHistory2 analysis:
            # - SFP LONG: -3194.3% PnL => BLOCKED
            # - EMA_PULLBACK LONG: -995.5% PnL => BLOCKED
            # - BB_BOUNCE LONG: -575.7% PnL => BLOCKED
            strat_dir_allowed, strat_dir_reason = self.trade_filter.validate_strategy_direction(
                strategy=best_setup.strategy.value,
                direction=best_setup.direction,
                  checklist_score=None  # Pre-check: let CONDITIONAL through, re-check after scoring
            )
            if not strat_dir_allowed:
                self._stats["filtered_by_strategy_direction"] = self._stats.get("filtered_by_strategy_direction", 0) + 1
                logger.info(f"🚫 {coin.symbol}: {strat_dir_reason}")
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_MTF_TREND,
                    filter_reason=f"Strategy-Direction Block: {strat_dir_reason}"
                )
            
            # Check if this is a mean reversion strategy (skip MTF filter)
            is_mean_reversion = best_setup.strategy.value in ["BB_BOUNCE", "SFP", "LIQ_SWEEP"]
            
            # Calculate optimized levels
            filter_result, filter_reason, optimized = self.trade_filter.filter_trade(
                symbol=coin.symbol,
                direction=direction,
                current_price=indicators.price,
                ema89_h1=indicators.ema89_h1,
                atr=indicators.atr,
                swing_high=indicators.swing_high_20,
                swing_low=indicators.swing_low_20,
                is_mean_reversion=is_mean_reversion
            )
            
            # If filter rejects (MTF/RR), return early with proper result
            if filter_result != FilterResult.PASS:
                logger.debug(f"⚠️ {coin.symbol}: Filter reject - {filter_result.value} - {filter_reason}")
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=filter_result,
                    filter_reason=filter_reason
                )
            
            # ========== SCORING SYSTEM v2.0 ==========
            # Prepare indicators dict for scoring
            indicator_dict = {
                'ema34_h1': indicators.ema34_h1,
                'ema89_h1': indicators.ema89_h1,
                'rsi_15m': indicators.rsi_15m,
                'rsi_divergence': indicators.rsi_divergence,
                'wt1': indicators.wt1,
                'wt2': indicators.wt2,
                'wt_signal': indicators.wt_signal,
                'volume_ratio': indicators.volume_ratio,
                'bb_upper': indicators.bb_upper,
                'in_ob_zone': best_setup.has_ob_confluence,
            }
            
            # Get swing levels
            swing_high_20 = float(df_m15['high'].tail(20).max())
            swing_low_20 = float(df_m15['low'].tail(20).min())
            
            # Evaluate Checklist (legacy)
            checklist = self.scoring_system.evaluate_checklist(
                df_m15=df_m15,
                df_h1=df_h1,
                indicators=indicator_dict,
                setup_type=best_setup.strategy.value,
                detected_direction=best_setup.direction
            )
            
            # ========== CHECKLIST VALIDATION v3.0 ==========
            # Based on TradeHistory2:
            # - 3/3 checklist: +1455.6% PnL, 56.1% WR (PROFITABLE)
            # - 2/3 checklist: -3471.4% PnL, 46.6% WR (LOSS, except SHORT)
            # - 1/3 checklist: -84.6% PnL, 30.8% WR (ALWAYS LOSS)
            checklist_score_num = checklist.total_score if hasattr(checklist, 'total_score') else 0
            min_checklist_required = self.trade_filter.get_checklist_requirements(best_setup.direction)
            
            if checklist_score_num < min_checklist_required:
                self._stats["filtered_by_checklist"] = self._stats.get("filtered_by_checklist", 0) + 1
                logger.info(f"🚫 {coin.symbol}: Checklist {checklist_score_num}/3 < {min_checklist_required}/3 required for {best_setup.direction}")
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_LOW_RR,
                    filter_reason=f"Checklist {checklist_score_num}/3 - {best_setup.direction} requires {min_checklist_required}/3",
                    checklist_score=checklist,
                    signal_grade=SignalGrade.D_REJECT
                )
            
            # Re-validate strategy-direction with checklist score (for CONDITIONAL strategies)
            strat_dir_allowed, strat_dir_reason = self.trade_filter.validate_strategy_direction(
                strategy=best_setup.strategy.value,
                direction=best_setup.direction,
                checklist_score=checklist_score_num
            )
            if not strat_dir_allowed:
                self._stats["filtered_by_strategy_direction"] = self._stats.get("filtered_by_strategy_direction", 0) + 1
                logger.info(f"🚫 {coin.symbol}: {strat_dir_reason} (checklist={checklist_score_num}/3)")
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_MTF_TREND,
                    filter_reason=strat_dir_reason,
                    checklist_score=checklist,
                    signal_grade=SignalGrade.D_REJECT
                )
            
            # Calculate Confidence Score (new v2.0)
            confidence = self.scoring_system.calculate_confidence(
                df_m15=df_m15,
                df_h1=df_h1,
                indicators=indicator_dict,
                setup_type=best_setup.strategy.value,
                detected_direction=best_setup.direction,
                has_sfp=best_setup.strategy in [StrategyType.SFP, StrategyType.LIQUIDITY_SWEEP],
                has_rsi_divergence=indicators.rsi_divergence in ["Bullish", "Bearish"],
                has_volume_spike=indicators.volume_ratio >= 2.0,
                has_wavetrend_cross=indicators.wt_signal in ["Bullish Cross", "Bearish Cross"],
                has_ob_confluence=best_setup.has_ob_confluence,
                swing_high_20=swing_high_20,
                swing_low_20=swing_low_20
            )
            
            # Evaluate 4-Layer Filter for SHORT
            four_layer = None
            if best_setup.direction == "SHORT":
                four_layer = self.scoring_system.evaluate_4layer_short(
                    df=df_m15,
                    indicators=indicator_dict
                )
            
            # Get combined score
            signal_grade, vol_weight, grade_reasons = self.scoring_system.get_combined_score(
                checklist, four_layer
            )
            
            # Get tier from confidence score
            signal_tier = confidence.tier
            confidence_points = confidence.total_score
            
            # Log ALL scores (not just debug) for monitoring  
            logger.info(f"📊 SCORE: {coin.symbol} | {best_setup.strategy.value} {best_setup.direction} | score={confidence_points}/100")
            
            # Check if tradeable by tier
            if not confidence.is_tradeable:
                # Log coins close to threshold for monitoring
                if confidence_points >= 30:  # Lowered from 50 for better visibility
                    logger.info(f"📈 Near threshold: {coin.symbol} score={confidence_points}/100 ({best_setup.strategy.value})")
                return ScanResult(
                    symbol=coin.symbol,
                    success=True,
                    indicators=indicators,
                    setup=best_setup,
                    filter_result=FilterResult.REJECT_LOW_RR,
                    filter_reason=f"Score {confidence_points}/100 - Below Gold threshold",
                    checklist_score=checklist,
                    signal_grade=signal_grade,
                    confidence_score=confidence,
                    signal_tier=signal_tier,
                    confidence_points=confidence_points
                )
            
            # ========== ADAPTIVE SHORT FILTER ==========
            # Combined filters:
            # 1. Market Regime (EMA200): BULLISH requires Diamond, BEARISH allows Gold
            # 2. Market State (WARNING/DANGER): Only Diamond SHORT allowed
            if best_setup.direction == "SHORT":
                # Get current market context
                ctx = await self.context_manager.get_current()
                market_state = ctx.market_state if ctx else "CAUTION"
                short_diamond_allowed = getattr(ctx, 'short_diamond_allowed', True) if ctx else True
                
                # Check Market State restrictions
                if market_state == "CRASH":
                    logger.info(f"🚫 SHORT blocked: {coin.symbol} - CRASH state (emergency stop)")
                    return ScanResult(
                        symbol=coin.symbol,
                        success=True,
                        indicators=indicators,
                        setup=best_setup,
                        filter_result=FilterResult.REJECT_BTC_DUMP,
                        filter_reason=f"Market State: CRASH - All trading blocked",
                        checklist_score=checklist,
                        signal_grade=signal_grade,
                        confidence_score=confidence,
                        signal_tier=signal_tier,
                        confidence_points=confidence_points
                    )
                
                # In DANGER: Only Diamond SHORT allowed
                # In WARNING: Allow Diamond and GOLD SHORT (relaxed for bear market)
                if market_state == "DANGER":
                    if signal_tier != SignalTier.DIAMOND:
                        logger.info(f"🚫 SHORT blocked: {coin.symbol} is {signal_tier.value} but DANGER state requires DIAMOND")
                        return ScanResult(
                            symbol=coin.symbol,
                            success=True,
                            indicators=indicators,
                            setup=best_setup,
                            filter_result=FilterResult.REJECT_LOW_RR,
                            filter_reason=f"Market State: DANGER - Only Diamond SHORT allowed",
                            checklist_score=checklist,
                            signal_grade=signal_grade,
                            confidence_score=confidence,
                            signal_tier=signal_tier,
                            confidence_points=confidence_points
                        )
                elif market_state == "WARNING":
                    # In WARNING (bearish): Allow SILVER+ for SHORT (relaxed for bear market)
                    # Previously required GOLD+, now allow SILVER (score>=40) too
                    if signal_tier == SignalTier.REJECT:
                        logger.info(f"🚫 SHORT blocked: {coin.symbol} is REJECT tier (score<40) - minimum SILVER needed")
                        return ScanResult(
                            symbol=coin.symbol,
                            success=True,
                            indicators=indicators,
                            setup=best_setup,
                            filter_result=FilterResult.REJECT_LOW_RR,
                            filter_reason=f"Market State: WARNING - Minimum SILVER tier needed for SHORT",
                            checklist_score=checklist,
                            signal_grade=signal_grade,
                            confidence_score=confidence,
                            signal_tier=signal_tier,
                            confidence_points=confidence_points
                        )
                    else:
                        logger.info(f"✅ {signal_tier.value} SHORT allowed in {market_state}: {coin.symbol}")
                
                # Apply Market Regime filter (EMA200-based)
                min_tier_for_short = self.risk_manager.btc_filter.get_short_allowed_tier()
                regime = self.risk_manager.btc_filter.market_regime.value
                
                if min_tier_for_short == "DIAMOND" and signal_tier != SignalTier.DIAMOND:
                    logger.info(f"🚫 SHORT blocked: {coin.symbol} is {signal_tier.value} but market regime {regime} requires DIAMOND")
                    return ScanResult(
                        symbol=coin.symbol,
                        success=True,
                        indicators=indicators,
                        setup=best_setup,
                        filter_result=FilterResult.REJECT_LOW_RR,
                        filter_reason=f"Adaptive SHORT: {regime} regime requires Diamond tier",
                        checklist_score=checklist,
                        signal_grade=signal_grade,
                        confidence_score=confidence,
                        signal_tier=signal_tier,
                        confidence_points=confidence_points
                    )
                else:
                    logger.info(f"✅ SHORT allowed: {coin.symbol} {signal_tier.value} in {regime} regime")
            
            # Note: Position size and leverage are already set by trade_filter.calculate_optimized_levels()
            
            # Apply dynamic stoploss for Pump Fade / counter-trend
            if confidence.use_dynamic_sl and confidence.dynamic_sl_price > 0:
                if optimized and best_setup.direction == "SHORT":
                    optimized.stop_loss = confidence.dynamic_sl_price
            
            return ScanResult(
                symbol=coin.symbol,
                success=True,
                indicators=indicators,
                setup=best_setup,
                filter_result=FilterResult.PASS,
                optimized_levels=optimized,
                checklist_score=checklist,
                four_layer_result=four_layer,
                signal_grade=signal_grade,
                grade_reasons=grade_reasons,
                confidence_score=confidence,
                signal_tier=signal_tier,
                confidence_points=confidence_points,
                is_pump_fade=confidence.is_pump_fade,
                is_kill_shot=confidence.is_kill_shot,
                dynamic_sl_price=confidence.dynamic_sl_price
            )
            
        except Exception as e:
            logger.info(f"⚠️ Scan error {coin.symbol}: {e}")
            return ScanResult(symbol=coin.symbol, success=False, error=str(e))
    
    async def _fetch_klines_safe(self, symbol: str, interval: str, limit: int):
        """Safe klines fetch."""
        try:
            return await self.rest_client.get_futures_klines(symbol, interval, limit)
        except:
            return None
    
    async def _fetch_with_retry(self, func):
        """Fetch with retry."""
        try:
            return await self.retry_handler.execute(func, self.ws_circuit_breaker)
        except:
            return None
    
    async def _update_btc_from_rest(self):
        """Update BTC from REST API and sync with RiskManager."""
        try:
            # Get BTC ticker
            btc_ticker = await self.rest_client.get_ticker("BTC-USDT")
            if not btc_ticker:
                return
            
            btc_price = float(btc_ticker.get("lastPrice", 0))
            
            # Get klines for change calculation
            klines = await self.rest_client.get_futures_klines("BTC-USDT", "1h", 5)
            if klines and len(klines) >= 2:
                df = self._to_df(klines)
                if not df.empty:
                    current = df['close'].iloc[-1]
                    prev_1h = df['close'].iloc[-2]
                    change_1h = ((current - prev_1h) / prev_1h) * 100
                    
                    # Get 4h change (approx)
                    if len(df) >= 5:
                        prev_4h = df['close'].iloc[-5]
                        change_4h = ((current - prev_4h) / prev_4h) * 100
                    else:
                        change_4h = 0
                    
                    # Update old trade_filter for compatibility
                    self.trade_filter.update_btc_state(current, prev_1h)
                    
                    # Update RiskManager's BTC filter with EMA200 for Market Regime
                    ctx = await self.context_manager.get_current()
                    if ctx:
                        # Update BTC trend for MTF filter
                        self.trade_filter.update_btc_trend(
                            trend=ctx.btc_trend,
                            ema_distance_pct=ctx.btc_ema89_distance_pct
                        )
                        self.risk_manager.update_btc_state(
                            price=btc_price,
                            change_1h=change_1h,
                            change_4h=change_4h,
                            ema34_h4=ctx.btc_ema34_h4,
                            ema89_h4=ctx.btc_ema89_h4,
                            ema200_h4=getattr(ctx, 'btc_ema200_h4', ctx.btc_ema89_h4)
                        )
        except Exception as e:
            logger.debug(f"BTC update error: {e}")
    
    def _to_df(self, klines: List) -> pd.DataFrame:
        """Convert klines to DataFrame."""
        if not klines:
            return pd.DataFrame()
        
        if isinstance(klines[0], list):
            data = [{'time': k[0], 'open': float(k[1]), 'high': float(k[2]),
                     'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])}
                    for k in klines if len(k) >= 6]
        else:
            data = [{'time': int(k.get('time', 0)), 'open': float(k.get('open', 0)),
                     'high': float(k.get('high', 0)), 'low': float(k.get('low', 0)),
                     'close': float(k.get('close', 0)), 'volume': float(k.get('volume', 0))}
                    for k in klines]
        
        df = pd.DataFrame(data)
        if 'time' in df.columns and not df.empty:
            df = df.sort_values('time', ascending=True).reset_index(drop=True)
        return df
    
    async def _send_alert(self, result: ScanResult):
        """Send alert."""
        try:
            if not result.indicators or not result.setup or not result.optimized_levels:
                return
            
            # Check cooldown
            if await self.redis.check_cooldown(result.symbol, result.setup.strategy.value):
                return
            
            # ========== REFRESH REALTIME PRICE ==========
            # Get fresh ticker right before sending to avoid stale entry price
            try:
                fresh_ticker = await self.rest_client.get_futures_ticker(result.symbol)
                if fresh_ticker:
                    fresh_price = float(fresh_ticker.get('lastPrice', 0))
                    old_entry = result.optimized_levels.entry
                    
                    # Update entry and recalculate levels if price changed significantly
                    if fresh_price > 0:
                        price_diff_pct = abs(fresh_price - old_entry) / old_entry * 100
                        
                        # If price moved >0.5%, update levels
                        if price_diff_pct > 0.5:
                            logger.info(f"💹 {result.symbol}: Refreshed price {old_entry:.6g} → {fresh_price:.6g} (+{price_diff_pct:.2f}%)")
                            
                            # Convert direction string to enum
                            from ..analysis.trade_filter import TradeDirection
                            direction_enum = TradeDirection.LONG if result.setup.direction == "LONG" else TradeDirection.SHORT
                            
                            # Recalculate levels with fresh price
                            new_levels = self.trade_filter.calculate_optimized_levels(
                                symbol=result.symbol,
                                direction=direction_enum,
                                entry_price=fresh_price,
                                atr=result.indicators.atr,
                                swing_high=result.indicators.swing_high_20,
                                swing_low=result.indicators.swing_low_20
                            )
                            result.optimized_levels = new_levels
            except Exception as e:
                logger.debug(f"Price refresh failed: {e}")
            
            # Funding rate
            funding_rate = None
            try:
                funding = await self.rest_client.get_funding_rate(result.symbol)
                if funding:
                    funding_rate = float(funding.get('funding_rate', 0))
            except:
                pass
            
            # Generate chart
            klines_h1 = await self.rest_client.get_futures_klines(result.symbol, "1h", 100)
            chart_bytes = None
            
            if klines_h1:
                # Get optimized levels for TP/SL
                levels = result.optimized_levels
                
                chart_bytes = self.chart_generator.generate_chart(
                    symbol=result.symbol,
                    klines=klines_h1,
                    interval="1h",
                    trade_direction=result.setup.direction,
                    # Trade levels
                    entry_price=levels.entry if levels else None,
                    stop_loss=levels.stop_loss if levels else None,
                    take_profit_1=levels.take_profit_1 if levels else None,
                    take_profit_2=levels.take_profit_2 if levels else None,
                    take_profit_3=levels.take_profit_3 if levels else None,
                    # SMC levels
                    swing_high=result.indicators.swing_high_20,
                    swing_low=result.indicators.swing_low_20,
                    trend=result.indicators.trend_h1,
                    zone=result.setup.zone_type,
                    strategy_name=result.setup.name
                )
            
            # Format message
            message = self._format_alert(result, funding_rate)
            
            # Buttons
            buttons = AlertFormatter.get_inline_buttons(result.symbol)
            
            # Send and get message_id for reply tracking
            message_id = None
            if chart_bytes:
                message_id = await self._send_photo(chart_bytes, message, buttons)
            else:
                message_id = await self.notifier.send_message(message)
            
            # ========== TRACK SIGNAL FOR INVALIDATION ==========
            # Track signal with RiskManager for invalidation monitoring
            levels = result.optimized_levels
            if levels and result.setup:
                # Get swing invalidation level based on direction
                swing_invalidation = (
                    result.indicators.swing_low_20 if result.setup.direction == "LONG" 
                    else result.indicators.swing_high_20
                )
                
                self.risk_manager.track_signal(
                    symbol=result.symbol,
                    direction=result.setup.direction,
                    entry=levels.entry,
                    stop_loss=levels.stop_loss,
                    swing_invalidation=swing_invalidation
                )
                
                # ========== LOG TO GOOGLE SHEETS ==========
                # Check for duplicate - if coin already has open trade, skip logging
                coin_name = result.symbol.replace('-USDT', '')
                has_open = await self.sheets_client.has_open_trade(coin_name)
                
                if has_open:
                    logger.warning(f"⚠️ {coin_name} already has open trade - NOT logging duplicate")
                else:
                    trade_id = str(uuid.uuid4())[:8]
                    
                    # Log to Google Sheets
                    # Extract grade/layers/checklist info for tracking
                    grade_str = result.signal_tier.value if result.signal_tier else ""
                    
                    # Format layers passed (e.g., "4/4", "3/4")
                    layers_str = ""
                    if result.four_layer_result:
                        fl = result.four_layer_result
                        # Use the layers_passed property or count manually
                        passed = fl.layers_passed if hasattr(fl, 'layers_passed') else sum([
                            getattr(fl, 'layer1_pass', False),
                            getattr(fl, 'layer2_pass', False),
                            getattr(fl, 'layer3_pass', False),
                            getattr(fl, 'layer4_pass', False)
                        ])
                        layers_str = f"{passed}/4"
                    
                    # Format checklist score (e.g., "3/3", "2/3")  
                    checklist_str = ""
                    if result.checklist_score:
                        cs = result.checklist_score
                        # Use total_score (0-3 legacy format)
                        checklist_str = f"{cs.total_score}/3"
                    
                    trade_record = TradeRecord(
                        trade_id=trade_id,
                        date=datetime.now().strftime("%d/%m/%Y"),
                        coin=coin_name,
                        signal=result.setup.direction,
                        leverage=levels.leverage,
                        entry=levels.entry,
                        stoploss=levels.stop_loss,
                        take_profit=levels.take_profit_1,  # Primary TP
                        note=result.setup.strategy.value,
                        grade=grade_str,
                        layers_passed=layers_str,
                        checklist_score=checklist_str
                    )
                    
                    # Try to log to sheet - if fails, skip trade tracking
                    try:
                        sheet_row = await self.sheets_client.log_trade(trade_record)
                        
                        # ========== ADD TO TRADE TRACKER ==========
                        active_trade = ActiveTrade(
                            trade_id=trade_id,
                            symbol=result.symbol,
                            direction=result.setup.direction,
                            entry_price=levels.entry,
                            stop_loss=levels.stop_loss,
                            take_profit_1=levels.take_profit_1,
                            take_profit_2=levels.take_profit_2,
                            take_profit_3=levels.take_profit_3,
                            leverage=levels.leverage,
                            position_size=levels.position_size,
                            sheet_row=sheet_row,
                            chat_id=config.telegram.chat_id,
                            message_id=message_id or 0  # Store message_id for reply
                        )
                        
                        self.trade_tracker.add_trade(active_trade)
                        logger.info(f"✅ Trade tracked: {result.symbol} @ row {sheet_row}")
                        
                    except Exception as sheet_error:
                        # Sheet log failed - alert user but don't track trade
                        logger.error(f"⚠️ Sheet log failed for {result.symbol}: {sheet_error}")
                        logger.error(f"   Alert sent but NOT TRACKED in sheet/tracker")
                        
                        # Optionally send warning to user
                        try:
                            await self.telegram.send_message(
                                f"⚠️ CẢNH BÁO: {result.symbol} alert đã gửi nhưng KHÔNG LOG VÔ SHEET\n"
                                f"Lỗi: Quota exceeded - quá nhiều write request\n"
                                f"Trade này KHÔNG được theo dõi tự động"
                            )
                        except:
                            pass
            
            # Cooldown
            await self.redis.set_cooldown(
                result.symbol,
                result.setup.strategy.value,
                1800  # 30 min cooldown
            )
            
            logger.info(f"📤 Sent: {result.symbol} - {result.setup.name} - {result.setup.direction}")
            
        except Exception as e:
            logger.error(f"Send error: {e}")
    
    def _format_alert(self, result: ScanResult, funding_rate: Optional[float]) -> str:
        """Format alert message with Scoring System v2.0."""
        import html
        
        setup = result.setup
        ind = result.indicators
        lvl = result.optimized_levels
        
        direction_emoji = "🟢" if setup.direction == "LONG" else "🔴"
        sym = result.symbol.replace('-USDT', '')
        
        # Tier label
        tier = result.signal_tier
        tier_label = ""
        if tier == SignalTier.DIAMOND:
            tier_label = "💎 DIAMOND SETUP"
        elif tier == SignalTier.GOLD:
            tier_label = "🥇 GOLD SETUP"
        
        # Kill Shot / Pump Fade tag
        special_tag = ""
        if result.is_kill_shot:
            special_tag = " 🔥 KILL SHOT"
        elif result.is_pump_fade:
            special_tag = " ⚡ PUMP FADE"
        elif result.is_golden:
            special_tag = " 🏆 GOLDEN"
        
        # Confidence score display
        conf = result.confidence_score
        score_text = f"📊 Score: {result.confidence_points}/100 | {tier.value}"
        
        # Confluence info
        confluence_items = []
        if setup.has_rsi_divergence:
            confluence_items.append("RSI Div")
        if setup.has_wavetrend_cross:
            confluence_items.append("WT Cross")
        if setup.has_volume_spike:
            confluence_items.append("Vol Spike")
        if setup.has_ob_confluence:
            confluence_items.append("OB Zone")
        if result.is_pump_fade:
            confluence_items.append("Shooting Star")
        
        confluence_text = " | ".join(confluence_items) if confluence_items else "Standard"
        
        btc_change = self.trade_filter._btc_change_pct
        btc_emoji = "🔴" if btc_change < -0.3 else "🟢" if btc_change > 0.3 else "🟡"
        
        # Volume recommendation based on tier
        vol_pct = int(result.volume_weight * 100)
        vol_recommend = f"{vol_pct}% Vol" if vol_pct < 100 else "100% Vol"
        
        # Build message
        lines = [
            tier_label,
            f"{direction_emoji} <b>{sym} {setup.direction}</b>{special_tag}",
            score_text,
            "━━━━━━━━━━━━━━━━━━━━━",
            "",
            f"📌 <b>{setup.name}</b>",
            f"📍 Zone: {html.escape(setup.zone_type)}",
            f"🔗 {confluence_text}",
        ]
        
        # Confidence breakdown
        if conf and conf.breakdown:
            lines.append("")
            lines.append("<b>📋 SCORING MATRIX</b>")
            for item in conf.breakdown[:6]:  # Limit to 6 items
                lines.append(html.escape(item[:40]))
        
        # 4-Layer for SHORT (only if Pump Fade)
        if result.four_layer_result and setup.direction == "SHORT" and result.is_pump_fade:
            fl = result.four_layer_result
            lines.append("")
            lines.append("<b>🔍 PUMP FADE FILTER</b>")
            l1_icon = "✅" if fl.layer1_pass else "❌"
            l2_icon = "✅" if fl.layer2_pass else "❌"
            l3_icon = "✅" if fl.layer3_pass else "❌"
            l4_icon = "✅" if fl.layer4_pass else "❌"
            lines.append(f"{l1_icon} Context: {html.escape(fl.layer1_reason[:20])}")
            lines.append(f"{l2_icon} Candle: {html.escape(fl.layer2_reason[:20])}")
            lines.append(f"{l3_icon} Volume: {html.escape(fl.layer3_reason[:20])}")
            lines.append(f"{l4_icon} Safety: {html.escape(fl.layer4_reason[:20])}")
        
        # Market info with all indicators
        lines.append("")
        lines.append("<b>📈 INDICATORS</b>")
        lines.append(f"Price: {ind.price:,.8g} | ATR: {ind.atr:,.8g}")
        lines.append(f"RSI: {ind.rsi_h1:.0f} | MFI: {getattr(ind, 'mfi', 50):.0f}")
        lines.append(f"WT: {ind.wt1:.0f}/{ind.wt2:.0f} ({html.escape(str(ind.wt_signal))})")
        lines.append(f"BB: {ind.bb_lower:,.6g} - {ind.bb_upper:,.6g}")
        lines.append(f"EMA H1: {ind.ema34_h1:,.6g} / {ind.ema89_h1:,.6g}")
        lines.append(f"Vol: {ind.volume_ratio:.1f}x | {btc_emoji} BTC: {btc_change:+.2f}%")
        
        if ind.rsi_divergence and ind.rsi_divergence != "None":
            lines.append(f"⚡ RSI Div: {html.escape(ind.rsi_divergence)}")
        
        if funding_rate is not None:
            fr_emoji = "📉" if funding_rate < 0 else "📈"
            lines.append(f"{fr_emoji} Funding: {funding_rate:+.4f}%")
        
        # Trade info
        lines.append("")
        lines.append(f"<b>🎯 TRADE ({lvl.leverage}x) - {vol_recommend}</b>")
        lines.append("━━━━━━━━━━━━━━━━━━━━━")
        lines.append(f"{direction_emoji} <b>{setup.direction}</b>")
        lines.append("")
        lines.append(f"Entry: {lvl.entry:,.8g} ⚡ (Market)")
        
        # Show dynamic SL for counter-trend
        if result.dynamic_sl_price > 0:
            lines.append(f"SL: {result.dynamic_sl_price:,.8g} (Dynamic)")
        else:
            lines.append(f"SL: {lvl.stop_loss:,.8g}")
        
        lines.append(f"TP1: {lvl.take_profit_1:,.8g} (2R)")
        lines.append(f"TP2: {lvl.take_profit_2:,.8g} (4R)")
        lines.append(f"TP3: {lvl.take_profit_3:,.8g} (6R)")
        lines.append("")
        lines.append(f"📊 <b>R:R = 1:{lvl.risk_reward}</b> | Size: {lvl.position_size:,.0f}$")
        lines.append(f"💀 Liq: {lvl.liquidation_price:,.8g}")
        
        # Counter-trend warning
        if conf and conf.is_counter_trend:
            lines.append("")
            lines.append("⚠️ <b>COUNTER-TREND</b> - Đánh ngược xu hướng H1")
        
        lines.append("")
        lines.append(f"⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}")
        
        return "\n".join(lines)
    
    async def _send_trade_close_notification(self, trade: Dict):
        """Send Telegram notification when trade auto-closes (TP/SL hit)."""
        try:
            if not self.notifier or not self.notifier._enabled:
                return
            
            symbol = trade['symbol'].replace('-USDT', '')
            status = trade['status']
            pnl = trade['pnl']
            close_time = trade['close_time']
            direction = trade['direction']
            
            # Format message
            if status == "TP":
                emoji = "🎯✅"
                result_text = "TAKE PROFIT HIT"
                color = "🟢"
            else:  # SL
                emoji = "🛑❌"
                result_text = "STOP LOSS HIT"
                color = "🔴"
            
            direction_emoji = "📈" if direction == "LONG" else "📉"
            pnl_emoji = "💰" if pnl > 0 else "💸"
            
            message = f"""
{emoji} <b>TRADE AUTO-CLOSED</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

{direction_emoji} <b>{symbol}</b> {direction}
{color} <b>{result_text}</b>

{pnl_emoji} <b>PnL:</b> <code>{pnl:+.2f}%</code>
⏰ <b>Closed:</b> {close_time}

<i>End Trade checkbox auto-marked ✅</i>
"""
            
            await self.notifier._send_message(message.strip())
            logger.info(f"📤 Sent trade close notification: {symbol} {status} {pnl:+.2f}%")
            
        except Exception as e:
            logger.error(f"Error sending trade close notification: {e}")
    
    async def _send_photo(self, photo_bytes: bytes, caption: str, buttons: list) -> Optional[int]:
        """Send photo and return message_id."""
        from telegram import InlineKeyboardMarkup
        
        if not self.notifier._enabled or not self.notifier._bot:
            return None
        
        try:
            keyboard = InlineKeyboardMarkup(buttons)
            sent = await self.notifier._bot.send_photo(
                chat_id=self.notifier.chat_id,
                photo=photo_bytes,
                caption=caption[:1024],
                parse_mode='HTML',
                reply_markup=keyboard
            )
            return sent.message_id
        except Exception as e:
            logger.error(f"Photo error: {e}")
            await self.notifier._send_message_with_buttons(caption, buttons)
            return None
    
    async def _send_startup_message(self):
        """Send startup message."""
        msg = f"""
🤖 <b>BingX Alert Bot v3.5</b>
━━━━━━━━━━━━━━━━━━━━━━━━━━

<b>🎯 SCORING SYSTEM v2.1</b>

<b>📊 CONFIDENCE MATRIX (0-100):</b>
• PUMP_FADE: +30 pts
• BB_BOUNCE: +30 pts
• LIQ_SWEEP: +25 pts
• SFP: +25 pts
• BREAKOUT: +20 pts
• EMA_ALIGN: +20 pts

<b>✨ CONFIRMATIONS:</b>
• RSI Divergence: +15 pts
• Fib Golden Pocket: +15 pts
• Fib Standard: +10 pts
• Volume Spike: +10 pts
• WaveTrend: +10 pts
• OB Confluence: +10 pts

<b>⚠️ PENALTIES:</b>
• Counter-Trend: -25 pts

<b>🏆 TIER SYSTEM:</b>
💎 DIAMOND (≥80): Priority, $2 Full
🥇 GOLD (60-79): Standard, $1 Half
🥈 SILVER (45-59): Skip
❌ REJECT (less than 45): Ignore

<b>🛡️ RISK MANAGEMENT:</b>
• BTC Correlation Filter ✅
• Circuit Breaker (5% daily) ✅
• News Filter (Auto pause) ✅
• Rate Limiter (10/hour) ✅

<b>📊 GOOGLE SHEETS:</b>
• Auto trade logging ✅
• Real-time PnL updates ✅
• 18 columns tracking ✅

<b>🌐 SESSION ALERTS:</b>
• Asia: 00:00 UTC
• Europe: 07:00 UTC
• US: 13:30 UTC

<b>⚙️ CONFIG:</b>
• Scan: every 90s
• Mode: WebSocket Real-time
• Coins: 48 monitored

⏰ {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}


"""
        await self.notifier._send_message(msg.strip())
    
    def pause(self):
        """Pause scanning."""
        self._paused = True
        logger.info("⏸️ Paused")
    
    def resume(self):
        """Resume scanning."""
        self._paused = False
        logger.info("▶️ Resumed")
    
    async def stop(self):
        """Stop."""
        self._running = False
        # Stop news manager first
        if self.news_manager:
            await self.news_manager.stop()
        if self.ws_client:
            await self.ws_client.disconnect()
        if self.context_manager:
            await self.context_manager.stop()
        if self.notifier:
            await self.notifier.disconnect()
        await self.rest_client.disconnect()
        await self.redis.disconnect()
        logger.info("Stopped")
    
    @property
    def stats(self) -> Dict:
        """Get stats."""
        news_status = {}
        if self.news_manager:
            news_status = {
                "news_paused": self.news_manager.is_trading_paused,
                "news_poller_mode": self.news_manager.poller.mode.value if self.news_manager.poller else None,
            }
        
        return {
            **self._stats,
            "paused": self._paused,
            "paused_for_news": self._paused_for_news,
            "running": self._running,
            "btc_dumping": self.trade_filter.is_btc_dumping(),
            "btc_change_15m": f"{self.trade_filter._btc_change_pct:+.2f}%",
            "circuit_state": self.ws_circuit_breaker.state.value,
            **news_status,
        }


async def main():
    """Entry point."""
    manager = AlertManager()
    try:
        await manager.start()
    except KeyboardInterrupt:
        await manager.stop()
    except Exception as e:
        logger.error(f"Fatal: {e}")
        await manager.stop()
        raise


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())
