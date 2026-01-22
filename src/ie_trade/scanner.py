"""
IE Trade Scanner

Main scanning loop that:
1. Monitors top coins for H1 FVG setups
2. Waits for price to enter FVG zone
3. Watches M5 for MSS confirmation
4. Calculates entry and sends alerts (Kill Zone only)
5. Manages multi-setup filtering and position limits
"""

import logging
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set, Callable, Awaitable
from enum import Enum

from .config import IETradeConfig, DEFAULT_CONFIG
from .bias_manager import BiasManager, DailyBias
from .fvg_detector import FVGDetector, FVG, Candle, candles_from_api_data
from .mss_detector import MSSDetector, MSS
from .entry_calculator import EntryCalculator, TradeSetup


logger = logging.getLogger(__name__)


class ScanPhase(Enum):
    """Current scanning phase for a coin"""
    WAITING_BIAS = "WAITING_BIAS"      # No bias set
    SCANNING_FVG = "SCANNING_FVG"      # Looking for H1 FVG
    MONITORING_FVG = "MONITORING_FVG"  # Price approaching FVG
    IN_FVG_ZONE = "IN_FVG_ZONE"        # Price in FVG, watching for MSS
    MSS_DETECTED = "MSS_DETECTED"      # MSS confirmed, calculating entry
    SETUP_READY = "SETUP_READY"        # Setup ready, waiting for kill zone
    ALERTED = "ALERTED"                # Alert sent


@dataclass
class CoinState:
    """Tracking state for each coin"""
    symbol: str
    phase: ScanPhase = ScanPhase.WAITING_BIAS
    
    # H1 FVG tracking
    active_fvg: Optional[FVG] = None
    fvg_found_at: Optional[datetime] = None
    
    # M5 MSS tracking
    mss: Optional[MSS] = None
    
    # Setup
    setup: Optional[TradeSetup] = None
    
    # Alert tracking
    last_alert_at: Optional[datetime] = None
    alerts_today: int = 0
    
    def reset(self) -> None:
        """Reset state for new scan cycle"""
        self.phase = ScanPhase.SCANNING_FVG
        self.active_fvg = None
        self.fvg_found_at = None
        self.mss = None
        self.setup = None


@dataclass
class ScanResult:
    """Result of a scan cycle"""
    timestamp: datetime
    coins_scanned: int
    fvgs_found: int
    coins_in_zone: int
    mss_detected: int
    setups_ready: int
    alerts_sent: int
    kill_zone: str
    scan_duration_ms: int


class IEScanner:
    """
    Main scanner for IE Trade module
    
    Workflow:
    1. Check if bias is set
    2. For each top coin:
       a. Fetch H1 candles, detect FVG in Premium/Discount zone
       b. If FVG found, monitor price
       c. When price enters FVG, fetch M5 candles
       d. Detect MSS on M5
       e. If MSS confirmed, calculate setup
       f. If in Kill Zone, send alert
    """
    
    def __init__(
        self,
        config: IETradeConfig = DEFAULT_CONFIG,
        fetch_candles: Optional[Callable] = None,
        send_alert: Optional[Callable] = None,
        log_to_sheet: Optional[Callable] = None
    ):
        self.config = config
        self.bias_manager = BiasManager(config)
        self.fvg_detector = FVGDetector(config)
        self.mss_detector = MSSDetector(config)
        self.entry_calculator = EntryCalculator(config)
        
        # External dependencies (injected)
        self.fetch_candles = fetch_candles
        self.send_alert = send_alert
        self.log_to_sheet = log_to_sheet
        
        # State tracking
        self.coin_states: Dict[str, CoinState] = {}
        self.active_positions: List[TradeSetup] = []
        self.pending_setups: List[TradeSetup] = []
        
        # Control
        self._running = False
        self._scan_task: Optional[asyncio.Task] = None
        self._scan_count = 0
        
        # Initialize coin states
        for symbol in config.TOP_COINS:
            self.coin_states[symbol] = CoinState(symbol=symbol)
    
    async def start(self) -> None:
        """Start the scanner"""
        if self._running:
            return
        
        self._running = True
        self._scan_task = asyncio.create_task(self._scan_loop())
        logger.info(f"🎯 IE Scanner started - monitoring {len(self.config.TOP_COINS)} coins")
    
    async def stop(self) -> None:
        """Stop the scanner"""
        self._running = False
        if self._scan_task:
            self._scan_task.cancel()
            try:
                await self._scan_task
            except asyncio.CancelledError:
                pass
        logger.info("🎯 IE Scanner stopped")
    
    async def _scan_loop(self) -> None:
        """Main scanning loop"""
        while self._running:
            try:
                result = await self._scan_cycle()
                self._scan_count += 1
                
                # Log summary
                kz_status = f"[{result.kill_zone}]" if result.kill_zone else "[No KZ]"
                logger.info(
                    f"🎯 IE Scan #{self._scan_count} {kz_status} | "
                    f"FVG:{result.fvgs_found} Zone:{result.coins_in_zone} "
                    f"MSS:{result.mss_detected} Ready:{result.setups_ready} "
                    f"Alerts:{result.alerts_sent} | {result.scan_duration_ms}ms"
                )
                
                # Wait before next scan
                await asyncio.sleep(self.config.SCAN_INTERVAL_SECONDS)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in IE scan loop: {e}", exc_info=True)
                await asyncio.sleep(60)
    
    async def _scan_cycle(self) -> ScanResult:
        """Execute one scan cycle"""
        start_time = datetime.utcnow()
        
        result = ScanResult(
            timestamp=start_time,
            coins_scanned=0,
            fvgs_found=0,
            coins_in_zone=0,
            mss_detected=0,
            setups_ready=0,
            alerts_sent=0,
            kill_zone="",
            scan_duration_ms=0
        )
        
        # 1. Check if bias is set
        if not self.bias_manager.is_bias_set:
            logger.debug("🎯 IE: No bias set, skipping scan")
            result.scan_duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return result
        
        bias = self.bias_manager.current_bias
        direction = "LONG" if bias == DailyBias.LONG else "SHORT"
        
        # 2. Check Kill Zone
        vn_hour = (datetime.utcnow().hour + 7) % 24
        is_kz, kz_name = self.config.is_kill_zone(vn_hour)
        result.kill_zone = kz_name
        
        # 3. Check position limit
        if len(self.active_positions) >= self.config.MAX_OPEN_POSITIONS:
            logger.debug(f"🎯 IE: Max positions ({self.config.MAX_OPEN_POSITIONS}) reached")
            result.scan_duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
            return result
        
        # 4. Scan each coin
        cycle_setups: List[TradeSetup] = []
        
        for symbol in self.config.TOP_COINS:
            result.coins_scanned += 1
            
            try:
                setup = await self._scan_coin(symbol, direction, is_kz, kz_name)
                
                # Update result counters
                state = self.coin_states[symbol]
                if state.active_fvg:
                    result.fvgs_found += 1
                if state.phase == ScanPhase.IN_FVG_ZONE:
                    result.coins_in_zone += 1
                if state.mss:
                    result.mss_detected += 1
                if setup and setup.is_valid:
                    result.setups_ready += 1
                    cycle_setups.append(setup)
                    
            except Exception as e:
                logger.warning(f"Error scanning {symbol}: {e}")
        
        # 5. Filter and select best setup
        if cycle_setups:
            best_setup = self.entry_calculator.filter_best_setup(cycle_setups)
            
            if best_setup and is_kz:
                # Send alert
                await self._send_setup_alert(best_setup)
                result.alerts_sent += 1
            elif best_setup and not is_kz:
                # Store for later (when Kill Zone starts)
                self.pending_setups.append(best_setup)
                logger.info(f"🎯 IE: Setup ready for {best_setup.symbol}, waiting for Kill Zone")
        
        result.scan_duration_ms = int((datetime.utcnow() - start_time).total_seconds() * 1000)
        return result
    
    async def _scan_coin(
        self, 
        symbol: str, 
        direction: str,
        is_kill_zone: bool,
        kill_zone_name: str
    ) -> Optional[TradeSetup]:
        """Scan a single coin for IE setup"""
        
        state = self.coin_states[symbol]
        
        # Fetch candles
        h1_candles = await self._fetch_candles(symbol, "1h", 100)
        if not h1_candles:
            return None
        
        current_price = h1_candles[-1].close
        
        # Phase 1: Find H1 FVG
        if state.phase in (ScanPhase.WAITING_BIAS, ScanPhase.SCANNING_FVG):
            state.phase = ScanPhase.SCANNING_FVG
            
            fvg = self.fvg_detector.find_best_fvg(
                candles=h1_candles,
                symbol=symbol,
                timeframe="1h",
                direction=direction,
                current_price=current_price
            )
            
            if fvg and fvg.is_valid:
                state.active_fvg = fvg
                state.fvg_found_at = datetime.utcnow()
                state.phase = ScanPhase.MONITORING_FVG
                logger.debug(f"🎯 IE: {symbol} H1 FVG found at {fvg.bottom:.4f}-{fvg.top:.4f}")
        
        # Phase 2: Monitor if price is approaching FVG
        if state.phase == ScanPhase.MONITORING_FVG and state.active_fvg:
            fvg = state.active_fvg
            
            # Check if FVG is still valid
            if not fvg.is_fresh(self.config.FVG_MAX_AGE_HOURS):
                state.reset()
                return None
            
            # Check if price is in FVG
            if fvg.is_price_in_gap(current_price):
                state.phase = ScanPhase.IN_FVG_ZONE
                logger.info(f"🎯 IE: {symbol} price entered H1 FVG zone!")
        
        # Phase 3: Watch for MSS on M5
        if state.phase == ScanPhase.IN_FVG_ZONE and state.active_fvg:
            m5_candles = await self._fetch_candles(symbol, "5m", 100)
            if not m5_candles:
                return None
            
            mss = self.mss_detector.detect_mss(
                candles=m5_candles,
                symbol=symbol,
                direction=direction,
                in_fvg_zone=True
            )
            
            if mss and mss.is_valid:
                state.mss = mss
                state.phase = ScanPhase.MSS_DETECTED
                logger.info(f"🎯 IE: {symbol} MSS detected! Displacement: {mss.displacement_body_ratio:.0%}")
        
        # Phase 4: Calculate setup
        if state.phase == ScanPhase.MSS_DETECTED and state.active_fvg and state.mss:
            m5_candles = await self._fetch_candles(symbol, "5m", 100)
            if not m5_candles:
                return None
            
            setup = self.entry_calculator.calculate_setup(
                symbol=symbol,
                direction=direction,
                current_price=current_price,
                h1_fvg=state.active_fvg,
                m5_fvg=state.mss.entry_fvg,
                mss=state.mss,
                h1_candles=h1_candles,
                m5_candles=m5_candles,
                kill_zone=kill_zone_name,
                daily_bias=direction
            )
            
            if setup:
                state.setup = setup
                state.phase = ScanPhase.SETUP_READY
                return setup
        
        return None
    
    async def _fetch_candles(
        self, 
        symbol: str, 
        timeframe: str, 
        limit: int
    ) -> List[Candle]:
        """Fetch candles from API"""
        if not self.fetch_candles:
            logger.warning("No fetch_candles function provided")
            return []
        
        try:
            data = await self.fetch_candles(symbol, timeframe, limit)
            return candles_from_api_data(data)
        except Exception as e:
            logger.warning(f"Error fetching candles for {symbol}: {e}")
            return []
    
    async def _send_setup_alert(self, setup: TradeSetup) -> None:
        """Send alert for a setup"""
        if not self.send_alert:
            logger.warning("No send_alert function provided")
            return
        
        try:
            message = setup.to_alert_message()
            await self.send_alert(message)
            
            # Update state
            state = self.coin_states[setup.symbol]
            state.phase = ScanPhase.ALERTED
            state.last_alert_at = datetime.utcnow()
            state.alerts_today += 1
            
            # Log to sheet
            if self.log_to_sheet:
                await self.log_to_sheet(setup.to_sheet_row())
            
            logger.info(f"🎯 IE: Alert sent for {setup.symbol} {setup.direction}")
            
        except Exception as e:
            logger.error(f"Error sending IE alert: {e}")
    
    def get_status(self) -> dict:
        """Get current scanner status"""
        bias_status = self.bias_manager.state
        
        phases = {}
        for symbol, state in self.coin_states.items():
            phases[state.phase.value] = phases.get(state.phase.value, 0) + 1
        
        vn_hour = (datetime.utcnow().hour + 7) % 24
        is_kz, kz_name = self.config.is_kill_zone(vn_hour)
        
        return {
            "running": self._running,
            "scan_count": self._scan_count,
            "bias": bias_status.bias.value if bias_status.is_active else "NONE",
            "bias_expires_in": f"{bias_status.hours_remaining:.1f}h" if bias_status.is_active else "N/A",
            "kill_zone": kz_name if is_kz else "Outside KZ",
            "coins_monitored": len(self.config.TOP_COINS),
            "phases": phases,
            "active_positions": len(self.active_positions),
            "pending_setups": len(self.pending_setups)
        }
    
    def reset_all_states(self) -> None:
        """Reset all coin states"""
        for state in self.coin_states.values():
            state.reset()
        self.pending_setups.clear()
        logger.info("🎯 IE: All states reset")


# Factory function to create scanner with dependencies
def create_ie_scanner(
    rest_client,  # BingX REST client
    telegram_bot,  # Telegram bot instance
    sheets_client = None,  # Optional Google Sheets client
    config: IETradeConfig = DEFAULT_CONFIG
) -> IEScanner:
    """
    Create IE Scanner with all dependencies wired up
    
    Args:
        rest_client: BingX REST API client
        telegram_bot: Telegram bot for sending alerts
        sheets_client: Optional Google Sheets client
        config: IE Trade configuration
        
    Returns:
        Configured IEScanner instance
    """
    
    async def fetch_candles(symbol: str, timeframe: str, limit: int):
        """Adapter for REST client"""
        return await rest_client.get_klines(symbol, timeframe, limit)
    
    async def send_alert(message: str):
        """Adapter for Telegram bot"""
        await telegram_bot.send_message(message, parse_mode='Markdown')
    
    async def log_to_sheet(row: dict):
        """Adapter for Sheets client"""
        if sheets_client:
            # This will need to be implemented based on your sheets_client API
            pass
    
    return IEScanner(
        config=config,
        fetch_candles=fetch_candles,
        send_alert=send_alert,
        log_to_sheet=log_to_sheet if sheets_client else None
    )
