"""
IE Trade Module - ICT Entry Strategy

A standalone module implementing the 4-step ICT entry methodology:
1. Daily Bias confirmation
2. H1 FVG detection in Premium/Discount zones
3. M5 MSS (Market Structure Shift) confirmation
4. M5 FVG entry with calculated SL/TP

This module operates independently from the main zone alert system.

Commands:
- /dbias B or /dbias S - Set daily bias
- /iestatus - Show module status
- /iestart - Start scanning
- /iestop - Stop scanning
"""

from .config import IETradeConfig, KillZone
from .bias_manager import BiasManager, DailyBias, BiasState, BiasScheduler
from .fvg_detector import FVGDetector, FVG, FVGType, ZoneType, Candle, candles_from_api_data
from .mss_detector import MSSDetector, MSS, MSSType, SwingPoint
from .entry_calculator import EntryCalculator, TradeSetup
from .scanner import IEScanner, ScanPhase, CoinState, create_ie_scanner
from .commands import IETradeCommandHandler, setup_ie_trade_commands
from .sheet_logger import IESheetLogger, IETradeRecord, create_ie_sheet_logger
from .runner import IETradeRunner, start_ie_trade, stop_ie_trade, get_ie_trade_runner

__all__ = [
    # Config
    'IETradeConfig',
    'KillZone',
    
    # Bias
    'BiasManager',
    'DailyBias',
    'BiasState',
    'BiasScheduler',
    
    # FVG Detection
    'FVGDetector',
    'FVG',
    'FVGType',
    'ZoneType',
    'Candle',
    'candles_from_api_data',
    
    # MSS Detection
    'MSSDetector', 
    'MSS',
    'MSSType',
    'SwingPoint',
    
    # Entry Calculation
    'EntryCalculator',
    'TradeSetup',
    
    # Scanner
    'IEScanner',
    'ScanPhase',
    'CoinState',
    'create_ie_scanner',
    
    # Commands
    'IETradeCommandHandler',
    'setup_ie_trade_commands',
    
    # Sheet Logging
    'IESheetLogger',
    'IETradeRecord',
    'create_ie_sheet_logger',
    
    # Runner
    'IETradeRunner',
    'start_ie_trade',
    'stop_ie_trade',
    'get_ie_trade_runner',
]

__version__ = '1.0.0'
