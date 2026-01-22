"""
IE Trade Sheet Integration

Handles logging IE Trade setups to Google Sheets with:
- Note field set to "IE trade" for identification
- Proper formatting for ICT methodology data
- Kill Zone and Bias information
"""

import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IETradeRecord:
    """
    IE Trade record for Google Sheets.
    
    Mirrors the standard TradeRecord but with IE-specific fields.
    """
    # Core trade data
    date: str
    coin: str
    signal: str  # LONG/SHORT
    leverage: int
    entry: float
    stoploss: float
    take_profit: float  # TP1
    take_profit_2: float  # TP2
    
    # IE Trade specific
    rr_ratio: float  # Risk:Reward to TP1
    kill_zone: str  # "London" or "New York"
    daily_bias: str  # "LONG" or "SHORT"
    h1_fvg_range: str  # "0.00123-0.00125"
    mss_displacement: float  # 0.75 = 75%
    
    # Standard fields
    status: str = "OPEN"
    pnl_percent: float = 0.0
    note: str = "IE trade"
    message_id: int = 0
    
    # Grade fields (IE trades are considered high quality)
    grade: str = "A_SNIPER"  # IE trades are sniper entries
    layers_passed: str = "4/4"  # Full ICT methodology
    checklist_score: str = "3/3"  # H1 FVG + M5 MSS + Entry


class IESheetLogger:
    """
    Logger for IE Trade setups to Google Sheets.
    
    Uses the existing GoogleSheetsClient but formats
    trades with IE-specific information.
    """
    
    def __init__(self, sheets_client):
        """
        Initialize with existing sheets client.
        
        Args:
            sheets_client: Existing GoogleSheetsClient instance
        """
        self.sheets_client = sheets_client
    
    async def log_ie_trade(self, setup) -> int:
        """
        Log an IE Trade setup to Google Sheets.
        
        Args:
            setup: TradeSetup from entry_calculator
            
        Returns:
            Row number where trade was logged, 0 if failed
        """
        try:
            # Create TradeRecord from IE setup
            from ..storage.sheets_client import TradeRecord
            
            # Format note with IE trade info
            note_parts = [
                "IE trade",
                f"KZ:{setup.kill_zone}" if setup.kill_zone else "",
                f"Bias:{setup.daily_bias}" if setup.daily_bias else "",
                f"R:R={setup.rr_ratio_1:.1f}"
            ]
            note = " | ".join(filter(None, note_parts))
            
            # Create standard TradeRecord
            record = TradeRecord(
                trade_id=f"IE_{setup.symbol}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                date=setup.timestamp.strftime("%Y-%m-%d %H:%M"),
                coin=setup.symbol.replace("-USDT", ""),
                signal=setup.direction,
                leverage=10,  # Default leverage for IE trades
                entry=setup.entry_price,
                stoploss=setup.stop_loss,
                take_profit=setup.take_profit_1,
                status="OPEN",
                pnl_percent=0.0,
                note=note,
                message_id=0,
                grade="A_SNIPER",  # IE trades are sniper grade
                layers_passed="4/4",  # Full ICT methodology
                checklist_score="3/3"  # All checks passed
            )
            
            # Log using standard sheets client
            row = await self.sheets_client.log_trade(record)
            
            if row > 0:
                logger.info(f"🎯 IE Trade logged to sheet: {setup.symbol} {setup.direction} @ row {row}")
            
            return row
            
        except Exception as e:
            logger.error(f"Error logging IE trade to sheet: {e}")
            return 0
    
    async def update_ie_trade_status(
        self,
        symbol: str,
        status: str,  # "TP1", "TP2", "SL", "CLOSED"
        pnl_percent: float = 0.0
    ) -> bool:
        """
        Update an IE trade status in the sheet.
        
        Args:
            symbol: Trading pair (e.g., "BTC-USDT")
            status: New status
            pnl_percent: PnL percentage
            
        Returns:
            True if updated successfully
        """
        try:
            # Find the trade by symbol and "IE trade" note
            coin = symbol.replace("-USDT", "")
            
            # Use existing sheets client method if available
            if hasattr(self.sheets_client, 'update_trade_status'):
                return await self.sheets_client.update_trade_status(
                    coin=coin,
                    status=status,
                    pnl_percent=pnl_percent
                )
            
            logger.warning("update_trade_status not available in sheets client")
            return False
            
        except Exception as e:
            logger.error(f"Error updating IE trade status: {e}")
            return False


def create_ie_sheet_logger(sheets_client) -> Optional[IESheetLogger]:
    """
    Factory function to create IE sheet logger.
    
    Args:
        sheets_client: Existing GoogleSheetsClient instance
        
    Returns:
        IESheetLogger instance or None if sheets not available
    """
    if sheets_client is None:
        logger.warning("Google Sheets client not available - IE trades won't be logged")
        return None
    
    return IESheetLogger(sheets_client)
