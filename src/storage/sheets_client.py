"""
Google Sheets Integration for Trading Journal.
Logs all trades and updates results automatically.
"""

import os
import logging
import time
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import asyncio
from functools import wraps

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False

logger = logging.getLogger(__name__)


def retry_on_quota(max_retries: int = 5, base_delay: float = 10.0):
    """Decorator to retry on Google Sheets quota errors (429)."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    last_error = e
                    if "429" in error_str or "Quota exceeded" in error_str:
                        if attempt < max_retries - 1:
                            delay = base_delay * (2 ** attempt)  # Exponential backoff: 10s, 20s, 40s, 80s
                            logger.warning(f"⏳ Quota exceeded, retrying in {delay:.0f}s... (attempt {attempt + 1}/{max_retries})")
                            await asyncio.sleep(delay)
                        else:
                            logger.error(f"❌ Quota exceeded after {max_retries} retries - GIVING UP")
                            raise
                    else:
                        raise
            # Should not reach here
            raise last_error
        return wrapper
    return decorator


@dataclass
class TradeRecord:
    """Trade record for Google Sheets."""
    trade_id: str
    date: str
    coin: str
    signal: str  # LONG/SHORT
    leverage: int
    entry: float
    stoploss: float
    take_profit: float
    status: str = "OPEN"  # OPEN, TP1, TP2, TP3, SL, CLOSED
    pnl_percent: float = 0.0
    note: str = ""
    message_id: int = 0  # Telegram message ID for reply
    # Phase 1: Grade-based tracking
    grade: str = ""           # A_SNIPER, B_SCALP, C_WEAK
    layers_passed: str = ""   # 4/4, 3/4, 2/4
    checklist_score: str = "" # 3/3, 2/3, 1/3


class GoogleSheetsClient:
    """
    Google Sheets client for Trading Journal.
    
    Features:
    - Log new trades automatically
    - Update trade status (TP/SL hit)
    - Calculate winrate automatically
    - Format sheet beautifully
    """
    
    SCOPES = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    
    # Sheet structure - 18 columns (Phase 1: added Grade, Layers, Checklist)
    # A: No, B: Date, C: Coin, D: Signal, E: Leverage
    # F: Entry, G: Stoploss, H: Take Profit
    # I: Price Now, J: PnL%, K: Status, L: Close Time, M: Note
    # N: End Trade (checkbox), O: User Note
    # P: Grade (A/B/C), Q: Layers (4/4, 3/4), R: Checklist (3/3, 2/3)
    HEADERS = [
        "No", "Date", "Coin", "Signal", "Leverage", 
        "Entry", "SL", "TP", "Price Now", "PnL %", "Status", "Close Time", "Note", 
        "End Trade", "User Note", "Grade", "Layers", "Checklist"
    ]
    
    def __init__(
        self,
        credentials_path: str = "credentials.json",
        spreadsheet_id: Optional[str] = None,
        spreadsheet_name: str = "Trading Journal"
    ):
        self.credentials_path = credentials_path
        self.spreadsheet_id = spreadsheet_id or os.getenv("GOOGLE_SHEET_ID", "")
        self.spreadsheet_name = spreadsheet_name
        self.client: Optional[gspread.Client] = None
        self.sheet: Optional[gspread.Worksheet] = None
        self._connected = False
        
    async def connect(self) -> bool:
        """Connect to Google Sheets."""
        if not GSPREAD_AVAILABLE:
            logger.warning("⚠️ gspread not installed. Google Sheets disabled.")
            return False
            
        try:
            # Load credentials
            if not os.path.exists(self.credentials_path):
                logger.error(f"❌ Credentials file not found: {self.credentials_path}")
                return False
            
            creds = Credentials.from_service_account_file(
                self.credentials_path,
                scopes=self.SCOPES
            )
            
            self.client = gspread.authorize(creds)
            
            # Open or create spreadsheet
            if self.spreadsheet_id:
                try:
                    spreadsheet = self.client.open_by_key(self.spreadsheet_id)
                except gspread.SpreadsheetNotFound:
                    logger.error(f"❌ Spreadsheet not found: {self.spreadsheet_id}")
                    return False
            else:
                # Try to open by name or create new
                try:
                    spreadsheet = self.client.open(self.spreadsheet_name)
                except gspread.SpreadsheetNotFound:
                    spreadsheet = self.client.create(self.spreadsheet_name)
                    logger.info(f"📊 Created new spreadsheet: {self.spreadsheet_name}")
            
            # Get first worksheet (main sheet) instead of creating new one
            try:
                # Try to get by common names first
                for name in ["Trade History", "Trades", "Sheet1"]:
                    try:
                        self.sheet = spreadsheet.worksheet(name)
                        break
                    except gspread.WorksheetNotFound:
                        continue
                else:
                    # Fallback to first sheet
                    self.sheet = spreadsheet.get_worksheet(0)
            except Exception as e:
                logger.error(f"❌ Cannot get worksheet: {e}")
                return False
                
            self._connected = True
            logger.info(f"✅ Google Sheets connected: {spreadsheet.title}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Google Sheets connection failed: {e}")
            return False
    
    async def _setup_sheet(self):
        """Setup sheet with headers and formatting - Professional Trading Journal Style."""
        if not self.sheet:
            return
            
        try:
            # === BRANDING SECTION (Row 1-4) ===
            # Logo area
            self.sheet.update('B1', 'TRADING')
            self.sheet.update('B2', 'JOURNAL')
            
            # Winrate display - count wins (PnL > 0%) vs total closed trades
            self.sheet.update('D1:E1', [['', '']])  # Clear
            # Formula: Count cells with >0% divided by cells with % sign (closed trades)
            self.sheet.update('E2', '=IFERROR(TEXT(COUNTIF(I6:I1000,">0%")/SUMPRODUCT(--(LEN(I6:I1000)>0),--(ISNUMBER(SEARCH("%",I6:I1000)))),"0.0%"),"0.0%")')
            self.sheet.update('E3', 'Winrate')
            
            # Info banner
            self.sheet.update('F1', 'Trade thi sẽ có kèo thắng kèo thua! Luôn đảm bảo đi đều lệnh + Số tiền mỗi lệnh bằng nhau')
            self.sheet.update('F2', 'Các kèo chia sẻ free, nên để mn chơi đỡ tốn phí giao dịch nhất')
            
            # === HEADERS (Row 5) ===
            self.sheet.update('A5:R5', [self.HEADERS])
            
            # Freeze header row
            self.sheet.freeze(rows=5)
            
            # === STYLING VIA BATCH UPDATE ===
            spreadsheet_id = self.sheet.spreadsheet.id
            sheet_id = self.sheet.id
            
            requests = [
                # Apply Comic Sans MS, size 13, center alignment to ENTIRE SHEET (A1:Z1000)
                {
                    "repeatCell": {
                        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 1000, "startColumnIndex": 0, "endColumnIndex": 26},
                        "cell": {
                            "userEnteredFormat": {
                                "textFormat": {"fontFamily": "Comic Sans MS", "fontSize": 13},
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(textFormat,horizontalAlignment,verticalAlignment)"
                    }
                },
                # Header row - Dark background with Comic Sans MS (18 columns A-R)
                {
                    "repeatCell": {
                        "range": {"sheetId": sheet_id, "startRowIndex": 4, "endRowIndex": 5, "startColumnIndex": 0, "endColumnIndex": 18},
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": {"red": 0.15, "green": 0.15, "blue": 0.15},
                                "textFormat": {"fontFamily": "Comic Sans MS", "foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": True, "fontSize": 13},
                                "horizontalAlignment": "CENTER",
                                "verticalAlignment": "MIDDLE"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"
                    }
                },
                # Winrate cell - Large Comic Sans MS font
                {
                    "repeatCell": {
                        "range": {"sheetId": sheet_id, "startRowIndex": 1, "endRowIndex": 2, "startColumnIndex": 4, "endColumnIndex": 5},
                        "cell": {
                            "userEnteredFormat": {
                                "textFormat": {"fontFamily": "Comic Sans MS", "bold": True, "fontSize": 28, "foregroundColor": {"red": 0.2, "green": 0.6, "blue": 0.2}},
                                "horizontalAlignment": "CENTER"
                            }
                        },
                        "fields": "userEnteredFormat(textFormat,horizontalAlignment)"
                    }
                },
                # TRADING JOURNAL branding - Comic Sans MS
                {
                    "repeatCell": {
                        "range": {"sheetId": sheet_id, "startRowIndex": 0, "endRowIndex": 2, "startColumnIndex": 1, "endColumnIndex": 2},
                        "cell": {
                            "userEnteredFormat": {
                                "textFormat": {"fontFamily": "Comic Sans MS", "bold": True, "fontSize": 16, "foregroundColor": {"red": 0.1, "green": 0.1, "blue": 0.1}},
                                "horizontalAlignment": "CENTER"
                            }
                        },
                        "fields": "userEnteredFormat(textFormat,horizontalAlignment)"
                    }
                },
                # Column widths
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 0, "endIndex": 1}, "properties": {"pixelSize": 50}, "fields": "pixelSize"}},  # No
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 1, "endIndex": 2}, "properties": {"pixelSize": 110}, "fields": "pixelSize"}},  # Date
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 2, "endIndex": 3}, "properties": {"pixelSize": 100}, "fields": "pixelSize"}},  # Coin
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 3, "endIndex": 4}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},  # Signal
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 4, "endIndex": 5}, "properties": {"pixelSize": 90}, "fields": "pixelSize"}},  # Leverage
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 5, "endIndex": 6}, "properties": {"pixelSize": 100}, "fields": "pixelSize"}},  # Entry
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 6, "endIndex": 7}, "properties": {"pixelSize": 100}, "fields": "pixelSize"}},  # Stoploss
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 7, "endIndex": 8}, "properties": {"pixelSize": 100}, "fields": "pixelSize"}},  # TP
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 8, "endIndex": 9}, "properties": {"pixelSize": 100}, "fields": "pixelSize"}},  # Price Now
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 9, "endIndex": 10}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},  # PnL %
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 10, "endIndex": 11}, "properties": {"pixelSize": 70}, "fields": "pixelSize"}},  # Status
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 11, "endIndex": 12}, "properties": {"pixelSize": 130}, "fields": "pixelSize"}},  # Close Time
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 12, "endIndex": 13}, "properties": {"pixelSize": 120}, "fields": "pixelSize"}}, # Note
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 13, "endIndex": 14}, "properties": {"pixelSize": 90}, "fields": "pixelSize"}},  # End Trade
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 14, "endIndex": 15}, "properties": {"pixelSize": 150}, "fields": "pixelSize"}}, # User Note
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 15, "endIndex": 16}, "properties": {"pixelSize": 90}, "fields": "pixelSize"}},  # Grade
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 16, "endIndex": 17}, "properties": {"pixelSize": 60}, "fields": "pixelSize"}},  # Layers
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 17, "endIndex": 18}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},  # Checklist
                # Row height for header
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "ROWS", "startIndex": 4, "endIndex": 5}, "properties": {"pixelSize": 35}, "fields": "pixelSize"}},
            ]
            
            # Batch update
            self.sheet.spreadsheet.batch_update({"requests": requests})
            
            logger.info("📊 Sheet setup completed with professional styling")
            
        except Exception as e:
            logger.error(f"Error setting up sheet: {e}")
    
    @retry_on_quota(max_retries=3, base_delay=5.0)
    async def log_trade(self, trade: TradeRecord) -> int:
        """
        Log a new trade to the sheet.
        
        Returns:
            Row number where trade was logged
        """
        if not self._connected or not self.sheet:
            logger.warning("Google Sheets not connected")
            return 0
            
        try:
            # ========== VALIDATE SL/TP DIRECTION ==========
            # CRITICAL: Ensure SL/TP are correct for direction
            # LONG: SL < Entry < TP (SL below, TP above)
            # SHORT: SL > Entry > TP (SL above, TP below)
            entry = trade.entry
            sl = trade.stoploss
            tp = trade.take_profit
            
            if trade.signal == "LONG":
                # For LONG: SL must be below entry, TP must be above entry
                if sl >= entry:
                    logger.warning(f"⚠️ {trade.coin} LONG: SL ({sl}) >= Entry ({entry}) - SWAPPING SL/TP")
                    trade.stoploss, trade.take_profit = tp, sl
                elif tp <= entry:
                    logger.warning(f"⚠️ {trade.coin} LONG: TP ({tp}) <= Entry ({entry}) - SWAPPING SL/TP")
                    trade.stoploss, trade.take_profit = tp, sl
            else:  # SHORT
                # For SHORT: SL must be above entry, TP must be below entry
                if sl <= entry:
                    logger.warning(f"⚠️ {trade.coin} SHORT: SL ({sl}) <= Entry ({entry}) - SWAPPING SL/TP")
                    trade.stoploss, trade.take_profit = tp, sl
                elif tp >= entry:
                    logger.warning(f"⚠️ {trade.coin} SHORT: TP ({tp}) >= Entry ({entry}) - SWAPPING SL/TP")
                    trade.stoploss, trade.take_profit = tp, sl
            
            # Log validated values
            logger.info(f"📊 Logging {trade.coin} {trade.signal}: Entry={trade.entry}, SL={trade.stoploss}, TP={trade.take_profit}")
            
            # Find next empty row
            all_values = self.sheet.get_all_values()
            
            # Simply find the next row after all data
            next_row = len(all_values) + 1
                
            if next_row < 6:
                next_row = 6  # Start after headers
            
            # Trade number
            trade_no = next_row - 5
            
            # Format numbers based on value - consistent formatting
            def format_number(num):
                if num == 0:
                    return ""
                # Use 6 decimals if number < 0.01 (very small coins like PEPE)
                if num < 0.01:
                    return f"{num:.8f}"
                elif num < 1:
                    return f"{num:.6f}"
                elif num < 10:
                    return f"{num:.4f}"
                elif num < 100:
                    return f"{num:.3f}"
                elif num < 1000:
                    return f"{num:.2f}"
                else:
                    return f"{num:.1f}"
            
            # Prepare row data with formatted numbers
            # Columns: No, Date, Coin, Signal, Leverage, Entry, SL, TP, Price Now, PnL%, Status, Close Time, Note, End Trade, User Note, Grade, Layers, Checklist
            row_data = [
                trade_no,
                trade.date,
                trade.coin,
                trade.signal,
                trade.leverage,
                format_number(trade.entry),
                format_number(trade.stoploss),
                format_number(trade.take_profit),
                format_number(trade.entry),  # Price Now - starts at entry
                "0.0%",  # PnL% - starts at 0
                "OPEN",  # Status
                "",  # Close Time - empty until closed
                trade.note,  # Note - strategy reason
                False,  # End Trade - checkbox unchecked (boolean for checkbox)
                "",  # User Note - empty for user to fill
                trade.grade,  # Grade (A_SNIPER, B_SCALP, etc)
                trade.layers_passed,  # Layers (4/4, 3/4, etc)
                trade.checklist_score  # Checklist (3/3, 2/3, etc)
            ]
            
            # Insert row (18 columns A-R)
            cell_range = f'A{next_row}:R{next_row}'
            self.sheet.update(cell_range, [row_data])
            
            # Add checkbox to End Trade column (N)
            self._add_checkbox(next_row)
            
            # Format number cells with blue text for Entry/SL/TP
            self._format_number_cells(next_row)
            
            # Apply formatting based on signal
            self._format_signal_cell(next_row, trade.signal)
            
            logger.info(f"📊 Trade logged: {trade.coin} {trade.signal} @ row {next_row}")
            
            # Update stats table (force update when new trade)
            await self._update_total_row(force=True)
            
            return next_row
            
        except Exception as e:
            # Log with full context for debugging
            logger.error(f"❌ FAILED to log {trade.coin} {trade.signal} to sheet: {e}")
            logger.error(f"   Entry={trade.entry}, SL={trade.stoploss}, TP={trade.take_profit}")
            
            # Re-raise to alert caller that logging failed
            # This prevents false tracking when sheet log fails
            raise
    
    def _add_checkbox(self, row: int):
        """Add checkbox data validation to End Trade cell."""
        try:
            sheet_id = self.sheet.id
            
            # First set the cell value to FALSE (boolean)
            self.sheet.update_acell(f'N{row}', False)
            
            # Then add data validation for checkbox
            requests = [{
                "setDataValidation": {
                    "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row, 
                              "startColumnIndex": 13, "endColumnIndex": 14},
                    "rule": {
                        "condition": {"type": "BOOLEAN"},
                        "showCustomUi": True
                    }
                }
            }]
            self.sheet.spreadsheet.batch_update({"requests": requests})
            logger.debug(f"✅ Checkbox added to row {row}")
        except Exception as e:
            logger.error(f"❌ Checkbox error at row {row}: {e}")
    
    async def has_open_trade(self, coin: str) -> bool:
        """
        Check if coin already has an open trade that user hasn't ended.
        
        Returns:
            True if there's an open trade for this coin (should skip new trade)
            False if no open trade or user has ended previous trade
        """
        if not self._connected or not self.sheet:
            return False
            
        try:
            all_data = self.sheet.get_all_values()
            
            for row in all_data[5:]:  # Skip headers
                if len(row) >= 14 and row[2]:  # Has coin
                    row_coin = row[2].upper()
                    if row_coin == coin.upper():
                        status = row[10] if len(row) > 10 else ""
                        end_trade = row[13] if len(row) > 13 else ""
                        
                        # Trade is still open if:
                        # 1. Status is NOT TP/SL/CLOSED, AND
                        # 2. End Trade is NOT checked
                        is_closed = status.upper() in ["TP", "SL", "CLOSED"] or end_trade.upper() in ["TRUE", "✓", "✔", "1"]
                        
                        if not is_closed:
                            logger.info(f"⚠️ {coin} already has open trade - skipping duplicate")
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking open trade: {e}")
            return False
    
    def _update_cell_raw(self, cell: str, value: str):
        """Update cell as RAW text to avoid locale decimal separator issues."""
        try:
            self.sheet.update(values=[[value]], range_name=cell, value_input_option='RAW')
        except Exception as e:
            # Fallback to regular update
            self.sheet.update_acell(cell, value)
    
    def _format_number_cells(self, row: int):
        """Format number cells (Entry, SL, TP, Price Now) with blue Comic Sans MS."""
        try:
            sheet_id = self.sheet.id
            
            # Blue color for Entry, SL, TP (F, G, H) and Price Now (I)
            requests = [{
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row, 
                              "startColumnIndex": 5, "endColumnIndex": 9},
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"fontFamily": "Comic Sans MS", "foregroundColor": {"red": 0.0, "green": 0.4, "blue": 0.8}, "fontSize": 13},
                            "horizontalAlignment": "CENTER"
                        }
                    },
                    "fields": "userEnteredFormat(textFormat,horizontalAlignment)"
                }
            }]
            
            self.sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.debug(f"Format error: {e}")
    
    def _format_signal_cell(self, row: int, signal: str):
        """Format signal cell with color badge style."""
        try:
            sheet_id = self.sheet.id
            
            if signal == "SHORT":
                # Red badge for SHORT (like in the image)
                bg_color = {"red": 0.8, "green": 0.0, "blue": 0.0}
            else:
                # Green badge for LONG
                bg_color = {"red": 0.0, "green": 0.6, "blue": 0.2}
            
            requests = [{
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row, "startColumnIndex": 3, "endColumnIndex": 4},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": bg_color,
                            "textFormat": {"fontFamily": "Comic Sans MS", "foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": False, "fontSize": 13},
                            "horizontalAlignment": "CENTER",
                            "verticalAlignment": "MIDDLE"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment,verticalAlignment)"
                }
            }]
            
            self.sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.debug(f"Format error: {e}")
    
    async def update_trade_result(
        self,
        row: int,
        pnl_percent: float,
        status: str = "",
        close_time: str = ""
    ) -> bool:
        """
        Update trade result when TP/SL hit.
        
        Args:
            row: Row number of the trade
            pnl_percent: Profit/Loss percentage
            status: Status (TP or SL)
            close_time: Time when trade closed (format: TP - HH:MM or SL - HH:MM)
        """
        if not self._connected or not self.sheet:
            return False
            
        try:
            # Update PnL column (J) with 2 decimal places
            pnl_str = f'{pnl_percent:.2f}%'
            self.sheet.update_acell(f'J{row}', pnl_str)
            
            # Update Status column (K)
            if status:
                self.sheet.update_acell(f'K{row}', status)
                self._format_status_cell(row, status)
            
            # Update Close Time column (L)
            if close_time:
                self.sheet.update_acell(f'L{row}', close_time)
            
            # Format PnL cell based on result
            self._format_pnl_cell(row, pnl_percent)
            
            logger.info(f"📊 Trade updated: row {row}, PnL={pnl_percent:.2f}%, Status={status}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating trade: {e}")
            return False
    
    def _format_status_cell(self, row: int, status: str):
        """Format Status cell with color - green for TP, red for SL."""
        try:
            sheet_id = self.sheet.id
            
            if status == "TP":
                bg_color = {"red": 0.13, "green": 0.55, "blue": 0.13}  # Green
            elif status == "SL":
                bg_color = {"red": 0.8, "green": 0.1, "blue": 0.1}  # Red
            else:
                bg_color = {"red": 0.9, "green": 0.9, "blue": 0.9}  # Light gray for OPEN
            
            requests = [{
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row, "startColumnIndex": 10, "endColumnIndex": 11},
                    "cell": {
                        "userEnteredFormat": {
                            "backgroundColor": bg_color,
                            "textFormat": {"fontFamily": "Comic Sans MS", "foregroundColor": {"red": 1, "green": 1, "blue": 1}, "bold": False, "fontSize": 13},
                            "horizontalAlignment": "CENTER"
                        }
                    },
                    "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
                }
            }]
            
            self.sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.debug(f"Format error: {e}")
    
    def _format_pnl_cell(self, row: int, pnl: float):
        """Format PnL cell (column J) with color - green for profit, red for loss."""
        try:
            sheet_id = self.sheet.id
            
            if pnl > 0:
                # Green text for profit
                text_color = {"red": 0.13, "green": 0.55, "blue": 0.13}
            elif pnl < 0:
                # Red text for loss
                text_color = {"red": 0.8, "green": 0.1, "blue": 0.1}
            else:
                # Gray for breakeven
                text_color = {"red": 0.5, "green": 0.5, "blue": 0.5}
            
            requests = [{
                "repeatCell": {
                    "range": {"sheetId": sheet_id, "startRowIndex": row-1, "endRowIndex": row, "startColumnIndex": 9, "endColumnIndex": 10},
                    "cell": {
                        "userEnteredFormat": {
                            "textFormat": {"fontFamily": "Comic Sans MS", "foregroundColor": text_color, "bold": False, "fontSize": 13},
                            "horizontalAlignment": "CENTER"
                        }
                    },
                    "fields": "userEnteredFormat(textFormat,horizontalAlignment)"
                }
            }]
            
            self.sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.debug(f"Format error: {e}")
    
    async def get_trade_by_row(self, row: int) -> Optional[Dict]:
        """Get trade data by row number."""
        if not self._connected or not self.sheet:
            return None
            
        try:
            row_data = self.sheet.row_values(row)
            if len(row_data) >= 8:
                return {
                    "no": row_data[0],
                    "date": row_data[1],
                    "coin": row_data[2],
                    "signal": row_data[3],
                    "leverage": row_data[4],
                    "entry": row_data[5],
                    "stoploss": row_data[6],
                    "take_profit": row_data[7],
                    "pnl": row_data[8] if len(row_data) > 8 else "",
                    "note": row_data[9] if len(row_data) > 9 else ""
                }
            return None
        except Exception as e:
            logger.error(f"Error getting trade: {e}")
            return None
    
    async def find_open_trades(self) -> List[Tuple[int, Dict]]:
        """Find all open trades (Status = OPEN and End Trade unchecked)."""
        if not self._connected or not self.sheet:
            return []
            
        try:
            all_data = self.sheet.get_all_values()
            open_trades = []
            
            for i, row in enumerate(all_data[5:], start=6):  # Skip headers (row 5)
                if len(row) >= 14 and row[2]:  # Has coin and enough columns
                    status = row[10] if len(row) > 10 else ""  # Column K = Status
                    end_trade = row[13] if len(row) > 13 else ""  # Column N = End Trade
                    
                    # Skip if already closed (TP/SL/CLOSED status)
                    if status.upper() in ["TP", "SL", "CLOSED"]:
                        continue
                    
                    # Include if:
                    # 1. Status is OPEN (normal monitoring)
                    # 2. End Trade is checked (need to close manually)
                    is_end_trade_checked = end_trade.upper() in ["TRUE", "✓", "✔", "1"]
                    
                    if status.upper() == "OPEN" or is_end_trade_checked:
                        # Parse entry/sl/tp (handle comma as decimal separator)
                        try:
                            entry = float(row[5].replace(',', '.')) if row[5] else 0
                            stoploss = float(row[6].replace(',', '.')) if row[6] else 0
                            take_profit = float(row[7].replace(',', '.')) if row[7] else 0
                            leverage = int(row[4]) if row[4] else 15
                            
                            open_trades.append((i, {
                                "coin": row[2],
                                "signal": row[3],
                                "leverage": leverage,
                                "entry": entry,
                                "stoploss": stoploss,
                                "take_profit": take_profit,
                                "end_trade_checked": is_end_trade_checked
                            }))
                        except (ValueError, IndexError) as e:
                            logger.debug(f"Skip row {i}: {e}")
                            continue
            
            return open_trades
            
        except Exception as e:
            logger.error(f"Error finding open trades: {e}")
            return []
    
    async def check_and_update_trades(self, rest_client) -> int:
        """
        Check all open trades and update their status based on current price.
        
        Args:
            rest_client: BingXRestClient instance to get current prices
            
        Returns:
            Number of trades updated
        """
        if not self._connected or not self.sheet:
            return 0
            
        try:
            open_trades = await self.find_open_trades()
            if not open_trades:
                return 0
            
            updated_count = 0
            
            for row, trade in open_trades:
                symbol = f"{trade['coin']}-USDT"
                
                # Check if End Trade checkbox is checked
                if trade.get('end_trade_checked', False):
                    try:
                        # User manually ended trade - close it now
                        # Get current price
                        ticker = await rest_client.get_futures_ticker(symbol)
                        if ticker and 'lastPrice' in ticker:
                            current_price = float(ticker['lastPrice'])
                            entry = trade['entry']
                            leverage = trade['leverage']
                            direction = trade['signal']
                            
                            # Format price
                            def format_price(p):
                                if p < 1:
                                    return f"{p:.6f}"
                                elif p < 100:
                                    return f"{p:.3f}"
                                else:
                                    return f"{p:.2f}"
                            
                            # Calculate final PnL
                            funding_fee_percent = 0.01 * leverage
                            if direction == "LONG":
                                pnl = ((current_price - entry) / entry) * 100 * leverage - funding_fee_percent
                            else:
                                pnl = ((entry - current_price) / entry) * 100 * leverage - funding_fee_percent
                            
                            # Close trade manually
                            time_str = datetime.now().strftime('%H:%M')
                            close_time = f"CLOSED - {time_str} - {format_price(current_price)}"
                            await self.update_trade_result(row, pnl, "CLOSED", close_time)
                            self._update_cell_raw(f'I{row}', format_price(current_price))
                            logger.info(f"🛑 {symbol} MANUALLY CLOSED by user: {pnl:.1f}%")
                            updated_count += 1
                            continue
                        else:
                            logger.warning(f"⚠️ Cannot get price for {symbol} to close manually")
                            continue
                    except Exception as e:
                        logger.error(f"Error closing {symbol} manually: {e}")
                        continue
                
                # Get current price
                try:
                    ticker = await rest_client.get_futures_ticker(symbol)
                    if not ticker or 'lastPrice' not in ticker:
                        continue
                    
                    current_price = float(ticker['lastPrice'])
                    entry = trade['entry']
                    sl = trade['stoploss']
                    tp = trade['take_profit']
                    leverage = trade['leverage']
                    direction = trade['signal']
                    
                    # ========== VALIDATE SL/TP DIRECTION ==========
                    # CRITICAL: Ensure data in sheet is correct for direction
                    # LONG: SL < Entry < TP
                    # SHORT: SL > Entry > TP
                    is_valid = True
                    if direction == "LONG":
                        if sl >= entry or tp <= entry:
                            logger.warning(f"⚠️ Row {row} {symbol} LONG: Invalid SL/TP (SL={sl}, Entry={entry}, TP={tp})")
                            # Try to fix by swapping
                            if sl > entry and tp < entry:
                                sl, tp = tp, sl
                                logger.info(f"🔧 Fixed by swapping: SL={sl}, TP={tp}")
                            else:
                                is_valid = False
                    else:  # SHORT
                        if sl <= entry or tp >= entry:
                            logger.warning(f"⚠️ Row {row} {symbol} SHORT: Invalid SL/TP (SL={sl}, Entry={entry}, TP={tp})")
                            # Try to fix by swapping
                            if sl < entry and tp > entry:
                                sl, tp = tp, sl
                                logger.info(f"🔧 Fixed by swapping: SL={sl}, TP={tp}")
                            else:
                                is_valid = False
                    
                    if not is_valid:
                        logger.error(f"❌ Row {row} {symbol}: Cannot fix SL/TP - skipping")
                        continue
                    
                    # Format price for display - consistent formatting
                    def format_price(p):
                        if p < 0.01:
                            return f"{p:.8f}"
                        elif p < 1:
                            return f"{p:.6f}"
                        elif p < 10:
                            return f"{p:.4f}"
                        elif p < 100:
                            return f"{p:.3f}"
                        elif p < 1000:
                            return f"{p:.2f}"
                        else:
                            return f"{p:.1f}"
                    
                    # Check if SL or TP hit
                    hit_sl = False
                    hit_tp = False
                    pnl = 0.0
                    status = "OPEN"
                    
                    # Calculate funding fee (assume 0.01% per 8h, estimate 0.03% per day)
                    # For simplicity, deduct 0.01% from PnL for each position
                    funding_fee_percent = 0.01 * leverage  # Funding impact based on leverage
                    
                    if direction == "LONG":
                        if current_price <= sl:
                            hit_sl = True
                            pnl = ((sl - entry) / entry) * 100 * leverage - funding_fee_percent
                            status = "SL"
                        elif current_price >= tp:
                            hit_tp = True
                            pnl = ((tp - entry) / entry) * 100 * leverage - funding_fee_percent
                            status = "TP"
                        else:
                            # Update current PnL
                            pnl = ((current_price - entry) / entry) * 100 * leverage - funding_fee_percent
                    else:  # SHORT
                        if current_price >= sl:
                            hit_sl = True
                            # SHORT hit SL = LOSS (price went UP against us)
                            pnl = ((entry - sl) / entry) * 100 * leverage - funding_fee_percent
                            status = "SL"
                        elif current_price <= tp:
                            hit_tp = True
                            # SHORT hit TP = WIN (price went DOWN for us)
                            pnl = ((entry - tp) / entry) * 100 * leverage - funding_fee_percent
                            status = "TP"
                        else:
                            # Update current PnL
                            pnl = ((entry - current_price) / entry) * 100 * leverage - funding_fee_percent
                    
                    # Update sheet with new column structure
                    if hit_sl or hit_tp:
                        # Get current timestamp and format close_time: "TP - 13:52 - 0.9"
                        time_str = datetime.now().strftime('%H:%M')
                        price_hit = sl if hit_sl else tp
                        close_time = f"{status} - {time_str} - {format_price(price_hit)}"
                        await self.update_trade_result(row, pnl, status, close_time)
                        # Update Price Now (column I)
                        self._update_cell_raw(f'I{row}', format_price(current_price))
                        # Mark End Trade as TRUE when TP/SL hits
                        self.sheet.update_acell(f'N{row}', 'TRUE')
                        logger.info(f"🔔 {symbol} {status}: {pnl:.1f}%")
                        updated_count += 1
                        
                        # Store trade close info for Telegram notification
                        if not hasattr(self, '_closed_trades'):
                            self._closed_trades = []
                        self._closed_trades.append({
                            'symbol': symbol,
                            'status': status,
                            'pnl': pnl,
                            'close_time': close_time,
                            'direction': direction
                        })
                    else:
                        # Update Price Now (column I) and PnL (column J) for open trades
                        self._update_cell_raw(f'I{row}', format_price(current_price))
                        self.sheet.update_acell(f'J{row}', f'{pnl:.1f}%')
                        self._format_pnl_cell(row, pnl)
                        logger.debug(f"📊 {symbol} @ {current_price:.4f}: {pnl:.1f}%")
                        updated_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error checking {symbol}: {e}")
                    continue
            
            # Update winrate in cell E2
            if updated_count > 0:
                try:
                    stats = await self.get_stats()
                    self.sheet.update_acell('E2', f"{stats['winrate']:.1f}%")
                    self.sheet.format('E2', {
                        "textFormat": {"bold": True, "fontSize": 13, "foregroundColor": {"red": 0, "green": 0.4, "blue": 0.8}},
                        "horizontalAlignment": "CENTER"
                    })
                except Exception as e:
                    logger.debug(f"Error updating winrate: {e}")
            
            # Update stats table after each scan
            await self._update_total_row()
            
            return updated_count
            
        except Exception as e:
            logger.error(f"Error in check_and_update_trades: {e}")
            return 0
    
    async def get_stats(self) -> Dict:
        """Get trading statistics from closed trades."""
        if not self._connected or not self.sheet:
            return {"total": 0, "wins": 0, "losses": 0, "winrate": 0}
            
        try:
            all_data = self.sheet.get_all_values()
            
            total = 0
            wins = 0
            losses = 0
            
            for row in all_data[5:]:
                # Only count closed trades (Status = TP or SL in column K, index 10)
                if len(row) > 10 and row[10] in ["TP", "SL"]:
                    pnl_str = row[9].replace('%', '').strip() if len(row) > 9 else ""  # Column J
                    try:
                        pnl = float(pnl_str)
                        total += 1
                        if pnl > 0:
                            wins += 1
                        elif pnl < 0:
                            losses += 1
                    except ValueError:
                        pass
            
            winrate = (wins / total * 100) if total > 0 else 0
            
            return {
                "total": total,
                "wins": wins,
                "losses": losses,
                "winrate": round(winrate, 2)
            }
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            return {"total": 0, "wins": 0, "losses": 0, "winrate": 0}
    
    async def end_all_trades(self) -> Dict:
        """
        End all open trades (set End Trade = TRUE for all).
        
        Returns:
            Dict with count of trades ended and total PnL
        """
        if not self._connected or not self.sheet:
            return {"count": 0, "total_pnl": 0.0, "error": "Not connected"}
            
        try:
            all_data = self.sheet.get_all_values()
            trades_ended = 0
            total_pnl = 0.0
            rows_to_update = []
            
            for i, row in enumerate(all_data[5:], start=6):  # Skip headers
                if len(row) >= 14 and row[2]:  # Has coin
                    status = row[10] if len(row) > 10 else ""
                    end_trade = row[13] if len(row) > 13 else ""
                    
                    # Skip already closed trades
                    is_closed = end_trade.upper() in ["TRUE", "✓", "✔", "1"]
                    if is_closed:
                        continue
                    
                    # Mark as ended
                    rows_to_update.append(i)
                    
                    # Get PnL if available
                    pnl_str = row[9] if len(row) > 9 else "0"
                    try:
                        pnl = float(pnl_str.replace('%', '').replace(',', '.'))
                        total_pnl += pnl
                    except ValueError:
                        pass
                    
                    trades_ended += 1
            
            # Batch update all End Trade checkboxes
            if rows_to_update:
                batch_data = []
                for row_num in rows_to_update:
                    batch_data.append({
                        'range': f'N{row_num}',
                        'values': [[True]]
                    })
                self.sheet.batch_update(batch_data)
                
                # Also set Status to CLOSED if not already TP/SL
                batch_status = []
                for row_num in rows_to_update:
                    row_data = all_data[row_num - 1]  # 0-indexed
                    status = row_data[10] if len(row_data) > 10 else ""
                    if status.upper() not in ["TP", "SL"]:
                        batch_status.append({
                            'range': f'K{row_num}',
                            'values': [["CLOSED"]]
                        })
                if batch_status:
                    self.sheet.batch_update(batch_status)
            
            logger.info(f"✅ Ended {trades_ended} trades, Total PnL: {total_pnl:.2f}%")
            
            # Update total row (force update)
            await self._update_total_row(force=True)
            
            return {
                "count": trades_ended,
                "total_pnl": round(total_pnl, 2)
            }
            
        except Exception as e:
            logger.error(f"Error ending all trades: {e}")
            return {"count": 0, "total_pnl": 0.0, "error": str(e)}
    
    async def _update_total_row(self, force: bool = False):
        """Update statistics table at T6:X12 (5 columns x 7 rows including TOTAL).
        
        Args:
            force: If True, bypass rate limiting
        """
        if not self._connected or not self.sheet:
            return
        
        # Rate limit: only update every 5 minutes unless forced
        now = datetime.now()
        if not force:
            if hasattr(self, '_last_stats_update'):
                elapsed = (now - self._last_stats_update).total_seconds()
                if elapsed < 300:  # 5 minutes
                    return
        self._last_stats_update = now
            
        try:
            all_data = self.sheet.get_all_values()
            
            # Initialize stats by strategy
            strategies = {
                "EMA_PULLBACK": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                "BB_BOUNCE": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                "IE": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                "LIQ_SWEEP": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
                "SFP": {"trades": 0, "wins": 0, "losses": 0, "pnl": 0.0},
            }
            
            # Parse all trades
            for row in all_data[5:]:  # Skip headers (row 1-5)
                if len(row) >= 14 and row[2]:  # Has coin
                    # Skip if it's a header or empty
                    coin = row[2].strip()
                    if not coin or coin.upper() in ["COIN", "TOTAL"]:
                        continue
                    
                    # Get strategy from Note column (M = index 12)
                    note = row[12] if len(row) > 12 else ""
                    
                    # Determine strategy
                    strategy = None
                    note_upper = note.upper()
                    if "IE" in note_upper or "IE TRADE" in note_upper:
                        strategy = "IE"
                    elif "EMA" in note_upper or "PULLBACK" in note_upper:
                        strategy = "EMA_PULLBACK"
                    elif "BB" in note_upper or "BOUNCE" in note_upper:
                        strategy = "BB_BOUNCE"
                    elif "LIQ" in note_upper or "SWEEP" in note_upper:
                        strategy = "LIQ_SWEEP"
                    elif "SFP" in note_upper or "BREAKER" in note_upper:
                        strategy = "SFP"
                    else:
                        # Default to EMA_PULLBACK if can't determine
                        strategy = "EMA_PULLBACK"
                    
                    # Get PnL
                    pnl_str = row[9] if len(row) > 9 else "0"
                    try:
                        pnl = float(pnl_str.replace('%', '').replace(',', '.'))
                        strategies[strategy]["trades"] += 1
                        strategies[strategy]["pnl"] += pnl
                        if pnl > 0:
                            strategies[strategy]["wins"] += 1
                        elif pnl < 0:
                            strategies[strategy]["losses"] += 1
                    except ValueError:
                        pass
            
            # Build stats table (5 columns: Strategy, Total trade, Winrate, PnL(%), Pnl(usd))
            # Header row
            table_data = [
                ["", "Total trade", "Winrate", "PnL(%)", "Pnl(usd)"],  # Header
            ]
            
            # Strategy rows
            strategy_order = ["EMA_PULLBACK", "BB_BOUNCE", "IE", "LIQ_SWEEP", "SFP"]
            for strat in strategy_order:
                stats = strategies[strat]
                trades = stats["trades"]
                wins = stats["wins"]
                pnl_pct = stats["pnl"]
                
                # Calculate winrate
                winrate = (wins / trades * 100) if trades > 0 else 0
                
                # Calculate USD (1 USD per trade, no leverage in calculation)
                pnl_usd = pnl_pct / 100  # Convert % to USD (1 USD base)
                
                table_data.append([
                    strat,
                    str(trades) if trades > 0 else "",
                    f"{winrate:.1f}%" if trades > 0 else "",
                    f"{pnl_pct:.2f}%" if trades > 0 else "",
                    f"${pnl_usd:.2f}" if trades > 0 else ""
                ])
            
            # Calculate grand totals for TOTAL row
            total_trades = sum(s["trades"] for s in strategies.values())
            total_wins = sum(s["wins"] for s in strategies.values())
            total_pnl = sum(s["pnl"] for s in strategies.values())
            total_winrate = (total_wins / total_trades * 100) if total_trades > 0 else 0
            total_pnl_usd = total_pnl / 100  # $1 per trade base
            
            # Add TOTAL row
            table_data.append([
                "TOTAL",
                str(total_trades),
                f"{total_winrate:.1f}%",
                f"{total_pnl:.2f}%",
                f"${total_pnl_usd:.2f}"
            ])
            
            # Update table at T6:X12 (7 rows: header + 5 strategies + TOTAL)
            self.sheet.update('T6:X12', table_data)
            
            # Format the stats table
            self._format_stats_table()
            
            logger.info(f"📊 Stats updated: {total_trades} trades, PnL: {total_pnl:.2f}%, WR: {total_winrate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error updating stats table: {e}")
    
    def _format_stats_table(self):
        """Format the statistics table at T6:X12 (including TOTAL row)."""
        try:
            sheet_id = self.sheet.id
            
            # T=19 (0-indexed), X=23
            requests = [
                # Header row bold with background
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 5,  # Row 6 (0-indexed)
                            "endRowIndex": 6,
                            "startColumnIndex": 19,  # Column T
                            "endColumnIndex": 24   # Column X
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": {"red": 0.8, "green": 0.8, "blue": 0.8},
                                "textFormat": {"bold": True},
                                "horizontalAlignment": "CENTER"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
                    }
                },
                # Strategy names column bold
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 6,  # Row 7-11
                            "endRowIndex": 11,
                            "startColumnIndex": 19,  # Column T
                            "endColumnIndex": 20
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "textFormat": {"bold": True}
                            }
                        },
                        "fields": "userEnteredFormat(textFormat)"
                    }
                },
                # TOTAL row with bold and background (Row 12)
                {
                    "repeatCell": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 11,  # Row 12 (0-indexed)
                            "endRowIndex": 12,
                            "startColumnIndex": 19,  # Column T
                            "endColumnIndex": 24   # Column X
                        },
                        "cell": {
                            "userEnteredFormat": {
                                "backgroundColor": {"red": 0.9, "green": 0.95, "blue": 0.9},
                                "textFormat": {"bold": True},
                                "horizontalAlignment": "CENTER"
                            }
                        },
                        "fields": "userEnteredFormat(backgroundColor,textFormat,horizontalAlignment)"
                    }
                },
                # Add borders (now includes row 12)
                {
                    "updateBorders": {
                        "range": {
                            "sheetId": sheet_id,
                            "startRowIndex": 5,
                            "endRowIndex": 12,  # Changed from 11 to 12
                            "startColumnIndex": 19,
                            "endColumnIndex": 24
                        },
                        "top": {"style": "SOLID", "width": 1},
                        "bottom": {"style": "SOLID", "width": 1},
                        "left": {"style": "SOLID", "width": 1},
                        "right": {"style": "SOLID", "width": 1},
                        "innerHorizontal": {"style": "SOLID", "width": 1},
                        "innerVertical": {"style": "SOLID", "width": 1}
                    }
                },
                # Set column widths
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 19, "endIndex": 20}, "properties": {"pixelSize": 120}, "fields": "pixelSize"}},
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 20, "endIndex": 21}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 21, "endIndex": 22}, "properties": {"pixelSize": 70}, "fields": "pixelSize"}},
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 22, "endIndex": 23}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},
                {"updateDimensionProperties": {"range": {"sheetId": sheet_id, "dimension": "COLUMNS", "startIndex": 23, "endIndex": 24}, "properties": {"pixelSize": 80}, "fields": "pixelSize"}},
            ]
            
            self.sheet.spreadsheet.batch_update({"requests": requests})
        except Exception as e:
            logger.debug(f"Format stats table error: {e}")
    
    def get_closed_trades(self) -> List[Dict]:
        """
        Get list of trades that were auto-closed since last check.
        Used for sending Telegram notifications.
        
        Returns:
            List of closed trade dicts, then clears the list
        """
        if not hasattr(self, '_closed_trades'):
            self._closed_trades = []
        
        closed = self._closed_trades.copy()
        self._closed_trades = []  # Clear after getting
        return closed
    
    async def fix_all_false_checkboxes(self):
        """
        Fix all rows that have text 'FALSE' in End Trade column (N).
        Convert them to proper checkbox with FALSE boolean value.
        """
        if not self._connected or not self.sheet:
            logger.warning("Sheet not connected")
            return 0
            
        try:
            all_data = self.sheet.get_all_values()
            sheet_id = self.sheet.id
            fixed_count = 0
            rows_to_fix = []
            
            # First, find all rows that need fixing
            for i, row in enumerate(all_data[5:], start=6):  # Skip headers (row 5)
                if len(row) >= 14:
                    end_trade_value = row[13] if len(row) > 13 else ""  # Column N
                    
                    # Check if it's text "FALSE" instead of checkbox
                    if end_trade_value == "FALSE" or end_trade_value == "false":
                        rows_to_fix.append(i)
            
            if not rows_to_fix:
                logger.info("✅ No FALSE text found, all checkboxes OK")
                return 0
            
            logger.info(f"🔧 Found {len(rows_to_fix)} rows to fix...")
            
            # Fix each row with delay to avoid quota
            for idx, row_num in enumerate(rows_to_fix, 1):
                try:
                    # Add checkbox validation
                    requests = [{
                        "setDataValidation": {
                            "range": {
                                "sheetId": sheet_id,
                                "startRowIndex": row_num-1,
                                "endRowIndex": row_num,
                                "startColumnIndex": 13,
                                "endColumnIndex": 14
                            },
                            "rule": {
                                "condition": {"type": "BOOLEAN"},
                                "showCustomUi": True
                            }
                        }
                    }]
                    
                    self.sheet.spreadsheet.batch_update({"requests": requests})
                    
                    # Update cell to FALSE boolean
                    self.sheet.update_acell(f'N{row_num}', False)
                    
                    fixed_count += 1
                    logger.info(f"  ✅ Fixed row {row_num} ({idx}/{len(rows_to_fix)})")
                    
                    # Delay to avoid quota (every 5 rows)
                    if idx % 5 == 0:
                        logger.info(f"  ⏳ Pausing 2s to avoid quota...")
                        await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"  ❌ Failed row {row_num}: {e}")
                    # Continue with next row even if this fails
                    continue
            
            logger.info(f"✅ Fixed {fixed_count}/{len(rows_to_fix)} rows")
            return fixed_count
            
        except Exception as e:
            logger.error(f"Error fixing checkboxes: {e}")
            return 0


# Singleton instance
_sheets_client: Optional[GoogleSheetsClient] = None


def get_sheets_client() -> GoogleSheetsClient:
    """Get or create Google Sheets client singleton."""
    global _sheets_client
    if _sheets_client is None:
        _sheets_client = GoogleSheetsClient()
    return _sheets_client
