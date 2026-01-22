"""
Trade Tracker - Monitors open trades and updates results.
Handles TP/SL detection and Telegram reply notifications.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ActiveTrade:
    """Represents an active trade being monitored."""
    trade_id: str
    symbol: str
    direction: str  # LONG/SHORT
    entry_price: float
    stop_loss: float
    take_profit_1: float  # 2R
    take_profit_2: float  # 4R
    take_profit_3: float  # 6R
    leverage: int
    position_size: float
    
    # Telegram tracking
    message_id: int = 0
    chat_id: str = ""
    
    # Sheet tracking
    sheet_row: int = 0
    
    # Status
    status: str = "OPEN"  # OPEN, TP1, TP2, TP3, SL, CLOSED
    entry_time: datetime = field(default_factory=datetime.now)
    
    # Partial exits
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    sl_hit: bool = False
    
    def check_price(self, current_price: float) -> Optional[str]:
        """
        Check if price hit any TP/SL level.
        
        Returns:
            Event string: "TP1", "TP2", "TP3", "SL", or None
        """
        if self.direction == "LONG":
            # Check SL first
            if current_price <= self.stop_loss and not self.sl_hit:
                return "SL"
            # Check TPs
            if current_price >= self.take_profit_3 and not self.tp3_hit:
                return "TP3"
            if current_price >= self.take_profit_2 and not self.tp2_hit:
                return "TP2"
            if current_price >= self.take_profit_1 and not self.tp1_hit:
                return "TP1"
        else:  # SHORT
            # Check SL first
            if current_price >= self.stop_loss and not self.sl_hit:
                return "SL"
            # Check TPs
            if current_price <= self.take_profit_3 and not self.tp3_hit:
                return "TP3"
            if current_price <= self.take_profit_2 and not self.tp2_hit:
                return "TP2"
            if current_price <= self.take_profit_1 and not self.tp1_hit:
                return "TP1"
        
        return None
    
    def calculate_pnl(self, exit_price: float) -> float:
        """Calculate PnL percentage."""
        if self.direction == "LONG":
            pnl = ((exit_price - self.entry_price) / self.entry_price) * 100 * self.leverage
        else:
            pnl = ((self.entry_price - exit_price) / self.entry_price) * 100 * self.leverage
        return round(pnl, 2)


class TradeTracker:
    """
    Tracks active trades and monitors for TP/SL hits.
    
    Features:
    - Real-time price monitoring
    - TP1/TP2/TP3/SL detection
    - Telegram reply notifications
    - Google Sheets update
    """
    
    def __init__(self):
        self.active_trades: Dict[str, ActiveTrade] = {}  # trade_id -> ActiveTrade
        self._running = False
        self._check_interval = 5  # seconds
        
        # Callbacks
        self.on_tp_hit = None  # async callback(trade, tp_level, pnl)
        self.on_sl_hit = None  # async callback(trade, pnl)
        
    def add_trade(self, trade: ActiveTrade) -> str:
        """Add a new trade to track."""
        self.active_trades[trade.trade_id] = trade
        logger.info(f"📍 Tracking: {trade.symbol} {trade.direction} @ {trade.entry_price}")
        return trade.trade_id
    
    def remove_trade(self, trade_id: str):
        """Remove a trade from tracking."""
        if trade_id in self.active_trades:
            del self.active_trades[trade_id]
            logger.info(f"🏁 Stopped tracking: {trade_id}")
    
    def get_trade(self, trade_id: str) -> Optional[ActiveTrade]:
        """Get active trade by ID."""
        return self.active_trades.get(trade_id)
    
    def get_trades_by_symbol(self, symbol: str) -> List[ActiveTrade]:
        """Get all active trades for a symbol."""
        return [t for t in self.active_trades.values() if t.symbol == symbol]
    
    async def check_trade(self, trade: ActiveTrade, current_price: float):
        """Check a single trade against current price."""
        event = trade.check_price(current_price)
        
        if not event:
            return
        
        pnl = trade.calculate_pnl(current_price)
        
        if event == "TP1":
            trade.tp1_hit = True
            trade.status = "TP1"
            logger.info(f"🎯 TP1 HIT: {trade.symbol} | PnL: {pnl:+.2f}%")
            if self.on_tp_hit:
                await self.on_tp_hit(trade, "TP1", pnl)
                
        elif event == "TP2":
            trade.tp2_hit = True
            trade.status = "TP2"
            logger.info(f"🎯 TP2 HIT: {trade.symbol} | PnL: {pnl:+.2f}%")
            if self.on_tp_hit:
                await self.on_tp_hit(trade, "TP2", pnl)
                
        elif event == "TP3":
            trade.tp3_hit = True
            trade.status = "TP3"
            logger.info(f"🎯 TP3 HIT: {trade.symbol} | PnL: {pnl:+.2f}%")
            if self.on_tp_hit:
                await self.on_tp_hit(trade, "TP3", pnl)
            # TP3 = full close
            self.remove_trade(trade.trade_id)
            
        elif event == "SL":
            trade.sl_hit = True
            trade.status = "SL"
            logger.info(f"💀 SL HIT: {trade.symbol} | PnL: {pnl:+.2f}%")
            if self.on_sl_hit:
                await self.on_sl_hit(trade, pnl)
            # SL = full close
            self.remove_trade(trade.trade_id)
    
    async def update_prices(self, prices: Dict[str, float]):
        """
        Update prices for all tracked symbols.
        
        Args:
            prices: Dict of symbol -> current_price
        """
        for trade_id, trade in list(self.active_trades.items()):
            if trade.symbol in prices:
                await self.check_trade(trade, prices[trade.symbol])
    
    def get_summary(self) -> Dict:
        """Get summary of active trades."""
        return {
            "active_count": len(self.active_trades),
            "symbols": list(set(t.symbol for t in self.active_trades.values())),
            "longs": sum(1 for t in self.active_trades.values() if t.direction == "LONG"),
            "shorts": sum(1 for t in self.active_trades.values() if t.direction == "SHORT")
        }


# Singleton
_trade_tracker: Optional[TradeTracker] = None


def get_trade_tracker() -> TradeTracker:
    """Get or create trade tracker singleton."""
    global _trade_tracker
    if _trade_tracker is None:
        _trade_tracker = TradeTracker()
    return _trade_tracker
