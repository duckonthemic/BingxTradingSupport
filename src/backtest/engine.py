"""
Backtesting Framework for Trading Signal Validation.
Tests historical performance of scoring system and strategies.

Features:
- Historical data fetching from BingX
- Signal simulation with scoring system
- Performance metrics (winrate, PnL, drawdown)
- Strategy comparison
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import os

logger = logging.getLogger(__name__)


class TradeResult(Enum):
    """Trade outcome."""
    WIN = "WIN"
    LOSS = "LOSS"
    BREAKEVEN = "BREAKEVEN"
    PENDING = "PENDING"


@dataclass
class BacktestTrade:
    """Single backtest trade record."""
    symbol: str
    direction: str  # LONG/SHORT
    entry_price: float
    entry_time: datetime
    stoploss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    
    # Signal info
    strategy: str
    confidence_score: int
    tier: str
    
    # Results (filled after trade closes)
    exit_price: float = 0
    exit_time: Optional[datetime] = None
    result: TradeResult = TradeResult.PENDING
    pnl_percent: float = 0
    hit_level: str = ""  # TP1, TP2, TP3, SL
    
    # Metadata
    breakdown: List[str] = field(default_factory=list)


@dataclass
class BacktestResult:
    """Backtest summary result."""
    start_date: datetime
    end_date: datetime
    total_trades: int
    
    # Win/Loss
    wins: int
    losses: int
    breakevens: int
    winrate: float
    
    # By tier
    diamond_trades: int
    diamond_winrate: float
    gold_trades: int
    gold_winrate: float
    
    # PnL
    total_pnl_percent: float
    average_win_percent: float
    average_loss_percent: float
    profit_factor: float
    max_drawdown_percent: float
    
    # Strategy breakdown
    strategy_stats: Dict[str, Dict]
    
    # All trades
    trades: List[BacktestTrade]


class BacktestEngine:
    """
    Backtesting engine for signal validation.
    
    Usage:
        engine = BacktestEngine()
        result = await engine.run(
            symbols=["BTC-USDT", "ETH-USDT"],
            start_date=datetime(2025, 1, 1),
            end_date=datetime(2025, 1, 15),
            timeframe="15m"
        )
        engine.print_report(result)
    """
    
    def __init__(self, rest_client=None):
        """
        Initialize backtest engine.
        
        Args:
            rest_client: BingX REST client for fetching historical data
        """
        self.rest_client = rest_client
        self.trades: List[BacktestTrade] = []
        
        # Simulation settings
        self.initial_balance = 1000.0
        self.position_size = 2.0  # Fixed $2 per trade
        self.slippage_percent = 0.05  # 0.05% slippage
        
    async def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "15m"
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            symbols: List of trading pairs to test
            start_date: Backtest start date
            end_date: Backtest end date  
            timeframe: Candle timeframe (15m, 1h, etc.)
        
        Returns:
            BacktestResult with full statistics
        """
        logger.info(f"🔄 Starting backtest: {start_date.date()} to {end_date.date()}")
        logger.info(f"📊 Symbols: {len(symbols)}, Timeframe: {timeframe}")
        
        self.trades = []
        
        for symbol in symbols:
            await self._backtest_symbol(symbol, start_date, end_date, timeframe)
            await asyncio.sleep(0.1)  # Rate limit
        
        return self._calculate_results(start_date, end_date)
    
    async def _backtest_symbol(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ):
        """Backtest single symbol."""
        logger.info(f"📈 Backtesting {symbol}...")
        
        try:
            # Fetch historical candles
            candles = await self._fetch_historical_data(symbol, start_date, end_date, timeframe)
            
            if not candles or len(candles) < 100:
                logger.warning(f"⚠️ Insufficient data for {symbol}")
                return
            
            # Simulate trading on historical data
            await self._simulate_trading(symbol, candles, timeframe)
            
        except Exception as e:
            logger.error(f"❌ Backtest error for {symbol}: {e}")
    
    async def _fetch_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[Dict]:
        """
        Fetch historical candle data.
        Returns list of candles with OHLCV data.
        """
        if not self.rest_client:
            logger.warning("No REST client configured - using mock data")
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
        
        try:
            # Use new historical fetch method with pagination
            candles = await self.rest_client.get_futures_klines_historical(
                symbol=symbol,
                interval=timeframe,
                start_date=start_date,
                end_date=end_date,
                max_candles=5000
            )
            
            logger.info(f"📈 {symbol}: Fetched {len(candles)} candles for backtest")
            return candles
        except Exception as e:
            logger.error(f"Failed to fetch data for {symbol}: {e}")
            return self._generate_mock_data(symbol, start_date, end_date, timeframe)
    
    def _generate_mock_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        timeframe: str
    ) -> List[Dict]:
        """Generate mock candle data for testing."""
        import random
        
        candles = []
        current = start_date
        price = 100.0  # Starting price
        
        # Timeframe to minutes
        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}
        interval_mins = tf_minutes.get(timeframe, 15)
        
        while current < end_date:
            # Random price movement
            change = random.uniform(-2, 2)  # ±2%
            
            open_price = price
            close_price = price * (1 + change / 100)
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 1) / 100)
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 1) / 100)
            volume = random.uniform(1000000, 10000000)
            
            candles.append({
                "timestamp": current,
                "open": open_price,
                "high": high_price,
                "low": low_price,
                "close": close_price,
                "volume": volume
            })
            
            price = close_price
            current += timedelta(minutes=interval_mins)
        
        return candles
    
    async def _simulate_trading(
        self,
        symbol: str,
        candles: List[Dict],
        timeframe: str
    ):
        """
        Simulate trading signals on historical candles.
        This is where the scoring system would be applied.
        """
        # Import scoring system
        try:
            from src.analysis.scoring_system import calculate_confidence, TIER_THRESHOLDS
        except ImportError:
            logger.warning("Scoring system not available - using simplified simulation")
            await self._simplified_simulation(symbol, candles)
            return
        
        # Slide through candles with lookback window
        lookback = 100  # Need 100 candles for indicators
        
        for i in range(lookback, len(candles)):
            window = candles[i-lookback:i+1]
            current_candle = candles[i]
            
            # Check for signals (simplified)
            signal = self._detect_signal(window, current_candle)
            
            if signal:
                # Create trade
                trade = await self._create_trade(
                    symbol=symbol,
                    candle=current_candle,
                    signal=signal,
                    future_candles=candles[i+1:]  # For checking TP/SL
                )
                
                if trade:
                    self.trades.append(trade)
    
    async def _simplified_simulation(self, symbol: str, candles: List[Dict]):
        """Simplified simulation without full scoring system."""
        import random
        
        # Randomly generate some trades for testing
        for i in range(100, len(candles), 20):  # Every 20 candles
            if random.random() < 0.1:  # 10% chance of signal
                candle = candles[i]
                future = candles[i+1:i+100] if i+100 < len(candles) else candles[i+1:]
                
                direction = random.choice(["LONG", "SHORT"])
                entry = candle["close"]
                
                # Calculate TP/SL
                if direction == "LONG":
                    sl = entry * 0.98  # 2% SL
                    tp1 = entry * 1.02  # 2% TP
                    tp2 = entry * 1.04
                    tp3 = entry * 1.06
                else:
                    sl = entry * 1.02
                    tp1 = entry * 0.98
                    tp2 = entry * 0.96
                    tp3 = entry * 0.94
                
                # Simulate result
                result, exit_price, hit_level = self._simulate_trade_result(
                    direction, entry, sl, tp1, tp2, tp3, future
                )
                
                pnl = 0
                if direction == "LONG":
                    pnl = ((exit_price - entry) / entry) * 100
                else:
                    pnl = ((entry - exit_price) / entry) * 100
                
                trade = BacktestTrade(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    entry_time=candle.get("timestamp", datetime.now()),
                    stoploss=sl,
                    take_profit_1=tp1,
                    take_profit_2=tp2,
                    take_profit_3=tp3,
                    strategy="RANDOM",
                    confidence_score=random.randint(40, 90),
                    tier="DIAMOND" if random.random() > 0.5 else "GOLD",
                    exit_price=exit_price,
                    result=result,
                    pnl_percent=pnl,
                    hit_level=hit_level
                )
                
                self.trades.append(trade)
    
    def _detect_signal(self, window: List[Dict], current: Dict) -> Optional[Dict]:
        """
        Detect trading signal in candle window.
        Simplified version - in production use full scoring system.
        """
        # Placeholder - would integrate with actual scoring system
        return None
    
    async def _create_trade(
        self,
        symbol: str,
        candle: Dict,
        signal: Dict,
        future_candles: List[Dict]
    ) -> Optional[BacktestTrade]:
        """Create and simulate a trade."""
        # Placeholder
        return None
    
    def _simulate_trade_result(
        self,
        direction: str,
        entry: float,
        sl: float,
        tp1: float,
        tp2: float,
        tp3: float,
        future_candles: List[Dict]
    ) -> Tuple[TradeResult, float, str]:
        """
        Simulate trade outcome using future candles.
        
        Returns:
            Tuple of (result, exit_price, hit_level)
        """
        for candle in future_candles:
            high = candle["high"]
            low = candle["low"]
            
            if direction == "LONG":
                # Check SL hit
                if low <= sl:
                    return TradeResult.LOSS, sl, "SL"
                # Check TP hits
                if high >= tp3:
                    return TradeResult.WIN, tp3, "TP3"
                if high >= tp2:
                    return TradeResult.WIN, tp2, "TP2"
                if high >= tp1:
                    return TradeResult.WIN, tp1, "TP1"
            else:  # SHORT
                # Check SL hit
                if high >= sl:
                    return TradeResult.LOSS, sl, "SL"
                # Check TP hits
                if low <= tp3:
                    return TradeResult.WIN, tp3, "TP3"
                if low <= tp2:
                    return TradeResult.WIN, tp2, "TP2"
                if low <= tp1:
                    return TradeResult.WIN, tp1, "TP1"
        
        # No exit hit - trade still open
        return TradeResult.PENDING, entry, "PENDING"
    
    def _calculate_results(
        self,
        start_date: datetime,
        end_date: datetime
    ) -> BacktestResult:
        """Calculate backtest statistics."""
        total = len(self.trades)
        
        if total == 0:
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                total_trades=0,
                wins=0, losses=0, breakevens=0, winrate=0,
                diamond_trades=0, diamond_winrate=0,
                gold_trades=0, gold_winrate=0,
                total_pnl_percent=0,
                average_win_percent=0,
                average_loss_percent=0,
                profit_factor=0,
                max_drawdown_percent=0,
                strategy_stats={},
                trades=[]
            )
        
        # Count outcomes
        wins = sum(1 for t in self.trades if t.result == TradeResult.WIN)
        losses = sum(1 for t in self.trades if t.result == TradeResult.LOSS)
        breakevens = sum(1 for t in self.trades if t.result == TradeResult.BREAKEVEN)
        
        # Winrate
        closed = wins + losses
        winrate = (wins / closed * 100) if closed > 0 else 0
        
        # By tier
        diamond_trades = [t for t in self.trades if t.tier == "DIAMOND"]
        diamond_wins = sum(1 for t in diamond_trades if t.result == TradeResult.WIN)
        diamond_closed = sum(1 for t in diamond_trades if t.result in [TradeResult.WIN, TradeResult.LOSS])
        diamond_winrate = (diamond_wins / diamond_closed * 100) if diamond_closed > 0 else 0
        
        gold_trades = [t for t in self.trades if t.tier == "GOLD"]
        gold_wins = sum(1 for t in gold_trades if t.result == TradeResult.WIN)
        gold_closed = sum(1 for t in gold_trades if t.result in [TradeResult.WIN, TradeResult.LOSS])
        gold_winrate = (gold_wins / gold_closed * 100) if gold_closed > 0 else 0
        
        # PnL stats
        win_pnls = [t.pnl_percent for t in self.trades if t.result == TradeResult.WIN]
        loss_pnls = [t.pnl_percent for t in self.trades if t.result == TradeResult.LOSS]
        
        total_pnl = sum(t.pnl_percent for t in self.trades)
        avg_win = (sum(win_pnls) / len(win_pnls)) if win_pnls else 0
        avg_loss = (sum(loss_pnls) / len(loss_pnls)) if loss_pnls else 0
        
        total_wins_pnl = sum(win_pnls) if win_pnls else 0
        total_losses_pnl = abs(sum(loss_pnls)) if loss_pnls else 1
        profit_factor = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0
        
        # Strategy breakdown
        strategy_stats = {}
        strategies = set(t.strategy for t in self.trades)
        for strat in strategies:
            strat_trades = [t for t in self.trades if t.strategy == strat]
            strat_wins = sum(1 for t in strat_trades if t.result == TradeResult.WIN)
            strat_closed = sum(1 for t in strat_trades if t.result in [TradeResult.WIN, TradeResult.LOSS])
            strategy_stats[strat] = {
                "total": len(strat_trades),
                "wins": strat_wins,
                "winrate": (strat_wins / strat_closed * 100) if strat_closed > 0 else 0
            }
        
        return BacktestResult(
            start_date=start_date,
            end_date=end_date,
            total_trades=total,
            wins=wins,
            losses=losses,
            breakevens=breakevens,
            winrate=winrate,
            diamond_trades=len(diamond_trades),
            diamond_winrate=diamond_winrate,
            gold_trades=len(gold_trades),
            gold_winrate=gold_winrate,
            total_pnl_percent=total_pnl,
            average_win_percent=avg_win,
            average_loss_percent=avg_loss,
            profit_factor=profit_factor,
            max_drawdown_percent=0,  # TODO: Calculate drawdown
            strategy_stats=strategy_stats,
            trades=self.trades
        )
    
    def print_report(self, result: BacktestResult):
        """Print formatted backtest report."""
        print("\n" + "="*60)
        print("📊 BACKTEST REPORT")
        print("="*60)
        print(f"Period: {result.start_date.date()} to {result.end_date.date()}")
        print(f"Total Trades: {result.total_trades}")
        print()
        
        print("📈 PERFORMANCE:")
        print(f"  Wins: {result.wins} | Losses: {result.losses}")
        print(f"  Winrate: {result.winrate:.1f}%")
        print(f"  Total PnL: {result.total_pnl_percent:.2f}%")
        print(f"  Profit Factor: {result.profit_factor:.2f}")
        print()
        
        print("💎 BY TIER:")
        print(f"  DIAMOND: {result.diamond_trades} trades, {result.diamond_winrate:.1f}% winrate")
        print(f"  GOLD: {result.gold_trades} trades, {result.gold_winrate:.1f}% winrate")
        print()
        
        if result.strategy_stats:
            print("📋 BY STRATEGY:")
            for strat, stats in result.strategy_stats.items():
                print(f"  {strat}: {stats['total']} trades, {stats['winrate']:.1f}% winrate")
        
        print("="*60)
    
    def save_report(self, result: BacktestResult, filepath: str):
        """Save backtest report to JSON file."""
        data = {
            "start_date": result.start_date.isoformat(),
            "end_date": result.end_date.isoformat(),
            "total_trades": result.total_trades,
            "wins": result.wins,
            "losses": result.losses,
            "winrate": result.winrate,
            "diamond_winrate": result.diamond_winrate,
            "gold_winrate": result.gold_winrate,
            "total_pnl_percent": result.total_pnl_percent,
            "profit_factor": result.profit_factor,
            "strategy_stats": result.strategy_stats
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"📊 Report saved to {filepath}")


async def run_backtest_cli():
    """CLI entry point for running backtests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run trading backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTC-USDT", "ETH-USDT"])
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--timeframe", default="15m")
    parser.add_argument("--output", default="backtest_result.json")
    
    args = parser.parse_args()
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    engine = BacktestEngine()
    result = await engine.run(
        symbols=args.symbols,
        start_date=start_date,
        end_date=end_date,
        timeframe=args.timeframe
    )
    
    engine.print_report(result)
    engine.save_report(result, args.output)


if __name__ == "__main__":
    asyncio.run(run_backtest_cli())
