"""
Run Backtest Script - Validate DIAMOND winrate

Usage:
    python run_backtest.py --days 7 --symbols BTC-USDT ETH-USDT SOL-USDT
    python run_backtest.py --days 14  # Test all top coins for 14 days
"""

import asyncio
import argparse
import logging
from datetime import datetime, timedelta
from typing import List

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Default top coins to backtest
DEFAULT_SYMBOLS = [
    "BTC-USDT", "ETH-USDT", "SOL-USDT", "XRP-USDT", "DOGE-USDT",
    "ADA-USDT", "AVAX-USDT", "LINK-USDT", "DOT-USDT", "MATIC-USDT",
    "UNI-USDT", "ATOM-USDT", "LTC-USDT", "FIL-USDT", "ARB-USDT"
]


async def run_backtest(days: int = 7, symbols: List[str] = None):
    """
    Run backtest on historical data.
    
    Args:
        days: Number of days to backtest
        symbols: List of trading pairs to test
    """
    from src.backtest.engine import BacktestEngine
    from src.ingestion.rest_client import BingXRestClient
    
    symbols = symbols or DEFAULT_SYMBOLS
    end_date = datetime.utcnow()
    start_date = end_date - timedelta(days=days)
    
    print("=" * 70)
    print(f"🔄 BingX Alert Bot - Backtest v1.0")
    print("=" * 70)
    print(f"📅 Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} ({days} days)")
    print(f"📊 Symbols: {len(symbols)}")
    print(f"💱 Timeframe: 15m")
    print("=" * 70)
    
    async with BingXRestClient() as client:
        # Initialize engine with REST client
        engine = BacktestEngine(rest_client=client)
        
        # Run backtest
        result = await engine.run(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe="15m"
        )
        
        # Print results
        print_report(result)
        
        return result


def print_report(result):
    """Print backtest report."""
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)
    
    print(f"\n📈 OVERALL PERFORMANCE:")
    print(f"   Total Trades: {result.total_trades}")
    print(f"   Win/Loss/BE: {result.wins}/{result.losses}/{result.breakevens}")
    print(f"   Winrate: {result.winrate:.1f}%")
    print(f"   Total PnL: {result.total_pnl_percent:+.2f}%")
    print(f"   Profit Factor: {result.profit_factor:.2f}")
    print(f"   Max Drawdown: {result.max_drawdown_percent:.2f}%")
    
    print(f"\n💎 TIER BREAKDOWN:")
    print(f"   DIAMOND Trades: {result.diamond_trades} (Winrate: {result.diamond_winrate:.1f}%)")
    print(f"   GOLD Trades: {result.gold_trades} (Winrate: {result.gold_winrate:.1f}%)")
    
    # Expected: Diamond >= 70%, Gold >= 55%
    diamond_target = 70
    gold_target = 55
    
    print(f"\n🎯 TARGET VALIDATION:")
    diamond_status = "✅" if result.diamond_winrate >= diamond_target else "❌"
    gold_status = "✅" if result.gold_winrate >= gold_target else "❌"
    print(f"   {diamond_status} DIAMOND: {result.diamond_winrate:.1f}% (target: {diamond_target}%)")
    print(f"   {gold_status} GOLD: {result.gold_winrate:.1f}% (target: {gold_target}%)")
    
    print(f"\n📋 STRATEGY BREAKDOWN:")
    for strategy, stats in result.strategy_stats.items():
        print(f"   {strategy}:")
        print(f"      Trades: {stats.get('count', 0)}")
        print(f"      Winrate: {stats.get('winrate', 0):.1f}%")
        print(f"      Avg PnL: {stats.get('avg_pnl', 0):+.2f}%")
    
    print("\n" + "=" * 70)
    
    # Show sample trades
    print("\n📝 SAMPLE TRADES (Last 10):")
    for trade in result.trades[-10:]:
        emoji = "✅" if trade.result.value == "WIN" else "❌"
        print(f"   {emoji} {trade.symbol} {trade.direction} | "
              f"Score: {trade.confidence_score} ({trade.tier}) | "
              f"PnL: {trade.pnl_percent:+.2f}% | Hit: {trade.hit_level}")


def main():
    parser = argparse.ArgumentParser(description="Run backtest on trading signals")
    parser.add_argument("--days", type=int, default=7, help="Days to backtest (default: 7)")
    parser.add_argument("--symbols", nargs="+", default=None, help="Symbols to test")
    
    args = parser.parse_args()
    
    asyncio.run(run_backtest(days=args.days, symbols=args.symbols))


if __name__ == "__main__":
    main()
