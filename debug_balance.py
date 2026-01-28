"""
Debug Balance Calculation Script
Connects to Google Sheets and shows exactly how balance is calculated.
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.sheets_client import GoogleSheetsClient

INITIAL_BALANCE = 25.0
POSITION_SIZE = 1.0


async def debug_balance():
    """Debug balance calculation by showing all trades."""
    print("🔄 Connecting to Google Sheets...")
    
    client = GoogleSheetsClient(
        credentials_path="credentials.json",
        initial_balance=INITIAL_BALANCE
    )
    
    connected = await client.connect()
    if not connected:
        print("❌ Failed to connect to Google Sheets")
        return
    
    print("✅ Connected\n")
    
    # Get all data
    all_data = client.sheet.get_all_values()
    
    # Print headers to verify column positions
    print("=" * 80)
    print("HEADERS (Row 5):")
    if len(all_data) > 4:
        headers = all_data[4]  # Row 5 = index 4
        for i, h in enumerate(headers):
            print(f"  [{i}] {h}")
    
    print("\n" + "=" * 80)
    print("CLOSED TRADES (Status = TP/SL/CLOSED):")
    print("-" * 80)
    
    total_pnl_usd = 0.0
    total_pnl_percent = 0.0
    trade_count = 0
    wins = 0
    losses = 0
    
    for i, row in enumerate(all_data[5:], start=6):  # Start from row 6 (after headers)
        # Check if this is a closed trade
        if len(row) > 10:
            status = row[10] if len(row) > 10 else ""
            pnl_str = row[9] if len(row) > 9 else ""
            coin = row[2] if len(row) > 2 else ""
            
            if status in ["TP", "SL", "CLOSED"]:
                pnl_clean = pnl_str.replace('%', '').strip()
                try:
                    pnl_percent = float(pnl_clean)
                    pnl_usd = POSITION_SIZE * pnl_percent / 100
                    total_pnl_usd += pnl_usd
                    total_pnl_percent += pnl_percent
                    trade_count += 1
                    
                    if pnl_percent > 0:
                        wins += 1
                        status_emoji = "✅"
                    else:
                        losses += 1
                        status_emoji = "❌"
                    
                    print(f"Row {i}: {coin:12} | {status:6} | PnL: {pnl_percent:+8.2f}% | USD: ${pnl_usd:+.4f} | {status_emoji}")
                except ValueError:
                    print(f"Row {i}: {coin:12} | {status:6} | PnL: '{pnl_str}' (PARSE ERROR)")
    
    print("-" * 80)
    print(f"\n{'='*60}")
    print(f"BALANCE SUMMARY")
    print(f"{'='*60}")
    print(f"Total Closed Trades:  {trade_count}")
    print(f"Wins / Losses:        {wins} / {losses}")
    print(f"Win Rate:             {wins/trade_count*100:.1f}%" if trade_count > 0 else "N/A")
    print(f"Total PnL %:          {total_pnl_percent:+.2f}%")
    print(f"Total PnL USD:        ${total_pnl_usd:+.4f}")
    print(f"{'='*60}")
    print(f"Initial Balance:      ${INITIAL_BALANCE:.2f}")
    print(f"Final Balance:        ${INITIAL_BALANCE + total_pnl_usd:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(debug_balance())
