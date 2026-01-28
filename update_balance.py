"""
Manual Balance Update Script
Updates G2 cell in Google Sheet with calculated account balance.
Run: python update_balance.py
"""
import asyncio
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.storage.sheets_client import GoogleSheetsClient

INITIAL_BALANCE = 25.0  # Starting capital
POSITION_SIZE = 1.0     # $1 per trade


async def calculate_and_update_balance():
    """Calculate balance from all trades and update G2."""
    print("🔄 Connecting to Google Sheets...")
    
    client = GoogleSheetsClient(
        credentials_path="credentials.json",
        initial_balance=INITIAL_BALANCE
    )
    
    connected = await client.connect()
    if not connected:
        print("❌ Failed to connect to Google Sheets")
        return
    
    print("✅ Connected to Google Sheets")
    
    # Get all trades and calculate PnL
    try:
        all_data = client.sheet.get_all_values()
        total_pnl_usd = 0.0
        trade_count = 0
        wins = 0
        losses = 0
        
        print("\n📊 Analyzing trades...")
        
        for row in all_data[5:]:  # Skip headers (row 5)
            # Only count closed trades (Status = TP or SL in column K, index 10)
            if len(row) > 10 and row[10] in ["TP", "SL", "CLOSED"]:
                pnl_str = row[9].replace('%', '').strip() if len(row) > 9 else ""
                try:
                    pnl_percent = float(pnl_str)
                    pnl_usd = POSITION_SIZE * pnl_percent / 100
                    total_pnl_usd += pnl_usd
                    trade_count += 1
                    
                    if pnl_percent > 0:
                        wins += 1
                    else:
                        losses += 1
                        
                except ValueError:
                    pass
        
        # Calculate final balance
        account_balance = INITIAL_BALANCE + total_pnl_usd
        winrate = (wins / trade_count * 100) if trade_count > 0 else 0
        
        print(f"\n{'='*50}")
        print(f"📈 BALANCE SUMMARY")
        print(f"{'='*50}")
        print(f"Initial Balance:  ${INITIAL_BALANCE:.2f}")
        print(f"Total Trades:     {trade_count}")
        print(f"Wins/Losses:      {wins}/{losses}")
        print(f"Win Rate:         {winrate:.1f}%")
        print(f"Total PnL:        ${total_pnl_usd:+.2f}")
        print(f"{'='*50}")
        print(f"💰 CURRENT BALANCE: ${account_balance:.2f}")
        print(f"{'='*50}")
        
        # Update G2 in sheet
        print("\n🔄 Updating G2 in sheet...")
        client.sheet.update('G2', f'${account_balance:.2f}')
        client.sheet.update('G3', 'Balance')
        
        print("✅ G2 updated successfully!")
        print(f"\n💰 Balance is now: ${account_balance:.2f}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(calculate_and_update_balance())
