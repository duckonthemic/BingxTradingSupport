"""
Analyze SFP trades from TradeHistory.csv (v2)
Header is on row 5, Strategy is in Note column
"""
import csv

def analyze_sfp_trades():
    with open('TradeHistory.csv', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    # Skip header rows (rows 1-4), actual header is on row 5 (index 4)
    header_line = lines[4].strip()
    headers = [h.strip() for h in header_line.split(',')]
    
    print(f"Headers: {headers[:15]}")  # First 15 columns
    
    trades = []
    for line in lines[5:]:  # Data starts from row 6
        values = line.strip().split(',')
        if len(values) >= 15 and values[0].strip():  # Has data
            trade = dict(zip(headers, values))
            trades.append(trade)
    
    print(f"Total trades: {len(trades)}")
    
    # SFP trades (Note column contains strategy)
    sfp_trades = [t for t in trades if 'SFP' in str(t.get('Note', '')).upper()]
    print(f"\nSFP trades: {len(sfp_trades)}")
    
    # Analyze by direction
    sfp_long = [t for t in sfp_trades if t.get('Signal', '').upper() == 'LONG']
    sfp_short = [t for t in sfp_trades if t.get('Signal', '').upper() == 'SHORT']
    
    print(f"  SFP LONG: {len(sfp_long)}")
    print(f"  SFP SHORT: {len(sfp_short)}")
    
    # Win rates
    def calc_wr(trades):
        wins = [t for t in trades if 'TP' in t.get('Status', '').upper() or 'WIN' in t.get('Status', '').upper()]
        closed = [t for t in trades if 'CLOSED' in t.get('Status', '').upper()]
        losses = [t for t in trades if 'SL' in t.get('Status', '').upper() or 'LOSE' in t.get('Status', '').upper()]
        # CLOSED could be win or loss, count as win if PnL > 0
        for t in closed:
            pnl = t.get('PnL %', '0').replace('%', '')
            try:
                if float(pnl) > 0:
                    wins.append(t)
                else:
                    losses.append(t)
            except:
                pass
        return len(wins), len(losses)
    
    sfp_wins, sfp_losses = calc_wr(sfp_trades)
    print(f"\nSFP Overall: {sfp_wins} wins, {sfp_losses} losses, WR: {sfp_wins/(sfp_wins+sfp_losses)*100:.1f}%" if sfp_wins+sfp_losses > 0 else "N/A")
    
    long_wins, long_losses = calc_wr(sfp_long)
    print(f"SFP LONG: {long_wins} wins, {long_losses} losses, WR: {long_wins/(long_wins+long_losses)*100:.1f}%" if long_wins+long_losses > 0 else "N/A")
    
    short_wins, short_losses = calc_wr(sfp_short)
    print(f"SFP SHORT: {short_wins} wins, {short_losses} losses, WR: {short_wins/(short_wins+short_losses)*100:.1f}%" if short_wins+short_losses > 0 else "N/A")
    
    # Calculate PnL
    def calc_pnl(trades):
        total = 0
        for t in trades:
            pnl = t.get('PnL %', '0').replace('%', '')
            try:
                total += float(pnl)
            except:
                pass
        return total
    
    print(f"\nPnL:")
    print(f"  SFP Overall: {calc_pnl(sfp_trades):+.2f}%")
    print(f"  SFP LONG: {calc_pnl(sfp_long):+.2f}%")
    print(f"  SFP SHORT: {calc_pnl(sfp_short):+.2f}%")
    
    # Analyze by leverage
    print(f"\nSFP by Leverage:")
    for lev in ['15', '50', '100']:
        lev_trades = [t for t in sfp_trades if t.get('Leverage', '') == lev]
        if lev_trades:
            wins, losses = calc_wr(lev_trades)
            pnl = calc_pnl(lev_trades)
            print(f"  x{lev}: {len(lev_trades)} trades, WR: {wins/(wins+losses)*100:.1f}%, PnL: {pnl:+.2f}%" if wins+losses > 0 else f"  x{lev}: {len(lev_trades)} trades")
    
    # Analyze by Checklist
    print(f"\nSFP by Checklist:")
    for cl in ['2/3', '3/3']:
        cl_trades = [t for t in sfp_trades if t.get('Checklist', '') == cl]
        if cl_trades:
            wins, losses = calc_wr(cl_trades)
            pnl = calc_pnl(cl_trades)
            print(f"  {cl}: {len(cl_trades)} trades, WR: {wins/(wins+losses)*100:.1f}%, PnL: {pnl:+.2f}%" if wins+losses > 0 else f"  {cl}: {len(cl_trades)} trades")
    
    # Show worst SFP LONG trades
    print(f"\n--- Worst SFP LONG trades (by PnL) ---")
    sfp_long_with_pnl = []
    for t in sfp_long:
        pnl = t.get('PnL %', '0').replace('%', '')
        try:
            sfp_long_with_pnl.append((float(pnl), t))
        except:
            pass
    sfp_long_with_pnl.sort(key=lambda x: x[0])
    
    for pnl, t in sfp_long_with_pnl[:10]:
        print(f"  {t.get('Coin')}: {pnl:+.2f}%, x{t.get('Leverage')}, {t.get('Checklist', 'N/A')}")

if __name__ == "__main__":
    analyze_sfp_trades()
