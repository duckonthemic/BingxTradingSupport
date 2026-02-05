"""
Comprehensive Trade History Analysis Script
Analyzes TradeHistory2.csv to identify patterns and areas for improvement
"""
import pandas as pd
import numpy as np
from collections import defaultdict
import re


def clean_pnl(val):
    """Convert PnL string to float"""
    if pd.isna(val):
        return 0.0
    val = str(val).replace('%', '').replace(',', '').strip()
    try:
        return float(val)
    except:
        return 0.0


def parse_checklist(val):
    """Parse checklist string like '2/3' to tuple"""
    if pd.isna(val) or not val:
        return (0, 3)
    match = re.search(r'(\d+)/(\d+)', str(val))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 3)


def parse_layers(val):
    """Parse layers string like '2/4' to tuple"""
    if pd.isna(val) or not val:
        return (0, 4)
    match = re.search(r'(\d+)/(\d+)', str(val))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 4)


def analyze():
    # Read CSV
    df = pd.read_csv('TradeHistory2.csv', skiprows=4)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Filter closed trades only
    closed_df = df[df['End Trade'] == True].copy()
    
    # Clean PnL
    closed_df['PnL_clean'] = closed_df['PnL %'].apply(clean_pnl)
    
    # Parse status
    closed_df['is_win'] = closed_df['PnL_clean'] > 0
    closed_df['is_loss'] = closed_df['PnL_clean'] < 0
    
    # Parse checklist and layers
    closed_df['checklist_tuple'] = closed_df['Checklist'].apply(parse_checklist)
    closed_df['checklist_score'] = closed_df['checklist_tuple'].apply(lambda x: x[0])
    closed_df['layers_tuple'] = closed_df['Layers'].apply(parse_layers)
    closed_df['layers_score'] = closed_df['layers_tuple'].apply(lambda x: x[0])
    
    # Convert leverage to int
    closed_df['Leverage'] = pd.to_numeric(closed_df['Leverage'], errors='coerce').fillna(15).astype(int)
    
    print("=" * 80)
    print("COMPREHENSIVE TRADE ANALYSIS - TradeHistory2.csv")
    print("=" * 80)
    
    # 1. OVERALL STATISTICS
    total = len(closed_df)
    wins = closed_df['is_win'].sum()
    losses = closed_df['is_loss'].sum()
    winrate = (wins / total * 100) if total > 0 else 0
    total_pnl = closed_df['PnL_clean'].sum()
    avg_win = closed_df[closed_df['is_win']]['PnL_clean'].mean()
    avg_loss = closed_df[closed_df['is_loss']]['PnL_clean'].mean()
    
    print(f"\n📊 OVERALL STATISTICS")
    print(f"{'─' * 40}")
    print(f"Total Closed Trades: {total}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Winrate: {winrate:.1f}%")
    print(f"Total PnL: {total_pnl:+.2f}%")
    print(f"Avg Win: {avg_win:+.2f}% | Avg Loss: {avg_loss:.2f}%")
    print(f"Risk/Reward Ratio: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "N/A")
    
    # 2. BY DIRECTION
    print(f"\n📈 BY DIRECTION")
    print(f"{'─' * 40}")
    for direction in ['LONG', 'SHORT']:
        dir_df = closed_df[closed_df['Signal'] == direction]
        if len(dir_df) == 0:
            continue
        dir_wins = dir_df['is_win'].sum()
        dir_total = len(dir_df)
        dir_wr = (dir_wins / dir_total * 100) if dir_total > 0 else 0
        dir_pnl = dir_df['PnL_clean'].sum()
        print(f"{direction:6}: {dir_total:3} trades | WR: {dir_wr:5.1f}% | PnL: {dir_pnl:+8.1f}%")
    
    # 3. BY STRATEGY
    print(f"\n🎯 BY STRATEGY")
    print(f"{'─' * 40}")
    strategies = closed_df['Note'].dropna().unique()
    strategy_stats = []
    for strat in strategies:
        # Extract strategy name (first part before ' | ')
        strat_name = str(strat).split('|')[0].strip()
        strat_df = closed_df[closed_df['Note'].str.contains(strat_name, case=False, na=False)]
        if len(strat_df) == 0:
            continue
        strat_wins = strat_df['is_win'].sum()
        strat_total = len(strat_df)
        strat_wr = (strat_wins / strat_total * 100) if strat_total > 0 else 0
        strat_pnl = strat_df['PnL_clean'].sum()
        strategy_stats.append({
            'strategy': strat_name[:20],
            'trades': strat_total,
            'wr': strat_wr,
            'pnl': strat_pnl
        })
    
    # Group by main strategy types
    main_strategies = ['EMA_PULLBACK', 'SFP', 'LIQ_SWEEP', 'BB_BOUNCE', 'BREAKER_RETEST', 'IE']
    for strat in main_strategies:
        strat_df = closed_df[closed_df['Note'].str.contains(strat, case=False, na=False)]
        if len(strat_df) == 0:
            continue
        strat_wins = strat_df['is_win'].sum()
        strat_total = len(strat_df)
        strat_wr = (strat_wins / strat_total * 100) if strat_total > 0 else 0
        strat_pnl = strat_df['PnL_clean'].sum()
        print(f"{strat:17}: {strat_total:3} trades | WR: {strat_wr:5.1f}% | PnL: {strat_pnl:+8.1f}%")
    
    # 4. BY LEVERAGE
    print(f"\n⚡ BY LEVERAGE")
    print(f"{'─' * 40}")
    for lev in sorted(closed_df['Leverage'].unique()):
        lev_df = closed_df[closed_df['Leverage'] == lev]
        if len(lev_df) == 0:
            continue
        lev_wins = lev_df['is_win'].sum()
        lev_total = len(lev_df)
        lev_wr = (lev_wins / lev_total * 100) if lev_total > 0 else 0
        lev_pnl = lev_df['PnL_clean'].sum()
        print(f"x{lev:3}: {lev_total:3} trades | WR: {lev_wr:5.1f}% | PnL: {lev_pnl:+8.1f}%")
    
    # 5. BY GRADE/TIER
    print(f"\n🏆 BY GRADE/TIER")
    print(f"{'─' * 40}")
    for grade in closed_df['Grade'].dropna().unique():
        grade_df = closed_df[closed_df['Grade'] == grade]
        if len(grade_df) == 0:
            continue
        grade_wins = grade_df['is_win'].sum()
        grade_total = len(grade_df)
        grade_wr = (grade_wins / grade_total * 100) if grade_total > 0 else 0
        grade_pnl = grade_df['PnL_clean'].sum()
        print(f"{str(grade):12}: {grade_total:3} trades | WR: {grade_wr:5.1f}% | PnL: {grade_pnl:+8.1f}%")
    
    # 6. BY CHECKLIST SCORE
    print(f"\n✅ BY CHECKLIST SCORE")
    print(f"{'─' * 40}")
    for score in sorted(closed_df['checklist_score'].unique()):
        cl_df = closed_df[closed_df['checklist_score'] == score]
        if len(cl_df) == 0:
            continue
        cl_wins = cl_df['is_win'].sum()
        cl_total = len(cl_df)
        cl_wr = (cl_wins / cl_total * 100) if cl_total > 0 else 0
        cl_pnl = cl_df['PnL_clean'].sum()
        print(f"{score}/3 checklist: {cl_total:3} trades | WR: {cl_wr:5.1f}% | PnL: {cl_pnl:+8.1f}%")
    
    # 7. BY LAYERS (SHORT only)
    print(f"\n🔲 BY LAYERS (SHORT trades)")
    print(f"{'─' * 40}")
    short_df = closed_df[closed_df['Signal'] == 'SHORT']
    for score in sorted(short_df['layers_score'].unique()):
        ly_df = short_df[short_df['layers_score'] == score]
        if len(ly_df) == 0:
            continue
        ly_wins = ly_df['is_win'].sum()
        ly_total = len(ly_df)
        ly_wr = (ly_wins / ly_total * 100) if ly_total > 0 else 0
        ly_pnl = ly_df['PnL_clean'].sum()
        print(f"{score}/4 layers: {ly_total:3} trades | WR: {ly_wr:5.1f}% | PnL: {ly_pnl:+8.1f}%")
    
    # 8. STRATEGY + DIRECTION BREAKDOWN
    print(f"\n🎯 STRATEGY + DIRECTION BREAKDOWN")
    print(f"{'─' * 50}")
    for strat in main_strategies:
        for direction in ['LONG', 'SHORT']:
            combo_df = closed_df[
                (closed_df['Note'].str.contains(strat, case=False, na=False)) &
                (closed_df['Signal'] == direction)
            ]
            if len(combo_df) == 0:
                continue
            combo_wins = combo_df['is_win'].sum()
            combo_total = len(combo_df)
            combo_wr = (combo_wins / combo_total * 100) if combo_total > 0 else 0
            combo_pnl = combo_df['PnL_clean'].sum()
            print(f"{strat:17} {direction:5}: {combo_total:3} trades | WR: {combo_wr:5.1f}% | PnL: {combo_pnl:+8.1f}%")
    
    # 9. WORST TRADES ANALYSIS
    print(f"\n💀 TOP 10 WORST TRADES")
    print(f"{'─' * 70}")
    worst = closed_df.nsmallest(10, 'PnL_clean')[['Date', 'Coin', 'Signal', 'Leverage', 'Note', 'PnL_clean', 'Checklist']]
    for _, row in worst.iterrows():
        strat = str(row['Note']).split('|')[0].strip()[:15] if pd.notna(row['Note']) else 'N/A'
        print(f"{row['Date']} | {row['Coin']:15} | {row['Signal']:5} | x{row['Leverage']:3} | {strat:15} | {row['PnL_clean']:+8.1f}% | {row['Checklist']}")
    
    # 10. BEST TRADES ANALYSIS
    print(f"\n🏆 TOP 10 BEST TRADES")
    print(f"{'─' * 70}")
    best = closed_df.nlargest(10, 'PnL_clean')[['Date', 'Coin', 'Signal', 'Leverage', 'Note', 'PnL_clean', 'Checklist']]
    for _, row in best.iterrows():
        strat = str(row['Note']).split('|')[0].strip()[:15] if pd.notna(row['Note']) else 'N/A'
        print(f"{row['Date']} | {row['Coin']:15} | {row['Signal']:5} | x{row['Leverage']:3} | {strat:15} | {row['PnL_clean']:+8.1f}% | {row['Checklist']}")
    
    # 11. HIGH LEVERAGE ANALYSIS (x50+)
    print(f"\n⚠️ HIGH LEVERAGE ANALYSIS (x50+)")
    print(f"{'─' * 50}")
    high_lev_df = closed_df[closed_df['Leverage'] >= 50]
    if len(high_lev_df) > 0:
        for coin in high_lev_df['Coin'].unique():
            coin_df = high_lev_df[high_lev_df['Coin'] == coin]
            coin_wins = coin_df['is_win'].sum()
            coin_total = len(coin_df)
            coin_wr = (coin_wins / coin_total * 100) if coin_total > 0 else 0
            coin_pnl = coin_df['PnL_clean'].sum()
            avg_lev = coin_df['Leverage'].mean()
            print(f"{coin:20}: {coin_total:2} trades | Avg Lev: x{avg_lev:.0f} | WR: {coin_wr:5.1f}% | PnL: {coin_pnl:+8.1f}%")
    
    # 12. GOLD/COMMODITIES ANALYSIS
    print(f"\n🥇 GOLD/SILVER/COMMODITIES ANALYSIS")
    print(f"{'─' * 50}")
    commodity_patterns = ['XAUT', 'PAXG', 'NCCO', 'NCC', 'SILVER', 'OIL', 'GOLD']
    for pattern in commodity_patterns:
        comm_df = closed_df[closed_df['Coin'].str.contains(pattern, case=False, na=False)]
        if len(comm_df) == 0:
            continue
        comm_wins = comm_df['is_win'].sum()
        comm_total = len(comm_df)
        comm_wr = (comm_wins / comm_total * 100) if comm_total > 0 else 0
        comm_pnl = comm_df['PnL_clean'].sum()
        avg_lev = comm_df['Leverage'].mean()
        print(f"{pattern:10}: {comm_total:3} trades | Avg Lev: x{avg_lev:.0f} | WR: {comm_wr:5.1f}% | PnL: {comm_pnl:+8.1f}%")
    
    # 13. KEY INSIGHTS
    print(f"\n" + "=" * 80)
    print("KEY INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    
    # Find patterns
    long_df = closed_df[closed_df['Signal'] == 'LONG']
    short_df = closed_df[closed_df['Signal'] == 'SHORT']
    
    long_wr = (long_df['is_win'].sum() / len(long_df) * 100) if len(long_df) > 0 else 0
    short_wr = (short_df['is_win'].sum() / len(short_df) * 100) if len(short_df) > 0 else 0
    long_pnl = long_df['PnL_clean'].sum()
    short_pnl = short_df['PnL_clean'].sum()
    
    print(f"\n1. DIRECTION BIAS:")
    print(f"   - LONG:  WR={long_wr:.1f}%, PnL={long_pnl:+.1f}%")
    print(f"   - SHORT: WR={short_wr:.1f}%, PnL={short_pnl:+.1f}%")
    if short_pnl > long_pnl + 500:
        print(f"   ⚠️ RECOMMENDATION: FOCUS ON SHORT trades, they're significantly more profitable")
    
    # Checklist analysis
    c3_df = closed_df[closed_df['checklist_score'] == 3]
    c2_df = closed_df[closed_df['checklist_score'] == 2]
    c1_df = closed_df[closed_df['checklist_score'] == 1]
    
    print(f"\n2. CHECKLIST CORRELATION:")
    if len(c3_df) > 0:
        c3_pnl = c3_df['PnL_clean'].sum()
        c3_wr = (c3_df['is_win'].sum() / len(c3_df) * 100)
        print(f"   - 3/3 Checklist: WR={c3_wr:.1f}%, PnL={c3_pnl:+.1f}%")
    if len(c2_df) > 0:
        c2_pnl = c2_df['PnL_clean'].sum()
        c2_wr = (c2_df['is_win'].sum() / len(c2_df) * 100)
        print(f"   - 2/3 Checklist: WR={c2_wr:.1f}%, PnL={c2_pnl:+.1f}%")
    if len(c1_df) > 0:
        c1_pnl = c1_df['PnL_clean'].sum()
        c1_wr = (c1_df['is_win'].sum() / len(c1_df) * 100)
        print(f"   - 1/3 Checklist: WR={c1_wr:.1f}%, PnL={c1_pnl:+.1f}%")
    
    # High leverage danger
    x500_df = closed_df[closed_df['Leverage'] >= 500]
    x100_df = closed_df[closed_df['Leverage'] == 100]
    x50_df = closed_df[closed_df['Leverage'] == 50]
    x15_df = closed_df[closed_df['Leverage'] == 15]
    
    print(f"\n3. LEVERAGE DANGER ZONES:")
    if len(x500_df) > 0:
        x500_pnl = x500_df['PnL_clean'].sum()
        x500_wr = (x500_df['is_win'].sum() / len(x500_df) * 100)
        print(f"   - x500: {len(x500_df)} trades, WR={x500_wr:.1f}%, PnL={x500_pnl:+.1f}%")
        if x500_pnl < -500:
            print(f"   ⚠️ CRITICAL: x500 leverage is destroying your account!")
    if len(x100_df) > 0:
        x100_pnl = x100_df['PnL_clean'].sum()
        x100_wr = (x100_df['is_win'].sum() / len(x100_df) * 100)
        print(f"   - x100: {len(x100_df)} trades, WR={x100_wr:.1f}%, PnL={x100_pnl:+.1f}%")
    if len(x15_df) > 0:
        x15_pnl = x15_df['PnL_clean'].sum()
        x15_wr = (x15_df['is_win'].sum() / len(x15_df) * 100)
        print(f"   - x15:  {len(x15_df)} trades, WR={x15_wr:.1f}%, PnL={x15_pnl:+.1f}%")
    
    # Strategy by direction
    print(f"\n4. STRATEGY-DIRECTION WINNERS:")
    best_combos = []
    for strat in main_strategies:
        for direction in ['LONG', 'SHORT']:
            combo_df = closed_df[
                (closed_df['Note'].str.contains(strat, case=False, na=False)) &
                (closed_df['Signal'] == direction)
            ]
            if len(combo_df) >= 5:
                combo_pnl = combo_df['PnL_clean'].sum()
                combo_wr = (combo_df['is_win'].sum() / len(combo_df) * 100)
                best_combos.append({
                    'combo': f"{strat} {direction}",
                    'trades': len(combo_df),
                    'wr': combo_wr,
                    'pnl': combo_pnl
                })
    
    best_combos.sort(key=lambda x: x['pnl'], reverse=True)
    for combo in best_combos[:5]:
        status = "✅" if combo['pnl'] > 0 else "❌"
        print(f"   {status} {combo['combo']:25}: {combo['trades']} trades, WR={combo['wr']:.1f}%, PnL={combo['pnl']:+.1f}%")
    
    print(f"\n5. WORST STRATEGY-DIRECTION COMBOS:")
    for combo in best_combos[-5:]:
        print(f"   ❌ {combo['combo']:25}: {combo['trades']} trades, WR={combo['wr']:.1f}%, PnL={combo['pnl']:+.1f}%")
    
    return closed_df


if __name__ == "__main__":
    analyze()
