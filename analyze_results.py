"""
Analyze trading results from 20-28 Jan 2026
"""
import csv
from collections import defaultdict
from datetime import datetime

def parse_date(date_str):
    """Parse date string"""
    if '-' in date_str and len(date_str) > 10:
        # Format: 2026-01-24 19:05
        return datetime.strptime(date_str.split()[0], "%Y-%m-%d")
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except:
        return None

def parse_pnl(pnl_str):
    """Parse PnL percentage"""
    try:
        return float(pnl_str.replace('%', '').replace(',', '.'))
    except:
        return 0.0

def analyze_trades(file_path):
    """Analyze trading results"""
    
    trades = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        rows = list(reader)
    
    # Skip header rows (first 5)
    for row in rows[5:]:
        if len(row) < 14 or not row[1] or not row[2]:
            continue
        
        trade = {
            'no': row[0],
            'date': row[1],
            'coin': row[2],
            'signal': row[3],
            'leverage': int(row[4]) if row[4].isdigit() else 15,
            'entry': row[5],
            'sl': row[6],
            'tp': row[7],
            'price_now': row[8],
            'pnl': parse_pnl(row[9]),
            'status': row[10],
            'close_time': row[11],
            'strategy': row[12],
            'end_trade': row[13],
            'grade': row[15] if len(row) > 15 else '',
            'layers': row[16] if len(row) > 16 else '',
            'checklist': row[17] if len(row) > 17 else '',
        }
        trades.append(trade)
    
    # Filter closed trades only
    closed_trades = [t for t in trades if t['end_trade'].upper() in ['TRUE', '✓', '✔', '1']]
    open_trades = [t for t in trades if t['end_trade'].upper() not in ['TRUE', '✓', '✔', '1']]
    
    print("=" * 70)
    print("📊 PHÂN TÍCH KẾT QUẢ GIAO DỊCH 20-28/01/2026")
    print("=" * 70)
    
    # Overall stats
    total_trades = len(closed_trades)
    wins = [t for t in closed_trades if t['pnl'] > 0]
    losses = [t for t in closed_trades if t['pnl'] < 0]
    breakeven = [t for t in closed_trades if t['pnl'] == 0]
    
    total_pnl = sum(t['pnl'] for t in closed_trades)
    winrate = len(wins) / total_trades * 100 if total_trades > 0 else 0
    
    print(f"\n📈 TỔNG QUAN:")
    print(f"   Tổng trades đã đóng: {total_trades}")
    print(f"   Open trades: {len(open_trades)}")
    print(f"   Wins: {len(wins)} | Losses: {len(losses)} | Breakeven: {len(breakeven)}")
    print(f"   Winrate: {winrate:.1f}%")
    print(f"   Total PnL: {total_pnl:.2f}%")
    print(f"   Avg PnL per trade: {total_pnl/total_trades:.2f}%" if total_trades > 0 else "")
    
    # By Strategy
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO STRATEGY:")
    print("=" * 70)
    
    by_strategy = defaultdict(list)
    for t in closed_trades:
        by_strategy[t['strategy']].append(t)
    
    strategy_stats = []
    for strategy, strades in sorted(by_strategy.items(), key=lambda x: len(x[1]), reverse=True):
        total = len(strades)
        win_count = len([t for t in strades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in strades)
        avg_pnl = pnl / total if total > 0 else 0
        
        strategy_stats.append({
            'strategy': strategy,
            'total': total,
            'wins': win_count,
            'winrate': wr,
            'pnl': pnl,
            'avg_pnl': avg_pnl
        })
        
        emoji = "✅" if wr >= 50 else "❌"
        print(f"\n{emoji} {strategy}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}% | Avg: {avg_pnl:.2f}%")
    
    # By Signal Direction
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO HƯỚNG:")
    print("=" * 70)
    
    by_signal = defaultdict(list)
    for t in closed_trades:
        by_signal[t['signal']].append(t)
    
    for signal, strades in by_signal.items():
        total = len(strades)
        win_count = len([t for t in strades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in strades)
        
        emoji = "🟢" if signal == "LONG" else "🔴"
        print(f"\n{emoji} {signal}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}%")
    
    # By Leverage
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO LEVERAGE:")
    print("=" * 70)
    
    by_leverage = defaultdict(list)
    for t in closed_trades:
        by_leverage[t['leverage']].append(t)
    
    for lev, ltrades in sorted(by_leverage.items()):
        total = len(ltrades)
        win_count = len([t for t in ltrades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in ltrades)
        
        print(f"\n   x{lev}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}%")
    
    # By Grade
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO GRADE:")
    print("=" * 70)
    
    by_grade = defaultdict(list)
    for t in closed_trades:
        grade = t['grade'] if t['grade'] else 'NO_GRADE'
        by_grade[grade].append(t)
    
    for grade, gtrades in sorted(by_grade.items()):
        total = len(gtrades)
        win_count = len([t for t in gtrades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in gtrades)
        
        emoji = "💎" if grade == "DIAMOND" else "🥇" if grade == "GOLD" else "🎯" if grade == "A_SNIPER" else "⚪"
        print(f"\n{emoji} {grade}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}%")
    
    # By Layers (4-layer filter)
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO 4-LAYER FILTER:")
    print("=" * 70)
    
    by_layers = defaultdict(list)
    for t in closed_trades:
        layers = t['layers'] if t['layers'] else 'NO_DATA'
        by_layers[layers].append(t)
    
    for layers, ltrades in sorted(by_layers.items()):
        total = len(ltrades)
        win_count = len([t for t in ltrades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in ltrades)
        
        print(f"\n   {layers}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}%")
    
    # By Checklist
    print("\n" + "=" * 70)
    print("📊 PHÂN TÍCH THEO CHECKLIST:")
    print("=" * 70)
    
    by_checklist = defaultdict(list)
    for t in closed_trades:
        checklist = t['checklist'] if t['checklist'] else 'NO_DATA'
        by_checklist[checklist].append(t)
    
    for checklist, ctrades in sorted(by_checklist.items()):
        total = len(ctrades)
        win_count = len([t for t in ctrades if t['pnl'] > 0])
        wr = win_count / total * 100 if total > 0 else 0
        pnl = sum(t['pnl'] for t in ctrades)
        
        print(f"\n   {checklist}:")
        print(f"   Trades: {total} | Wins: {win_count} | WR: {wr:.1f}%")
        print(f"   Total PnL: {pnl:.2f}%")
    
    # Biggest Winners and Losers
    print("\n" + "=" * 70)
    print("🏆 TOP 5 WINNERS:")
    print("=" * 70)
    
    sorted_by_pnl = sorted(closed_trades, key=lambda x: x['pnl'], reverse=True)
    for i, t in enumerate(sorted_by_pnl[:5], 1):
        print(f"   {i}. {t['coin']} {t['signal']} | {t['strategy']} | +{t['pnl']:.2f}%")
    
    print("\n" + "=" * 70)
    print("💀 TOP 5 LOSERS:")
    print("=" * 70)
    
    for i, t in enumerate(sorted_by_pnl[-5:], 1):
        print(f"   {i}. {t['coin']} {t['signal']} | {t['strategy']} | {t['pnl']:.2f}%")
    
    # Issues Analysis
    print("\n" + "=" * 70)
    print("⚠️ VẤN ĐỀ PHÁT HIỆN:")
    print("=" * 70)
    
    # High leverage losses
    high_lev_losses = [t for t in closed_trades if t['leverage'] >= 50 and t['pnl'] < -50]
    if high_lev_losses:
        print(f"\n   ❌ Leverage cao (≥50x) gây loss lớn: {len(high_lev_losses)} trades")
        for t in high_lev_losses:
            print(f"      - {t['coin']} x{t['leverage']} | {t['pnl']:.2f}%")
    
    # SFP strategy analysis
    sfp_trades = [t for t in closed_trades if t['strategy'] == 'SFP']
    sfp_wins = len([t for t in sfp_trades if t['pnl'] > 0])
    sfp_wr = sfp_wins / len(sfp_trades) * 100 if sfp_trades else 0
    if sfp_wr < 50:
        print(f"\n   ❌ SFP có winrate thấp: {sfp_wr:.1f}% ({sfp_wins}/{len(sfp_trades)})")
    
    # LONG in bearish market
    long_losses = [t for t in closed_trades if t['signal'] == 'LONG' and t['pnl'] < 0]
    if len(long_losses) > len([t for t in closed_trades if t['signal'] == 'LONG']) / 2:
        print(f"\n   ❌ LONG trades đang thua nhiều trong thị trường BEARISH")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("📋 ĐỀ XUẤT CẢI TIẾN:")
    print("=" * 70)
    
    recommendations = []
    
    # Based on strategy performance
    good_strategies = [s for s in strategy_stats if s['winrate'] >= 55 and s['total'] >= 5]
    bad_strategies = [s for s in strategy_stats if s['winrate'] < 45 and s['total'] >= 5]
    
    if good_strategies:
        print(f"\n   ✅ Tập trung vào strategies hiệu quả:")
        for s in good_strategies:
            print(f"      - {s['strategy']}: WR={s['winrate']:.1f}%, Avg PnL={s['avg_pnl']:.2f}%")
    
    if bad_strategies:
        print(f"\n   ❌ Cần review/tạm dừng strategies kém:")
        for s in bad_strategies:
            print(f"      - {s['strategy']}: WR={s['winrate']:.1f}%, Avg PnL={s['avg_pnl']:.2f}%")
    
    print(f"""
   📌 ĐỀ XUẤT CHI TIẾT:
   
   1. LEVERAGE MANAGEMENT:
      - Giảm leverage cho altcoins về x10-15 thay vì x50
      - Chỉ dùng x50+ cho majors (BTC, ETH, SOL) với setup chất lượng cao
      
   2. STRATEGY FILTERING:
      - Tăng minimum score cho SFP lên 55+ (hiện tại WR thấp)
      - Ưu tiên EMA_PULLBACK và LIQ_SWEEP (WR cao nhất)
      
   3. DIRECTION BIAS:
      - Trong thị trường BEARISH, giảm số lượng LONG trades
      - Chỉ LONG khi có 3/4 layers passed
      
   4. 4-LAYER FILTER:
      - Bắt buộc ≥2/4 layers để vào lệnh
      - Trades 0/4 và 1/4 đang có WR thấp
      
   5. CHECKLIST:
      - Bắt buộc 3/3 checklist cho LONG trades
      - 2/3 minimum cho SHORT trades
      
   6. QUOTA MANAGEMENT:
      - Batch updates với delay 2s giữa các lệnh
      - Giảm tần suất update stats table
""")
    
    return {
        'total': total_trades,
        'winrate': winrate,
        'pnl': total_pnl,
        'by_strategy': strategy_stats
    }

if __name__ == "__main__":
    analyze_trades("TradeHistory.csv")
