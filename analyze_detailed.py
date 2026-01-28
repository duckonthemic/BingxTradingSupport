"""
Deep Analysis of Trading Results (20-28 Jan 2026)
Analyze by: Strategy, Direction, Leverage, Checklist, Layers, Time, Patterns
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Read CSV
df = pd.read_csv('TradeHistory.csv', skiprows=4)
df.columns = ['No', 'Date', 'Coin', 'Signal', 'Leverage', 'Entry', 'SL', 'TP', 
              'Price_Now', 'PnL_Pct', 'Status', 'Close_Time', 'Note', 'End_Trade',
              'User_Note', 'Grade', 'Layers', 'Checklist'] + [f'Col_{i}' for i in range(18, len(df.columns))]

# Filter valid rows
df = df[df['Status'].notna() & (df['Status'] != '')]
df = df[df['Coin'].notna()]

# Parse PnL
def parse_pnl(val):
    if pd.isna(val) or val == '': return 0
    try:
        return float(str(val).replace('%', '').replace(',', ''))
    except: return 0

df['PnL'] = df['PnL_Pct'].apply(parse_pnl)
df['Leverage'] = pd.to_numeric(df['Leverage'], errors='coerce').fillna(15).astype(int)

# Parse checklist to numbers
def parse_checklist(val):
    if pd.isna(val) or val == '': return None
    try:
        parts = str(val).split('/')
        return int(parts[0])
    except: return None

def parse_layers(val):
    if pd.isna(val) or val == '': return None
    try:
        parts = str(val).split('/')
        return int(parts[0])
    except: return None

df['Checklist_Num'] = df['Checklist'].apply(parse_checklist)
df['Layers_Num'] = df['Layers'].apply(parse_layers)

# Win status
df['Is_Win'] = df['Status'].isin(['TP', 'CLOSED']) & (df['PnL'] > 0)
df['Is_Loss'] = df['Status'] == 'SL'
df['Is_Open'] = df['Status'] == 'OPEN'

# Closed trades only
closed_df = df[~df['Is_Open']].copy()

# Parse date/time for time analysis
def parse_hour(date_val, close_time_val):
    try:
        # Try parsing close_time like "TP - 15:42 - 4.885"
        if pd.notna(close_time_val) and '-' in str(close_time_val):
            parts = str(close_time_val).split(' - ')
            if len(parts) >= 2:
                time_part = parts[1]
                if ':' in time_part:
                    hour = int(time_part.split(':')[0])
                    return hour
        # Try date like "2026-01-24 19:05"
        if pd.notna(date_val) and ':' in str(date_val):
            time_part = str(date_val).split(' ')[1]
            hour = int(time_part.split(':')[0])
            return hour
    except: pass
    return None

closed_df['Hour'] = closed_df.apply(lambda r: parse_hour(r['Date'], r['Close_Time']), axis=1)

print("=" * 70)
print("🔍 DEEP ANALYSIS - Trading Results 20-28 Jan 2026")
print("=" * 70)

# ========== 1. OVERALL STATS ==========
print(f"\n📊 OVERALL STATISTICS")
print("-" * 50)
total = len(closed_df)
wins = closed_df['Is_Win'].sum()
losses = closed_df['Is_Loss'].sum()
winrate = wins / total * 100 if total > 0 else 0
total_pnl = closed_df['PnL'].sum()
avg_win = closed_df[closed_df['Is_Win']]['PnL'].mean() if wins > 0 else 0
avg_loss = closed_df[closed_df['Is_Loss']]['PnL'].mean() if losses > 0 else 0

print(f"Total Closed: {total} | Wins: {wins} | Losses: {losses}")
print(f"Winrate: {winrate:.1f}%")
print(f"Total PnL: {total_pnl:+.1f}%")
print(f"Avg Win: {avg_win:+.1f}% | Avg Loss: {avg_loss:.1f}%")
print(f"RR Ratio: {abs(avg_win/avg_loss):.2f} (Avg Win / Avg Loss)")

# ========== 2. BY STRATEGY ==========
print(f"\n📈 ANALYSIS BY STRATEGY")
print("-" * 50)
strategies = ['EMA_PULLBACK', 'SFP', 'LIQ_SWEEP', 'BB_BOUNCE', 'BREAKER_RETEST', 'IE']
for strat in strategies:
    strat_df = closed_df[closed_df['Note'].str.contains(strat, na=False)]
    if len(strat_df) == 0: continue
    
    strat_wins = strat_df['Is_Win'].sum()
    strat_total = len(strat_df)
    strat_wr = strat_wins / strat_total * 100 if strat_total > 0 else 0
    strat_pnl = strat_df['PnL'].sum()
    strat_avg = strat_df['PnL'].mean()
    
    # Breakdown by direction
    long_df = strat_df[strat_df['Signal'] == 'LONG']
    short_df = strat_df[strat_df['Signal'] == 'SHORT']
    
    long_wr = long_df['Is_Win'].sum() / len(long_df) * 100 if len(long_df) > 0 else 0
    short_wr = short_df['Is_Win'].sum() / len(short_df) * 100 if len(short_df) > 0 else 0
    
    trend = "✅" if strat_wr >= 50 and strat_pnl > 0 else "⚠️" if strat_wr >= 45 else "❌"
    print(f"\n{trend} {strat}:")
    print(f"   Trades: {strat_total} | WR: {strat_wr:.1f}% | PnL: {strat_pnl:+.1f}% | Avg: {strat_avg:+.1f}%")
    print(f"   LONG: {len(long_df)} ({long_wr:.0f}%) | SHORT: {len(short_df)} ({short_wr:.0f}%)")

# ========== 3. BY DIRECTION ==========
print(f"\n🔄 ANALYSIS BY DIRECTION")
print("-" * 50)
for direction in ['LONG', 'SHORT']:
    dir_df = closed_df[closed_df['Signal'] == direction]
    if len(dir_df) == 0: continue
    
    dir_wins = dir_df['Is_Win'].sum()
    dir_total = len(dir_df)
    dir_wr = dir_wins / dir_total * 100 if dir_total > 0 else 0
    dir_pnl = dir_df['PnL'].sum()
    
    # Breakdown by checklist
    c3_df = dir_df[dir_df['Checklist_Num'] == 3]
    c2_df = dir_df[dir_df['Checklist_Num'] == 2]
    
    c3_wr = c3_df['Is_Win'].sum() / len(c3_df) * 100 if len(c3_df) > 0 else 0
    c2_wr = c2_df['Is_Win'].sum() / len(c2_df) * 100 if len(c2_df) > 0 else 0
    
    trend = "✅" if dir_wr >= 50 else "⚠️" if dir_wr >= 40 else "❌"
    print(f"\n{trend} {direction}:")
    print(f"   Trades: {dir_total} | WR: {dir_wr:.1f}% | PnL: {dir_pnl:+.1f}%")
    print(f"   With 3/3 checklist: {len(c3_df)} ({c3_wr:.0f}%)")
    print(f"   With 2/3 checklist: {len(c2_df)} ({c2_wr:.0f}%)")

# ========== 4. BY LEVERAGE ==========
print(f"\n⚡ ANALYSIS BY LEVERAGE")
print("-" * 50)
for lev in [10, 15, 50, 100]:
    lev_df = closed_df[closed_df['Leverage'] == lev]
    if len(lev_df) == 0: continue
    
    lev_wins = lev_df['Is_Win'].sum()
    lev_total = len(lev_df)
    lev_wr = lev_wins / lev_total * 100 if lev_total > 0 else 0
    lev_pnl = lev_df['PnL'].sum()
    
    trend = "✅" if lev_wr >= 50 else "⚠️" if lev_wr >= 40 else "❌"
    print(f"{trend} x{lev}: Trades: {lev_total} | WR: {lev_wr:.1f}% | PnL: {lev_pnl:+.1f}%")

# ========== 5. BY LAYERS (SHORT only) ==========
print(f"\n🔍 ANALYSIS BY LAYERS (SHORT signals)")
print("-" * 50)
short_df = closed_df[(closed_df['Signal'] == 'SHORT') & closed_df['Layers_Num'].notna()]
for layer in range(5):
    layer_df = short_df[short_df['Layers_Num'] == layer]
    if len(layer_df) == 0: continue
    
    layer_wins = layer_df['Is_Win'].sum()
    layer_total = len(layer_df)
    layer_wr = layer_wins / layer_total * 100 if layer_total > 0 else 0
    layer_pnl = layer_df['PnL'].sum()
    
    trend = "✅" if layer_wr >= 50 else "⚠️" if layer_wr >= 40 else "❌"
    print(f"{trend} {layer}/4 Layers: Trades: {layer_total} | WR: {layer_wr:.1f}% | PnL: {layer_pnl:+.1f}%")

# ========== 6. BY CHECKLIST ==========
print(f"\n📋 ANALYSIS BY CHECKLIST")
print("-" * 50)
check_df = closed_df[closed_df['Checklist_Num'].notna()]
for chk in [3, 2, 1]:
    chk_df = check_df[check_df['Checklist_Num'] == chk]
    if len(chk_df) == 0: continue
    
    chk_wins = chk_df['Is_Win'].sum()
    chk_total = len(chk_df)
    chk_wr = chk_wins / chk_total * 100 if chk_total > 0 else 0
    chk_pnl = chk_df['PnL'].sum()
    
    trend = "✅" if chk_wr >= 50 else "⚠️" if chk_wr >= 40 else "❌"
    print(f"{trend} {chk}/3 Checklist: Trades: {chk_total} | WR: {chk_wr:.1f}% | PnL: {chk_pnl:+.1f}%")

# ========== 7. WORST TRADES ==========
print(f"\n💀 TOP 10 WORST TRADES")
print("-" * 50)
worst = closed_df.nsmallest(10, 'PnL')[['Coin', 'Signal', 'Leverage', 'Note', 'PnL', 'Checklist', 'Layers']]
for _, row in worst.iterrows():
    print(f"   {row['Coin']} {row['Signal']} x{row['Leverage']}: {row['PnL']:.1f}% | {row['Note'][:20]} | {row['Checklist']} | {row['Layers']}")

# ========== 8. PATTERN ANALYSIS ==========
print(f"\n🔬 PATTERN ANALYSIS")
print("-" * 50)

# High leverage + LONG = Disaster?
high_lev_long = closed_df[(closed_df['Leverage'] >= 50) & (closed_df['Signal'] == 'LONG')]
if len(high_lev_long) > 0:
    hl_wr = high_lev_long['Is_Win'].sum() / len(high_lev_long) * 100
    hl_pnl = high_lev_long['PnL'].sum()
    print(f"🎯 High Leverage (≥50) + LONG: {len(high_lev_long)} trades | WR: {hl_wr:.1f}% | PnL: {hl_pnl:+.1f}%")

# SFP + LONG = Bad?
sfp_long = closed_df[(closed_df['Note'].str.contains('SFP', na=False)) & (closed_df['Signal'] == 'LONG')]
if len(sfp_long) > 0:
    sl_wr = sfp_long['Is_Win'].sum() / len(sfp_long) * 100
    sl_pnl = sfp_long['PnL'].sum()
    print(f"🎯 SFP + LONG: {len(sfp_long)} trades | WR: {sl_wr:.1f}% | PnL: {sl_pnl:+.1f}%")

# SFP + SHORT
sfp_short = closed_df[(closed_df['Note'].str.contains('SFP', na=False)) & (closed_df['Signal'] == 'SHORT')]
if len(sfp_short) > 0:
    ss_wr = sfp_short['Is_Win'].sum() / len(sfp_short) * 100
    ss_pnl = sfp_short['PnL'].sum()
    print(f"🎯 SFP + SHORT: {len(sfp_short)} trades | WR: {ss_wr:.1f}% | PnL: {ss_pnl:+.1f}%")

# EMA_PULLBACK + 3/3 checklist
ema_c3 = closed_df[(closed_df['Note'].str.contains('EMA_PULLBACK', na=False)) & (closed_df['Checklist_Num'] == 3)]
if len(ema_c3) > 0:
    ec_wr = ema_c3['Is_Win'].sum() / len(ema_c3) * 100
    ec_pnl = ema_c3['PnL'].sum()
    print(f"🎯 EMA_PULLBACK + 3/3 Checklist: {len(ema_c3)} trades | WR: {ec_wr:.1f}% | PnL: {ec_pnl:+.1f}%")

# 0/4 layers + SHORT
low_layer_short = closed_df[(closed_df['Layers_Num'] == 0) & (closed_df['Signal'] == 'SHORT')]
if len(low_layer_short) > 0:
    ll_wr = low_layer_short['Is_Win'].sum() / len(low_layer_short) * 100
    ll_pnl = low_layer_short['PnL'].sum()
    print(f"🎯 0/4 Layers + SHORT: {len(low_layer_short)} trades | WR: {ll_wr:.1f}% | PnL: {ll_pnl:+.1f}%")

# ========== 9. RECOMMENDATIONS ==========
print(f"\n" + "=" * 70)
print("💡 KEY FINDINGS & RECOMMENDATIONS")
print("=" * 70)

findings = []

# Check SFP + LONG
if len(sfp_long) > 0 and sl_wr < 40:
    findings.append(f"❌ SFP + LONG thất bại nặng: WR {sl_wr:.0f}%, PnL {sl_pnl:+.0f}%")
    findings.append(f"   → Đề xuất: BLOCK SFP LONG hoặc yêu cầu checklist 3/3")

# Check high leverage
high_lev = closed_df[closed_df['Leverage'] >= 50]
if len(high_lev) > 0:
    hl_wr = high_lev['Is_Win'].sum() / len(high_lev) * 100
    if hl_wr < 40:
        findings.append(f"❌ Leverage ≥50x thảm họa: WR {hl_wr:.0f}%")
        findings.append(f"   → Đề xuất: Giới hạn x15 cho altcoin, x50 chỉ cho BTC/ETH")

# Check LONG overall
long_all = closed_df[closed_df['Signal'] == 'LONG']
if len(long_all) > 0:
    long_wr = long_all['Is_Win'].sum() / len(long_all) * 100
    if long_wr < 40:
        findings.append(f"❌ LONG trades kém: WR {long_wr:.0f}% (Bear market)")
        findings.append(f"   → Đề xuất: Chỉ LONG khi BTC > EMA89 + Checklist 3/3")

# Check 0/4 layers
if len(low_layer_short) > 0 and ll_wr < 55:
    findings.append(f"⚠️ 0/4 Layers SHORT không tốt: WR {ll_wr:.0f}%")
    findings.append(f"   → Đề xuất: Yêu cầu tối thiểu 1/4 layers cho SHORT")

# Good patterns
ema_all = closed_df[closed_df['Note'].str.contains('EMA_PULLBACK', na=False)]
if len(ema_all) > 0:
    ema_wr = ema_all['Is_Win'].sum() / len(ema_all) * 100
    if ema_wr > 50:
        findings.append(f"✅ EMA_PULLBACK là chiến lược tốt nhất: WR {ema_wr:.0f}%")

liq_all = closed_df[closed_df['Note'].str.contains('LIQ_SWEEP', na=False)]
if len(liq_all) > 0:
    liq_wr = liq_all['Is_Win'].sum() / len(liq_all) * 100
    if liq_wr > 55:
        findings.append(f"✅ LIQ_SWEEP hiệu quả cao: WR {liq_wr:.0f}%")

for f in findings:
    print(f)

print(f"\n" + "=" * 70)
print("📋 PROPOSED RULE CHANGES")
print("=" * 70)
print("""
1. LEVERAGE RESTRICTION:
   - x15 mặc định cho altcoin
   - x50 chỉ cho BTC, ETH khi Diamond setup
   - x100 chỉ cho IE Trade (đã thực hiện)

2. SFP STRATEGY FIX:
   - SFP + LONG: Chặn hoàn toàn trong bear market
   - SFP + SHORT: Yêu cầu 2/4 layers minimum
   - Hoặc: Tăng min_score cho SFP lên 55

3. LONG TRADES IN BEAR MARKET:
   - Yêu cầu: BTC > EMA89 H4
   - Yêu cầu: Checklist 3/3 (không chấp nhận 2/3)
   - Yêu cầu: Strategy là EMA_PULLBACK hoặc LIQ_SWEEP

4. LAYERS FILTER (SHORT):
   - Tối thiểu 1/4 layers để vào lệnh
   - 3/4 layers = Diamond tier upgrade

5. CHECKLIST ENFORCEMENT:
   - LONG: Bắt buộc 3/3
   - SHORT: Tối thiểu 2/3

6. STRATEGY PRIORITY:
   - Ưu tiên: EMA_PULLBACK > LIQ_SWEEP > BREAKER_RETEST
   - Hạn chế: SFP (nhất là LONG), BB_BOUNCE
""")

# ========== 10. EXPECTED IMPROVEMENT ==========
print(f"\n📈 EXPECTED IMPROVEMENT")
print("-" * 50)

# Simulate filtering out bad trades
bad_trades = closed_df[
    (closed_df['Signal'] == 'LONG') & 
    ((closed_df['Checklist_Num'] < 3) | (closed_df['Note'].str.contains('SFP', na=False)))
]
remaining = closed_df.drop(bad_trades.index)
if len(remaining) > 0:
    new_wr = remaining['Is_Win'].sum() / len(remaining) * 100
    new_pnl = remaining['PnL'].sum()
    print(f"Nếu block LONG + (2/3 checklist hoặc SFP):")
    print(f"   Bỏ: {len(bad_trades)} trades")
    print(f"   Còn: {len(remaining)} trades | WR: {new_wr:.1f}% | PnL: {new_pnl:+.1f}%")
    print(f"   So với hiện tại: WR {winrate:.1f}% → {new_wr:.1f}% | PnL {total_pnl:+.1f}% → {new_pnl:+.1f}%")

# High leverage filter
bad_lev = closed_df[closed_df['Leverage'] >= 50]
remaining2 = closed_df[closed_df['Leverage'] < 50]
if len(remaining2) > 0:
    new_wr2 = remaining2['Is_Win'].sum() / len(remaining2) * 100
    new_pnl2 = remaining2['PnL'].sum()
    print(f"\nNếu block tất cả leverage ≥50:")
    print(f"   Bỏ: {len(bad_lev)} trades")
    print(f"   Còn: {len(remaining2)} trades | WR: {new_wr2:.1f}% | PnL: {new_pnl2:+.1f}%")
