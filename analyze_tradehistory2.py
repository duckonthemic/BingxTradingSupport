"""
Deep Analysis for TradeHistory2.csv
"""
import pandas as pd
import numpy as np

FILE_PATH = "TradeHistory2.csv"

df = pd.read_csv(FILE_PATH, skiprows=4)

# Normalize columns
cols = ['No', 'Date', 'Coin', 'Signal', 'Leverage', 'Entry', 'SL', 'TP',
        'Price_Now', 'PnL_Pct', 'Status', 'Close_Time', 'Note', 'End_Trade',
        'User_Note', 'Grade', 'Layers', 'Checklist']

# Pad extra columns
df.columns = cols + [f'Col_{i}' for i in range(len(cols), len(df.columns))]

# Basic cleanup
df = df[df['Status'].notna() & (df['Status'] != '')]
df = df[df['Coin'].notna()]

# Parse PnL
def parse_pnl(val):
    try:
        return float(str(val).replace('%', '').replace(',', ''))
    except Exception:
        return 0.0

df['PnL'] = df['PnL_Pct'].apply(parse_pnl)

# Leverage
try:
    df['Leverage'] = pd.to_numeric(df['Leverage'], errors='coerce').fillna(15).astype(int)
except Exception:
    df['Leverage'] = 15

# Checklist / Layers

def parse_num(val):
    if pd.isna(val) or val == '':
        return None
    try:
        return int(str(val).split('/')[0])
    except Exception:
        return None

df['Checklist_Num'] = df['Checklist'].apply(parse_num)
df['Layers_Num'] = df['Layers'].apply(parse_num)

# Win/Loss flags
# TP, CLOSED with positive PnL -> win
# SL -> loss
# OPEN -> exclude

df['Is_Win'] = df['Status'].isin(['TP', 'CLOSED']) & (df['PnL'] > 0)
df['Is_Loss'] = df['Status'] == 'SL'
df['Is_Open'] = df['Status'] == 'OPEN'

closed_df = df[~df['Is_Open']].copy()

print("=" * 70)
print("📊 TradeHistory2 Deep Analysis")
print("=" * 70)

# Overall

total = len(closed_df)
wins = closed_df['Is_Win'].sum()
losses = closed_df['Is_Loss'].sum()
winrate = wins / total * 100 if total > 0 else 0
pnl_total = closed_df['PnL'].sum()

avg_win = closed_df[closed_df['Is_Win']]['PnL'].mean() if wins > 0 else 0
avg_loss = closed_df[closed_df['Is_Loss']]['PnL'].mean() if losses > 0 else 0

print(f"Total Closed: {total} | Wins: {wins} | Losses: {losses}")
print(f"Winrate: {winrate:.1f}%")
print(f"Total PnL: {pnl_total:+.1f}%")
print(f"Avg Win: {avg_win:+.1f}% | Avg Loss: {avg_loss:.1f}% | RR: {abs(avg_win/avg_loss):.2f}")

# By Strategy
print("\n📈 By Strategy")
print("-" * 50)
strategies = ['EMA_PULLBACK', 'SFP', 'LIQ_SWEEP', 'BB_BOUNCE', 'BREAKER_RETEST', 'IE trade']
for strat in strategies:
    strat_df = closed_df[closed_df['Note'].str.contains(strat, na=False)]
    if len(strat_df) == 0:
        continue
    strat_wins = strat_df['Is_Win'].sum()
    strat_total = len(strat_df)
    strat_wr = strat_wins / strat_total * 100 if strat_total > 0 else 0
    strat_pnl = strat_df['PnL'].sum()
    long_df = strat_df[strat_df['Signal'] == 'LONG']
    short_df = strat_df[strat_df['Signal'] == 'SHORT']
    long_wr = long_df['Is_Win'].sum() / len(long_df) * 100 if len(long_df) > 0 else 0
    short_wr = short_df['Is_Win'].sum() / len(short_df) * 100 if len(short_df) > 0 else 0

    trend = "✅" if strat_wr >= 50 and strat_pnl > 0 else "⚠️" if strat_wr >= 45 else "❌"
    print(f"\n{trend} {strat}:")
    print(f"   Trades: {strat_total} | WR: {strat_wr:.1f}% | PnL: {strat_pnl:+.1f}%")
    print(f"   LONG: {len(long_df)} ({long_wr:.0f}%) | SHORT: {len(short_df)} ({short_wr:.0f}%)")

# By Direction
print("\n🔄 By Direction")
print("-" * 50)
for direction in ['LONG', 'SHORT']:
    dir_df = closed_df[closed_df['Signal'] == direction]
    if len(dir_df) == 0:
        continue
    dir_wins = dir_df['Is_Win'].sum()
    dir_total = len(dir_df)
    dir_wr = dir_wins / dir_total * 100 if dir_total > 0 else 0
    dir_pnl = dir_df['PnL'].sum()
    print(f"{direction}: Trades {dir_total} | WR {dir_wr:.1f}% | PnL {dir_pnl:+.1f}%")

# By Leverage
print("\n⚡ By Leverage")
print("-" * 50)
for lev in sorted(closed_df['Leverage'].unique()):
    lev_df = closed_df[closed_df['Leverage'] == lev]
    lev_wins = lev_df['Is_Win'].sum()
    lev_total = len(lev_df)
    lev_wr = lev_wins / lev_total * 100 if lev_total > 0 else 0
    lev_pnl = lev_df['PnL'].sum()
    trend = "✅" if lev_wr >= 50 and lev_pnl > 0 else "⚠️" if lev_wr >= 45 else "❌"
    print(f"{trend} x{lev}: Trades {lev_total} | WR {lev_wr:.1f}% | PnL {lev_pnl:+.1f}%")

# By Checklist
print("\n📋 By Checklist")
print("-" * 50)
check_df = closed_df[closed_df['Checklist_Num'].notna()]
for chk in [3, 2, 1]:
    chk_df = check_df[check_df['Checklist_Num'] == chk]
    if len(chk_df) == 0:
        continue
    chk_wins = chk_df['Is_Win'].sum()
    chk_total = len(chk_df)
    chk_wr = chk_wins / chk_total * 100 if chk_total > 0 else 0
    chk_pnl = chk_df['PnL'].sum()
    trend = "✅" if chk_wr >= 50 and chk_pnl > 0 else "⚠️" if chk_wr >= 45 else "❌"
    print(f"{trend} {chk}/3: Trades {chk_total} | WR {chk_wr:.1f}% | PnL {chk_pnl:+.1f}%")

# Worst trades
print("\n💀 Top 10 Worst Trades")
print("-" * 50)
worst = closed_df.nsmallest(10, 'PnL')[['Coin','Signal','Leverage','Note','PnL','Checklist','Layers']]
for _, row in worst.iterrows():
    print(f"{row['Coin']} {row['Signal']} x{row['Leverage']}: {row['PnL']:.1f}% | {row['Note']} | {row['Checklist']} | {row['Layers']}")
