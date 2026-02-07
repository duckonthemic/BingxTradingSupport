"""Quick analysis of TradeHistory3.csv for rule optimization."""
import csv
import re

trades = []
with open('TradeHistory3.csv', 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

for row in rows[5:]:  # Skip header
    if len(row) < 14 or not row[0].strip().isdigit():
        continue
    try:
        no = int(row[0].strip())
        coin = row[2].strip()
        signal = row[3].strip()  # LONG/SHORT
        leverage = int(row[4].strip())
        pnl_str = row[9].strip().replace('%', '')
        pnl = float(pnl_str)
        status = row[10].strip()
        strategy = row[12].strip()
        grade = row[15].strip() if len(row) > 15 else ''
        layers = row[16].strip() if len(row) > 16 else ''
        checklist = row[17].strip() if len(row) > 17 else ''
        
        # Parse checklist score
        cl_score = 0
        if '/' in checklist:
            try:
                cl_score = int(checklist.split('/')[0])
            except:
                cl_score = 0
        
        # Normalize strategy
        strat = strategy.split('|')[0].strip().split(' ')[0].strip()
        if 'IE' in strat or 'IE' in strategy:
            strat = 'IE'
        
        if signal in ('LONG', 'SHORT') and status:
            trades.append({
                'no': no, 'coin': coin, 'signal': signal, 'leverage': leverage,
                'pnl': pnl, 'status': status, 'strategy': strat, 'grade': grade,
                'layers': layers, 'checklist': checklist, 'cl_score': cl_score
            })
    except (ValueError, IndexError):
        continue

print(f"Total closed trades: {len(trades)}")
print()

# By strategy + direction
combos = {}
for t in trades:
    key = f"{t['strategy']} {t['signal']}"
    if key not in combos:
        combos[key] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
    combos[key]['trades'] += 1
    combos[key]['pnl'] += t['pnl']
    if t['pnl'] > 0:
        combos[key]['wins'] += 1

print("=" * 80)
print(f"{'Strategy+Dir':<25} {'Trades':>6} {'WR%':>8} {'PnL%':>12}")
print("=" * 80)
for key in sorted(combos.keys()):
    d = combos[key]
    wr = (d['wins'] / d['trades'] * 100) if d['trades'] > 0 else 0
    print(f"{key:<25} {d['trades']:>6} {wr:>7.1f}% {d['pnl']:>+11.1f}%")

# By direction
print("\n--- By Direction ---")
for direction in ['LONG', 'SHORT']:
    dt = [t for t in trades if t['signal'] == direction]
    if dt:
        wins = sum(1 for t in dt if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in dt)
        print(f"{direction}: {len(dt)} trades, {wins/len(dt)*100:.1f}% WR, {total_pnl:+.1f}% PnL")

# By leverage
print("\n--- By Leverage ---")
lev_groups = {}
for t in trades:
    lev = t['leverage']
    if lev not in lev_groups:
        lev_groups[lev] = {'trades': 0, 'wins': 0, 'pnl': 0.0}
    lev_groups[lev]['trades'] += 1
    lev_groups[lev]['pnl'] += t['pnl']
    if t['pnl'] > 0:
        lev_groups[lev]['wins'] += 1

for lev in sorted(lev_groups.keys()):
    d = lev_groups[lev]
    wr = (d['wins'] / d['trades'] * 100) if d['trades'] > 0 else 0
    print(f"x{lev}: {d['trades']} trades, {wr:.1f}% WR, {d['pnl']:+.1f}% PnL")

# By checklist
print("\n--- By Checklist ---")
for cl in [0, 1, 2, 3]:
    ct = [t for t in trades if t['cl_score'] == cl]
    if ct:
        wins = sum(1 for t in ct if t['pnl'] > 0)
        total_pnl = sum(t['pnl'] for t in ct)
        print(f"{cl}/3: {len(ct)} trades, {wins/len(ct)*100:.1f}% WR, {total_pnl:+.1f}% PnL")

# By checklist + direction
print("\n--- By Checklist + Direction ---")
for cl in [0, 1, 2, 3]:
    for d in ['LONG', 'SHORT']:
        ct = [t for t in trades if t['cl_score'] == cl and t['signal'] == d]
        if ct:
            wins = sum(1 for c in ct if c['pnl'] > 0)
            total_pnl = sum(c['pnl'] for c in ct)
            print(f"{cl}/3 {d}: {len(ct)} trades, {wins/len(ct)*100:.1f}% WR, {total_pnl:+.1f}% PnL")

# x500 analysis
print("\n--- x500 Leverage Trades ---")
x500 = [t for t in trades if t['leverage'] >= 500]
for t in x500:
    print(f"  #{t['no']} {t['coin']} {t['signal']} x{t['leverage']} {t['strategy']} CL:{t['checklist']} PnL:{t['pnl']:+.1f}%")

# SFP LONG breakdown
print("\n--- SFP LONG Detailed ---")
sfp_long = [t for t in trades if t['strategy'] == 'SFP' and t['signal'] == 'LONG']
if sfp_long:
    wins = [t for t in sfp_long if t['pnl'] > 0]
    losses = [t for t in sfp_long if t['pnl'] <= 0]
    print(f"  Wins: {len(wins)}, avg PnL: {sum(t['pnl'] for t in wins)/len(wins) if wins else 0:+.1f}%")
    print(f"  Losses: {len(losses)}, avg PnL: {sum(t['pnl'] for t in losses)/len(losses) if losses else 0:+.1f}%")
    # Worst SFP LONG
    worst = sorted(sfp_long, key=lambda t: t['pnl'])[:5]
    for t in worst:
        print(f"  WORST: #{t['no']} {t['coin']} x{t['leverage']} CL:{t['checklist']} PnL:{t['pnl']:+.1f}%")

# What the old rules would block
print("\n" + "=" * 80)
print("SIMULATION: What old v3 rules would do")
print("=" * 80)

blocked = []
passed = []
for t in trades:
    strat = t['strategy']
    direction = t['signal']
    cl = t['cl_score']
    
    # Strategy-direction check
    block = False
    reason = ''
    
    if strat == 'SFP' and direction == 'LONG':
        block = True; reason = 'SFP LONG blocked'
    elif strat == 'EMA_PULLBACK' and direction == 'LONG':
        block = True; reason = 'EMA_PULLBACK LONG blocked'
    elif strat == 'BB_BOUNCE' and direction == 'LONG':
        block = True; reason = 'BB_BOUNCE LONG blocked'
    elif strat in ('LIQ_SWEEP', 'BREAKER_RETEST', 'IE') and direction == 'LONG' and cl < 3:
        block = True; reason = f'{strat} LONG needs 3/3 CL (has {cl}/3)'
    
    # Checklist check
    if not block:
        if direction == 'LONG' and cl < 3:
            block = True; reason = f'LONG needs 3/3 CL (has {cl}/3)'
        elif direction == 'SHORT' and cl < 2:
            block = True; reason = f'SHORT needs 2/3 CL (has {cl}/3)'
    
    if block:
        blocked.append((t, reason))
    else:
        passed.append(t)

print(f"\nBlocked: {len(blocked)} ({len(blocked)/len(trades)*100:.1f}%)")
print(f"Passed: {len(passed)} ({len(passed)/len(trades)*100:.1f}%)")

blocked_pnl = sum(t['pnl'] for t, _ in blocked)
passed_pnl = sum(t['pnl'] for t in passed)
blocked_wins = sum(1 for t, _ in blocked if t['pnl'] > 0)
passed_wins = sum(1 for t in passed if t['pnl'] > 0)

print(f"\nBlocked trades: PnL={blocked_pnl:+.1f}%, WR={blocked_wins/len(blocked)*100:.1f}%")
print(f"Passed trades: PnL={passed_pnl:+.1f}%, WR={passed_wins/len(passed)*100:.1f}%")

# Block breakdown
print("\n--- Block Reasons ---")
reasons = {}
for t, r in blocked:
    if r not in reasons:
        reasons[r] = {'count': 0, 'pnl': 0}
    reasons[r]['count'] += 1
    reasons[r]['pnl'] += t['pnl']
for r in sorted(reasons.keys()):
    d = reasons[r]
    print(f"  {r}: {d['count']} trades, {d['pnl']:+.1f}% PnL blocked")

# What if we only blocked x500 + SFP LONG?
print("\n" + "=" * 80)
print("ALTERNATIVE: Only block x500 lev + SFP LONG")
print("=" * 80)
alt_blocked = []
alt_passed = []
for t in trades:
    block = False
    if t['strategy'] == 'SFP' and t['signal'] == 'LONG':
        block = True
    if t['leverage'] >= 500:
        block = True
    if block:
        alt_blocked.append(t)
    else:
        alt_passed.append(t)

alt_b_pnl = sum(t['pnl'] for t in alt_blocked)
alt_p_pnl = sum(t['pnl'] for t in alt_passed)
print(f"Blocked: {len(alt_blocked)}, PnL={alt_b_pnl:+.1f}%")
print(f"Passed: {len(alt_passed)}, PnL={alt_p_pnl:+.1f}%")
if alt_passed:
    print(f"Passed WR: {sum(1 for t in alt_passed if t['pnl'] > 0)/len(alt_passed)*100:.1f}%")
