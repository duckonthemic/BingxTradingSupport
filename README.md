# BingX Trading Support

A professional trading signal detection system with ICT (Inner Circle Trading) entry methodology for cryptocurrency futures on BingX.

## Overview

BingX Trading Support combines advanced price action analysis with institutional-grade signal detection to identify high-probability trade setups in cryptocurrency markets.

**Key Features:**
- Real-time signal scoring system (0-100 confidence scale)
- ICT 4-step entry validation with FVG and MSS detection
- Smart kill zone filtering (London & US sessions only)
- Daily bias confirmation before trading
- Rate limiting (10 alerts/hour) to prevent over-trading
- Multi-timeframe analysis (H1, M5)
- Support for top 50 crypto pairs

## Architecture

### Core Components

```
BingX Alert Bot (Main Process)
├── Alert Manager (90s scan cycle)
│   └── Scoring Engine (EMA, Structure, Momentum)
├── IE Trade Scanner (30s scan cycle)
│   ├── FVG Detector (H1 candles)
│   ├── MSS Detector (M5 candles)
│   ├── Bias Manager (Daily confirmation)
│   └── Entry Calculator (SL/TP)
├── Real-time Engine (Optional WebSocket)
│   └── Volume Spike Detection
└── Telegram Bot Interface
    └── Commands: /dbias, /iestatus, /iestart
```

### Data Pipeline

```
1. Data Ingestion
   ├── BingX REST API: Historical candles (H1, M5)
   └── WebSocket (Optional): Real-time price ticks

2. Signal Detection
   ├── Alert Manager: Scan all 50 coins
   │   ├── EMA Trend Analysis
   │   ├── Market Structure (HH/HL, LL/LH)
   │   ├── Momentum (RSI, WaveTrend, Volume)
   │   └── Score Calculation (0-100)
   │
   └── IE Trade Scanner: Monitor 15 premium coins
       ├── H1 FVG Detection (Premium/Discount zones)
       ├── Price Entry Monitoring
       ├── M5 MSS Confirmation
       └── Trade Setup Calculation

3. Alert Filtering
   ├── Kill Zone Check (London/NY sessions only)
   ├── Daily Bias Confirmation
   ├── Position Limit (max open)
   └── Rate Limiting (10/hour)

4. Alert Delivery
   ├── Telegram Chat
   ├── Google Sheets (optional)
   └── Trade Journal
```

## Scoring System

### Signal Confidence Levels

| Tier | Score | Criteria | Action |
|------|-------|----------|--------|
| Diamond | >= 80 | EMA + Structure + Momentum confirmed | Send immediately |
| Gold | 55-79 | Strong structural setup + momentum | Send in kill zone |
| Silver | 40-54 | Partial setup, lower confidence | Watchlist only |
| Below | < 40 | Insufficient signals | Skip |

### Scoring Components

**Context (20 points)**
- EMA Trend: Price above/below EMA34, EMA89
- Market Structure: HH/HL (Uptrend) or LL/LH (Downtrend)

**Trigger (30 points)** - At least 1 required
- Sweep & Flush (SFP): Price sweeps support/resistance then reverses
- Retest Zone: Retests previous breakout or Fibonacci zone

**Momentum (50 points)**
- RSI/WaveTrend strength
- Volume spike (1.5-3x average)
- Entry candle quality

## IE Trade Module

### 4-Step Entry Methodology

1. **Daily Bias**: Set via /dbias B (LONG) or /dbias S (SHORT)
2. **H1 FVG Detection**: Find Fair Value Gap in Premium/Discount zone
3. **M5 MSS Confirmation**: Wait for Multiple Supply Side test
4. **M5 FVG Entry**: Enter at FVG with predefined SL/TP

### Kill Zone Configuration

- London: 14:00-17:00 VN (GMT+7)
- US Open: 19:00-23:00 VN (GMT+7)
- Other times: Setups stored as pending

### Status Monitoring

Use /iestatus to see:
- Current bias and expiration
- FVGs detected and monitored
- MSS confirmations
- Ready setups
- Position count

## Installation

### Requirements

- Python 3.11+
- Docker & Docker Compose
- BingX API credentials
- Telegram Bot token

### Quick Start

```bash
# Clone repository
git clone https://github.com/duckonthemic/BingxTradingSupport.git
cd BingxTradingSupport

# Configure environment
cp .env.example .env
# Edit .env with your credentials

# Start with Docker
docker-compose up -d

# View logs
docker logs -f bingx-zone-alert-bot
```

### Configuration

Key settings in src/config.py:

```python
# Alert Manager
SCAN_INTERVAL = 90
ALERT_RATE_LIMIT = 10
MAX_ALERTS_PER_SCAN = 5

# IE Trade
SCAN_INTERVAL_IE = 30
MAX_OPEN_POSITIONS = 3
PREMIUM_THRESHOLD = 0.618

# Kill Zones (VN time)
LONDON_START = 14
LONDON_END = 17
NY_START = 19
NY_END = 23
```

## Telegram Commands

| Command | Description |
|---------|-------------|
| /dbias B | Set LONG bias for 24h |
| /dbias S | Set SHORT bias for 24h |
| /dbias | Show current bias |
| /iestatus | Show IE Trade status |
| /iestart | Start IE Trade scanner |
| /iestop | Stop IE Trade scanner |

## Deployment

### Docker Compose

```yaml
services:
  bot:
    image: notibingxbot
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
      - BINGX_API_KEY=${BINGX_API_KEY}
      - BINGX_SECRET_KEY=${BINGX_SECRET_KEY}
      - IE_TRADE_ENABLED=true
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
```

### Environment Variables

```
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BINGX_API_KEY=your_api_key
BINGX_SECRET_KEY=your_secret_key
IE_TRADE_ENABLED=true
```

## Performance

### Backtesting Results (90 days: Oct 2025 - Jan 2026)

- Total Trades: 92
- Win Rate: 76.1%
- Profit: +$785.93 (+78.59%)
- Max Drawdown: 0.4%
- LONG: 22 trades (63.6% WR)
- SHORT: 70 trades (80.0% WR)

### Key Metrics

- Strategy profitable in bull and bear markets
- 80% of profit from remaining trades (not top 5)
- Smooth equity curve with no hockey stick effect

## License

Proprietary - See LICENSE file

---

Version 3.6 | Last Updated January 2026
| /dbias | Show current bias |
| /iestatus | Show IE Trade status |
| /iestart | Start IE Trade scanner |
| /iestop | Stop IE Trade scanner |

## Deployment

### Docker Compose

```yaml
services:
  bot:
    image: notibingxbot
    environment:
      - TELEGRAM_TOKEN=${TELEGRAM_TOKEN}
      - BINGX_API_KEY=${BINGX_API_KEY}
      - BINGX_SECRET_KEY=${BINGX_SECRET_KEY}
      - IE_TRADE_ENABLED=true
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    restart: unless-stopped
```

### Environment Variables

```
TELEGRAM_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
BINGX_API_KEY=your_api_key
BINGX_SECRET_KEY=your_secret_key
IE_TRADE_ENABLED=true
```

## Performance

### Backtesting Results (90 days: Oct 2025 - Jan 2026)

- Total Trades: 92
- Win Rate: 76.1%
- Profit: +$785.93 (+78.59%)
- Max Drawdown: 0.4%
- LONG: 22 trades (63.6% WR)
- SHORT: 70 trades (80.0% WR)

### Key Metrics

- Strategy profitable in bull and bear markets
- 80% of profit from remaining trades (not top 5)
- Smooth equity curve with no hockey stick effect

## License

Proprietary - See LICENSE file

---

Version 3.6 | Last Updated January 2026

```
Strategy Points:          Confirmation Points:
├─ PUMP_FADE   +30       ├─ RSI Divergence    +15
├─ BB_BOUNCE   +30       ├─ Fib Golden Pocket +15
├─ LIQ_SWEEP   +25       ├─ Volume Spike      +10
├─ SFP         +25       ├─ WaveTrend         +10
├─ BREAKOUT    +20       └─ OB Confluence     +10
└─ EMA_ALIGN   +20       

Penalties:
└─ Counter-Trend  -25
```

---

## 📱 Telegram Commands

| Command | Description |
|---------|-------------|
| `/status` | Bot status & circuit breaker |
| `/stats` | Session statistics |
| `/ana BTC` | Quick analysis of any coin |
| `/btc` | BTC mood check (dump/pump) |
| `/coins` | Top 15 by volume |
| `/news` | Economic calendar (next 48h) |
| `/session` | Trading sessions status |
| `/pause` `/resume` | Control scanning |

---

## 📋 Google Sheets Journal (18 Columns)

| Col | Field | Auto-Updated |
|-----|-------|--------------|
| A-E | No, Date, Coin, Signal, Leverage | ✅ On entry |
| F-H | Entry, SL, TP | ✅ On entry |
| I-J | Price Now, PnL % | ✅ Every 2 min |
| K-L | Status, Close Time | ✅ On TP/SL hit |
| M | Note (Strategy) | ✅ On entry |
| N | End Trade (Checkbox) | ⬜ Manual |
| O | User Note | ⬜ Manual |
| P-R | Grade, Layers, Checklist | ✅ On entry |

---

## 🌐 Trading Sessions

Bot automatically sends notifications when major sessions open:

| Session | UTC Time | Description |
|---------|----------|-------------|
| 🌏 Asia | 00:00 | Tokyo, Hong Kong, Singapore |
| 🌍 Europe | 07:00 | London open |
| 🌎 US | 13:30 | New York open |

---

## 🔧 Configuration

### Environment Variables (.env)

```env
# Required
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Google Sheets (Optional)
GOOGLE_SHEET_ID=your_spreadsheet_id

# Redis (Docker handles this)
REDIS_HOST=redis
REDIS_PORT=6379

# Mode
REALTIME_MODE=true  # WebSocket (recommended)
```

### Credentials

1. **Telegram**: Create bot via [@BotFather](https://t.me/BotFather)
2. **Google Sheets**: 
   - Create project at [Google Cloud Console](https://console.cloud.google.com)
   - Enable Sheets API
   - Create Service Account → Download `credentials.json`
   - Share spreadsheet with service account email

---

## 📁 Project Structure

```
notibingxbot/
├── main.py                 # Entry point
├── src/
│   ├── analysis/           # Indicators, strategies, scoring
│   ├── notification/       # Telegram, alerts, sessions
│   ├── storage/            # Redis, Google Sheets
│   ├── ingestion/          # BingX API, WebSocket
│   ├── risk/               # Risk management
│   └── context/            # Market context
├── docs/                   # Documentation
└── docker-compose.yml      # Deployment
```

---

## 📖 Documentation

- [SYSTEM.md](docs/SYSTEM.md) - Full system documentation & workflow
- [TECHNICAL_SPECIFICATION.md](docs/TECHNICAL_SPECIFICATION.md) - Technical details

---

## ⚠️ Disclaimer

This bot is for educational purposes only. Trading cryptocurrencies involves substantial risk of loss. Past performance does not guarantee future results.

---

## 📄 License

MIT License
