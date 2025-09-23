# Multi-Symbol Futures Trading Bot with Smart Money Concepts

A sophisticated trading bot that analyzes **ALL cryptocurrency futures pairs** using Smart Money Concepts (SMC) and sends trading signals to a Telegram channel.

## Features

### Complete Smart Money Concepts Implementation
**Core Concepts:**
- **Market Structure Analysis**: Higher highs/lower lows and trend direction
- **Order Blocks**: Institutional order placement areas
- **Fair Value Gaps (FVG)**: Price gaps that need to be filled
- **Liquidity Zones**: Areas of high trading activity
- **Supply & Demand Zones**: Key support and resistance levels
- **Inducement Patterns**: Stop hunting before significant moves

**Advanced Concepts:**
- **Liquidity Grabs**: Stop hunts above/below significant levels
- **Breaker Blocks**: Failed order blocks that flip polarity
- **Mitigation Blocks**: Order block retests for entries
- **Change of Character (ChoCh)**: Trend reversal signals
- **Break of Structure (BoS)**: Trend continuation signals
- **Displacement**: Strong institutional moves (2x average range)
- **Premium/Discount**: Fibonacci-based price positioning

### Trading Features
- **Multi-Symbol Monitoring**: Analyzes top 50 cryptocurrency futures pairs by volume
- **Real-time Analysis**: Processes all symbols every 4 hours with smart money analysis
- **Confluence-based Signals**: Generates signals only when multiple SMC concepts align
- **Dynamic Risk Management**: Calculates leverage, stop loss, and take profit based on each token's volatility
- **Confidence Scoring**: Shows percentage match of smart money concepts
- **Concurrent Analysis**: Analyzes up to 10 symbols simultaneously for optimal performance

### Telegram Integration
- **Formatted Signals**: Professional signal messages with all trading details
- **Real-time Notifications**: Instant signal delivery to your channel
- **Status Updates**: Bot status and performance notifications

## Installation

1. **Clone and Setup**
   ```bash
   cd /Users/vv/PycharmProjects/futuresbot
   pip install -r requirements.txt
   ```

2. **Environment Configuration**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` file with your credentials:
   ```
   TELEGRAM_BOT_TOKEN=your_telegram_bot_token
   TELEGRAM_CHANNEL_ID=@your_channel_or_chat_id
   BINANCE_API_KEY=your_binance_api_key
   BINANCE_SECRET_KEY=your_binance_secret_key
   ```

3. **Telegram Bot Setup**
   - Create a bot via @BotFather on Telegram
   - Get your bot token
   - Add the bot to your channel as an administrator
   - Get your channel ID (use @userinfobot)

## Usage

### Start the Bot
```bash
python futures_bot.py
```

### Configuration Options
Edit `config.py` to customize:
- Update intervals (default: 15 minutes)
- Risk parameters
- Leverage limits
- Smart money thresholds

## Smart Money Signal Example

```
🟢 SMART MONEY SIGNAL 🟢

🎯 Symbol: XRP/USDT
📈 Direction: BUY
📊 Confidence: 87.5%
⚡ Leverage: 4x

💰 Trade Setup:
🎯 Entry: $0.5234
🛑 Stop Loss: $0.5156
🎁 Take Profit: $0.5390
📈 Risk/Reward: 1:2.0

🧠 Smart Money Concepts Matched:
• Bullish Market Structure
• Bullish Order Block
• Near Demand Zone
• Bearish Inducement (Bullish Signal)

⏰ Time: 2025-09-12 14:30:00 UTC
```

## How It Works

### 1. Data Collection
- Fetches real-time XRP/USDT data from Binance
- Calculates historical volatility for risk management
- Analyzes orderbook for liquidity information

### 2. Smart Money Analysis
- **Market Structure**: Analyzes price action for trend direction
- **Order Blocks**: Identifies areas where large orders were executed
- **Fair Value Gaps**: Detects price imbalances that need correction
- **Liquidity Analysis**: Finds areas of concentrated trading activity
- **Confluence Detection**: Only signals when multiple concepts align

### 3. Signal Generation
- Requires minimum 2 confluences for signal generation
- Calculates confidence percentage based on matched concepts
- Determines optimal entry, stop loss, and take profit levels
- Adjusts leverage based on market volatility and confidence

### 4. Risk Management
- Dynamic stop loss based on Average True Range (ATR)
- Leverage adjustment for high volatility periods
- Position sizing based on risk percentage
- Risk/reward ratio optimization

## Signal Quality Features

- **Confluence Requirement**: Minimum 2 smart money concepts must align
- **Recency Check**: Only sends signals less than 5 minutes old
- **Natural Signal Spacing**: No artificial cooldown needed - 5-minute freshness and unique signals prevent spam
- **Volatility Adjustment**: Reduces leverage during high volatility periods

## Monitoring and Logs

- All activities logged to `futures_bot.log`
- Real-time status updates in console
- Telegram status messages for important events

## Safety Features

- **Paper Trading Ready**: No actual trading execution
- **Error Handling**: Robust error handling and recovery
- **Connection Monitoring**: Automatic reconnection attempts
- **Data Validation**: Comprehensive data quality checks

## Disclaimer

This bot is for educational and analysis purposes only. It does not execute actual trades. Always:
- Do your own research
- Test thoroughly before any live trading
- Never risk more than you can afford to lose
- Understand that past performance doesn't guarantee future results

## Support

For issues or questions:
1. Check the log files for error details
2. Verify your API credentials and permissions
3. Ensure your Telegram bot has proper channel access
