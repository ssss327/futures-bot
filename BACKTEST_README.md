# Qudo SMC Backtester

A comprehensive backtesting tool for the Qudo Smart Money Concepts strategy.

## Features

✅ **Full Qudo Strategy Testing**
- HTF (4H) Order Flow context validation
- MTF (15m) Liquidity grab detection (Asian Low/High, PDL/PDH)
- MTF (15m) BOS confirmation after liquidity capture
- MTF (15m) Fresh POI identification (Order Block, Breaker, FVG)
- LTF (1m) CHoCH confirmation
- Dynamic SL/TP based on liquidity pools

✅ **Realistic Simulation**
- Bar-by-bar analysis on 15m timeframe
- Precise SL/TP execution on 1m data
- Proper sequential validation (HTF → MTF → LTF)
- Timeout handling for incomplete trades

✅ **Detailed Statistics**
- Win rate, profit factor, max drawdown
- Average duration, R:R ratios
- Breakdown by HTF context (Bullish/Bearish)
- Breakdown by liquidity type (Asian/PDL/PDH)
- Breakdown by POI type (OB/Breaker/FVG)

## Usage

### Command Line

```bash
python qudo_backtester.py \
  --symbols "BTC/USDT,ETH/USDT,SOL/USDT" \
  --start "2024-01-01" \
  --end "2024-03-31" \
  --output "results.csv"
```

### Python Script

```python
import asyncio
from qudo_backtester import QudoBacktester

async def run_test():
    backtester = QudoBacktester()
    
    stats = await backtester.run_backtest(
        symbols=['BTC/USDT', 'ETH/USDT'],
        start_date='2024-01-01',
        end_date='2024-03-31',
        lookback_days=60
    )
    
    backtester.save_results('my_backtest.csv')
    
    print(f"Win Rate: {stats.win_rate:.2f}%")
    print(f"Total Return: {stats.total_return:.2f}%")
    print(f"Profit Factor: {stats.profit_factor:.2f}")

asyncio.run(run_test())
```

## Output

The backtester generates:

1. **CSV File** with detailed results for each trade:
   - Entry/exit timestamps and prices
   - SL/TP levels
   - HTF context, liquidity type, POI type
   - Outcome (WIN/LOSS/TIMEOUT)
   - Return %, P&L, duration

2. **Console Statistics** including:
   - Overall performance metrics
   - Win/loss breakdown
   - HTF context analysis
   - Liquidity and POI type distributions

## Example Output

```
============================================================
QUDO SMC BACKTEST RESULTS
============================================================
Total Signals: 45
Total Trades: 42
Wins: 28 | Losses: 12 | Timeouts: 3
Win Rate: 66.67%
Total Return: 145.30%
Avg Return: 3.46%
Avg Profit: 5.20% | Avg Loss: -2.10%
Profit Factor: 2.48
Max Drawdown: 8.50%
Avg Duration: 12.3h
Best Trade: 12.50% | Worst Trade: -4.20%
Avg R:R Ratio: 2.10

------------------------------------------------------------
HTF CONTEXT BREAKDOWN
------------------------------------------------------------
Bullish: 25 | Bearish: 20

------------------------------------------------------------
LIQUIDITY TYPE BREAKDOWN
------------------------------------------------------------
Asian Low: 15 | Asian High: 12
PDL: 10 | PDH: 8

------------------------------------------------------------
POI TYPE BREAKDOWN
------------------------------------------------------------
Order Block: 35 | Breaker: 8 | FVG: 2
============================================================
```

## Requirements

- All Qudo strategy timeframes: 4H, 15m, 1m
- Sufficient historical data (60+ days recommended)
- Valid Binance API credentials

## Notes

- The backtester uses 1m data for precise SL/TP execution
- Trades timeout after 200 bars (~50 hours) if neither SL nor TP is hit
- Results include slippage and realistic entry timing
- Old backtester saved as `backtester_old.py` for reference

