"""
Qudo SMC Strategy Backtester

Tests the full Qudo strategy:
1. HTF (4H) Order Flow context
2. MTF (15m) Liquidity grab + BOS
3. MTF (15m) Fresh POI in discount/premium
4. LTF (1m) CHoCH confirmation
5. SL/TP based on liquidity pools
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import ccxt.async_support as ccxt

from qudo_smc_strategy import QudoSMCStrategy, QudoSignal
from config import Config


@dataclass
class QudoBacktestResult:
    """Individual backtest result for a Qudo signal"""
    timestamp: datetime
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    htf_context: str
    liquidity_grabbed: str
    poi_type: str
    outcome: str  # 'WIN', 'LOSS', 'TIMEOUT'
    exit_price: float
    exit_timestamp: datetime
    return_pct: float
    profit_loss: float
    duration_hours: float
    bars_to_exit: int
    rr_ratio: float


@dataclass
class QudoBacktestStatistics:
    """Overall backtest statistics"""
    total_signals: int
    total_trades: int
    wins: int
    losses: int
    timeouts: int
    win_rate: float
    avg_return: float
    total_return: float
    avg_profit: float
    avg_loss: float
    profit_factor: float
    max_drawdown: float
    avg_duration_hours: float
    best_trade: float
    worst_trade: float
    avg_rr_ratio: float
    
    # HTF context breakdown
    bullish_signals: int
    bearish_signals: int
    
    # Liquidity type breakdown
    asian_low_grabs: int
    asian_high_grabs: int
    pdl_grabs: int
    pdh_grabs: int
    
    # POI type breakdown
    order_block_pois: int
    breaker_pois: int
    fvg_pois: int


class QudoBacktester:
    """
    Backtester for Qudo SMC Strategy
    
    Simulates realistic trading by:
    - Fetching 4H, 15m, and 1m data
    - Running strategy bar-by-bar
    - Waiting for LTF confirmation before entry
    - Tracking exit via SL/TP hits
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.strategy = QudoSMCStrategy()
        self.exchange = ccxt.binance({
            'enableRateLimit': True,
            'options': {'defaultType': 'future'}
        })
        self.results: List[QudoBacktestResult] = []
        
    async def run_backtest(
        self, 
        symbols: List[str], 
        start_date: str, 
        end_date: str,
        lookback_days: int = 60
    ) -> QudoBacktestStatistics:
        """
        Run backtest on specified symbols and date range
        
        Args:
            symbols: List of symbols to test (e.g., ['BTC/USDT', 'ETH/USDT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            lookback_days: Days of data to load before start_date for indicators
            
        Returns:
            QudoBacktestStatistics with results
        """
        self.logger.info(f"Starting Qudo backtest: {len(symbols)} symbols from {start_date} to {end_date}")
        
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        data_start_dt = start_dt - timedelta(days=lookback_days)
        
        # Load markets
        await self.exchange.load_markets()
        
        # Backtest each symbol
        for symbol in symbols:
            try:
                await self._backtest_symbol(symbol, data_start_dt, start_dt, end_dt)
            except Exception as e:
                self.logger.error(f"Error backtesting {symbol}: {e}")
                import traceback
                traceback.print_exc()
        
        # Calculate statistics
        stats = self._calculate_statistics()
        
        # Close exchange
        await self.exchange.close()
        
        self.logger.info(f"Backtest complete: {stats.total_signals} signals, {stats.wins}W/{stats.losses}L, WR={stats.win_rate:.1f}%")
        
        return stats
    
    async def _backtest_symbol(
        self, 
        symbol: str, 
        data_start: datetime, 
        test_start: datetime, 
        test_end: datetime
    ):
        """Backtest a single symbol"""
        self.logger.info(f"Backtesting {symbol}...")
        
        # Fetch all required timeframes
        df_4h = await self._fetch_ohlcv(symbol, '4h', data_start, test_end)
        df_15m = await self._fetch_ohlcv(symbol, '15m', data_start, test_end)
        df_1m = await self._fetch_ohlcv(symbol, '1m', data_start, test_end)
        
        if df_4h is None or df_15m is None or df_1m is None:
            self.logger.warning(f"Insufficient data for {symbol}")
            return
        
        self.logger.info(f"{symbol} data loaded: 4h={len(df_4h)}, 15m={len(df_15m)}, 1m={len(df_1m)}")
        
        # Filter to test period only
        test_15m = df_15m[df_15m['timestamp'] >= test_start].copy()
        
        # Simulate bar-by-bar on 15m timeframe
        for i in range(len(test_15m)):
            current_time = test_15m.iloc[i]['timestamp']
            
            if current_time > test_end:
                break
            
            # Get historical data up to current bar
            hist_4h = df_4h[df_4h['timestamp'] <= current_time].tail(100).copy()
            hist_15m = df_15m[df_15m['timestamp'] <= current_time].tail(200).copy()
            hist_1m = df_1m[df_1m['timestamp'] <= current_time].tail(100).copy()
            
            if len(hist_4h) < 50 or len(hist_15m) < 100 or len(hist_1m) < 50:
                continue
            
            # Run Qudo strategy
            try:
                signal = self.strategy.analyze(hist_4h, hist_15m, hist_1m)
            except Exception as e:
                self.logger.debug(f"Strategy error at {current_time}: {e}")
                continue
            
            if signal is None:
                continue
            
            # Signal found! Now simulate the trade
            self.logger.info(f"Signal: {symbol} {signal.direction} at {current_time}")
            
            # Get future data for simulation
            future_15m = df_15m[df_15m['timestamp'] > current_time].copy()
            future_1m = df_1m[df_1m['timestamp'] > current_time].copy()
            
            if future_15m.empty or future_1m.empty:
                continue
            
            # Simulate trade execution and exit
            result = self._simulate_trade(
                symbol, 
                signal, 
                current_time, 
                future_15m, 
                future_1m,
                max_bars=200  # Max 200 bars (50 hours on 15m)
            )
            
            if result:
                self.results.append(result)
    
    def _simulate_trade(
        self,
        symbol: str,
        signal: QudoSignal,
        entry_time: datetime,
        future_15m: pd.DataFrame,
        future_1m: pd.DataFrame,
        max_bars: int = 200
    ) -> Optional[QudoBacktestResult]:
        """
        Simulate trade execution with realistic SL/TP checking
        
        Uses 1m data for precise SL/TP hit detection
        """
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        take_profit = signal.take_profit
        
        # Calculate risk
        if signal.direction == "BUY":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Simulate bar-by-bar on 1m
        for idx, row in future_1m.head(max_bars * 15).iterrows():  # max_bars * 15 (1m bars per 15m bar)
            high = row['high']
            low = row['low']
            close = row['close']
            bar_time = row['timestamp']
            
            # Check for SL/TP hit
            if signal.direction == "BUY":
                if low <= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    outcome = "LOSS"
                    profit_loss = exit_price - entry_price
                    duration = (bar_time - entry_time).total_seconds() / 3600
                    bars_to_exit = len(future_1m[future_1m['timestamp'] <= bar_time])
                    
                    return QudoBacktestResult(
                        timestamp=entry_time,
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        htf_context=signal.htf_context,
                        liquidity_grabbed=signal.liquidity_grabbed,
                        poi_type=signal.poi.poi_type,
                        outcome=outcome,
                        exit_price=exit_price,
                        exit_timestamp=bar_time,
                        return_pct=(profit_loss / entry_price) * 100,
                        profit_loss=profit_loss,
                        duration_hours=duration,
                        bars_to_exit=bars_to_exit,
                        rr_ratio=rr_ratio
                    )
                
                elif high >= take_profit:
                    # Take profit hit
                    exit_price = take_profit
                    outcome = "WIN"
                    profit_loss = exit_price - entry_price
                    duration = (bar_time - entry_time).total_seconds() / 3600
                    bars_to_exit = len(future_1m[future_1m['timestamp'] <= bar_time])
                    
                    return QudoBacktestResult(
                        timestamp=entry_time,
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        htf_context=signal.htf_context,
                        liquidity_grabbed=signal.liquidity_grabbed,
                        poi_type=signal.poi.poi_type,
                        outcome=outcome,
                        exit_price=exit_price,
                        exit_timestamp=bar_time,
                        return_pct=(profit_loss / entry_price) * 100,
                        profit_loss=profit_loss,
                        duration_hours=duration,
                        bars_to_exit=bars_to_exit,
                        rr_ratio=rr_ratio
                    )
            
            else:  # SELL
                if high >= stop_loss:
                    # Stop loss hit
                    exit_price = stop_loss
                    outcome = "LOSS"
                    profit_loss = entry_price - exit_price
                    duration = (bar_time - entry_time).total_seconds() / 3600
                    bars_to_exit = len(future_1m[future_1m['timestamp'] <= bar_time])
                    
                    return QudoBacktestResult(
                        timestamp=entry_time,
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        htf_context=signal.htf_context,
                        liquidity_grabbed=signal.liquidity_grabbed,
                        poi_type=signal.poi.poi_type,
                        outcome=outcome,
                        exit_price=exit_price,
                        exit_timestamp=bar_time,
                        return_pct=(profit_loss / entry_price) * 100,
                        profit_loss=profit_loss,
                        duration_hours=duration,
                        bars_to_exit=bars_to_exit,
                        rr_ratio=rr_ratio
                    )
                
                elif low <= take_profit:
                    # Take profit hit
                    exit_price = take_profit
                    outcome = "WIN"
                    profit_loss = entry_price - exit_price
                    duration = (bar_time - entry_time).total_seconds() / 3600
                    bars_to_exit = len(future_1m[future_1m['timestamp'] <= bar_time])
                    
                    return QudoBacktestResult(
                        timestamp=entry_time,
                        symbol=symbol,
                        direction=signal.direction,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        htf_context=signal.htf_context,
                        liquidity_grabbed=signal.liquidity_grabbed,
                        poi_type=signal.poi.poi_type,
                        outcome=outcome,
                        exit_price=exit_price,
                        exit_timestamp=bar_time,
                        return_pct=(profit_loss / entry_price) * 100,
                        profit_loss=profit_loss,
                        duration_hours=duration,
                        bars_to_exit=bars_to_exit,
                        rr_ratio=rr_ratio
                    )
        
        # Timeout - neither SL nor TP hit within max_bars
        return QudoBacktestResult(
            timestamp=entry_time,
            symbol=symbol,
            direction=signal.direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            htf_context=signal.htf_context,
            liquidity_grabbed=signal.liquidity_grabbed,
            poi_type=signal.poi.poi_type,
            outcome="TIMEOUT",
            exit_price=entry_price,
            exit_timestamp=entry_time + timedelta(hours=50),
            return_pct=0.0,
            profit_loss=0.0,
            duration_hours=50.0,
            bars_to_exit=max_bars,
            rr_ratio=rr_ratio
        )
    
    async def _fetch_ohlcv(
        self, 
        symbol: str, 
        timeframe: str, 
        start: datetime, 
        end: datetime
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from Binance"""
        try:
            start_ms = int(start.timestamp() * 1000)
            end_ms = int(end.timestamp() * 1000)
            
            all_data = []
            current_start = start_ms
            
            while current_start < end_ms:
                ohlcv = await self.exchange.fetch_ohlcv(
                    symbol, 
                    timeframe=timeframe, 
                    since=current_start, 
                    limit=1000
                )
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                if len(ohlcv) < 1000:
                    break
                
                current_start = ohlcv[-1][0] + 1
                await asyncio.sleep(0.1)  # Rate limit
            
            if not all_data:
                return None
            
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df[~df.duplicated(subset=['timestamp'])].sort_values('timestamp')
            df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching {symbol} {timeframe}: {e}")
            return None
    
    def _calculate_statistics(self) -> QudoBacktestStatistics:
        """Calculate overall backtest statistics"""
        if not self.results:
            return QudoBacktestStatistics(
                total_signals=0, total_trades=0, wins=0, losses=0, timeouts=0,
                win_rate=0, avg_return=0, total_return=0, avg_profit=0, avg_loss=0,
                profit_factor=0, max_drawdown=0, avg_duration_hours=0,
                best_trade=0, worst_trade=0, avg_rr_ratio=0,
                bullish_signals=0, bearish_signals=0,
                asian_low_grabs=0, asian_high_grabs=0, pdl_grabs=0, pdh_grabs=0,
                order_block_pois=0, breaker_pois=0, fvg_pois=0
            )
        
        wins = [r for r in self.results if r.outcome == "WIN"]
        losses = [r for r in self.results if r.outcome == "LOSS"]
        timeouts = [r for r in self.results if r.outcome == "TIMEOUT"]
        
        total_trades = len(wins) + len(losses)
        win_rate = (len(wins) / total_trades * 100) if total_trades > 0 else 0
        
        avg_profit = np.mean([w.return_pct for w in wins]) if wins else 0
        avg_loss = np.mean([l.return_pct for l in losses]) if losses else 0
        
        total_return = sum(r.return_pct for r in wins) + sum(r.return_pct for r in losses)
        
        gross_profit = sum(w.return_pct for w in wins)
        gross_loss = abs(sum(l.return_pct for l in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0
        
        # Calculate max drawdown
        cumulative_returns = []
        cumulative = 0
        for r in sorted(self.results, key=lambda x: x.timestamp):
            if r.outcome in ["WIN", "LOSS"]:
                cumulative += r.return_pct
                cumulative_returns.append(cumulative)
        
        if cumulative_returns:
            peak = cumulative_returns[0]
            max_dd = 0
            for val in cumulative_returns:
                if val > peak:
                    peak = val
                dd = peak - val
                if dd > max_dd:
                    max_dd = dd
        else:
            max_dd = 0
        
        avg_duration = np.mean([r.duration_hours for r in self.results])
        
        best_trade = max([r.return_pct for r in self.results]) if self.results else 0
        worst_trade = min([r.return_pct for r in self.results]) if self.results else 0
        
        avg_rr = np.mean([r.rr_ratio for r in self.results])
        
        # Breakdown by HTF context
        bullish = len([r for r in self.results if r.htf_context == "BULLISH"])
        bearish = len([r for r in self.results if r.htf_context == "BEARISH"])
        
        # Breakdown by liquidity type
        asian_low = len([r for r in self.results if r.liquidity_grabbed == "asian_low"])
        asian_high = len([r for r in self.results if r.liquidity_grabbed == "asian_high"])
        pdl = len([r for r in self.results if r.liquidity_grabbed == "pdl"])
        pdh = len([r for r in self.results if r.liquidity_grabbed == "pdh"])
        
        # Breakdown by POI type
        ob = len([r for r in self.results if r.poi_type == "order_block"])
        breaker = len([r for r in self.results if r.poi_type == "breaker"])
        fvg = len([r for r in self.results if r.poi_type == "fvg"])
        
        return QudoBacktestStatistics(
            total_signals=len(self.results),
            total_trades=total_trades,
            wins=len(wins),
            losses=len(losses),
            timeouts=len(timeouts),
            win_rate=win_rate,
            avg_return=total_return / total_trades if total_trades > 0 else 0,
            total_return=total_return,
            avg_profit=avg_profit,
            avg_loss=avg_loss,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            avg_duration_hours=avg_duration,
            best_trade=best_trade,
            worst_trade=worst_trade,
            avg_rr_ratio=avg_rr,
            bullish_signals=bullish,
            bearish_signals=bearish,
            asian_low_grabs=asian_low,
            asian_high_grabs=asian_high,
            pdl_grabs=pdl,
            pdh_grabs=pdh,
            order_block_pois=ob,
            breaker_pois=breaker,
            fvg_pois=fvg
        )
    
    def save_results(self, filename: str = "qudo_backtest_results.csv"):
        """Save backtest results to CSV"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        import csv
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self.results[0]).keys()))
            writer.writeheader()
            for result in self.results:
                writer.writerow(asdict(result))
        
        self.logger.info(f"Results saved to {filename}")


# CLI interface
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest Qudo SMC Strategy')
    parser.add_argument('--symbols', type=str, required=True, help='Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)')
    parser.add_argument('--start', type=str, required=True, help='Start date YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='End date YYYY-MM-DD')
    parser.add_argument('--output', type=str, default='qudo_backtest_results.csv', help='Output CSV file')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    # Run backtest
    backtester = QudoBacktester()
    stats = await backtester.run_backtest(symbols, args.start, args.end)
    
    # Save results
    backtester.save_results(args.output)
    
    # Print statistics
    print("\n" + "="*60)
    print("QUDO SMC BACKTEST RESULTS")
    print("="*60)
    print(f"Total Signals: {stats.total_signals}")
    print(f"Total Trades: {stats.total_trades}")
    print(f"Wins: {stats.wins} | Losses: {stats.losses} | Timeouts: {stats.timeouts}")
    print(f"Win Rate: {stats.win_rate:.2f}%")
    print(f"Total Return: {stats.total_return:.2f}%")
    print(f"Avg Return: {stats.avg_return:.2f}%")
    print(f"Avg Profit: {stats.avg_profit:.2f}% | Avg Loss: {stats.avg_loss:.2f}%")
    print(f"Profit Factor: {stats.profit_factor:.2f}")
    print(f"Max Drawdown: {stats.max_drawdown:.2f}%")
    print(f"Avg Duration: {stats.avg_duration_hours:.1f}h")
    print(f"Best Trade: {stats.best_trade:.2f}% | Worst Trade: {stats.worst_trade:.2f}%")
    print(f"Avg R:R Ratio: {stats.avg_rr_ratio:.2f}")
    print("\n" + "-"*60)
    print("HTF CONTEXT BREAKDOWN")
    print("-"*60)
    print(f"Bullish: {stats.bullish_signals} | Bearish: {stats.bearish_signals}")
    print("\n" + "-"*60)
    print("LIQUIDITY TYPE BREAKDOWN")
    print("-"*60)
    print(f"Asian Low: {stats.asian_low_grabs} | Asian High: {stats.asian_high_grabs}")
    print(f"PDL: {stats.pdl_grabs} | PDH: {stats.pdh_grabs}")
    print("\n" + "-"*60)
    print("POI TYPE BREAKDOWN")
    print("-"*60)
    print(f"Order Block: {stats.order_block_pois} | Breaker: {stats.breaker_pois} | FVG: {stats.fvg_pois}")
    print("="*60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())

