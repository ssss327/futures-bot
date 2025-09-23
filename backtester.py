"""
Backtesting Module for Tier-Based Smart Money Signal System

This module provides comprehensive backtesting functionality to evaluate
the performance of the tier-based signal generation system across different
market conditions and timeframes.
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import csv
import ccxt

from smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneySignal, save_rejected_signals
from data_fetcher import DataFetcher
from config import Config

@dataclass
class BacktestResult:
    """Individual backtest result for a signal"""
    timestamp: pd.Timestamp
    symbol: str
    signal_type: str
    tier: int
    tier_name: str
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    outcome: str  # 'WIN', 'LOSS', 'BE' (break-even)
    exit_price: float
    exit_timestamp: pd.Timestamp
    return_pct: float  # Return percentage
    profit_loss: float  # P&L in points
    duration_hours: float  # How long the trade lasted
    matched_concepts: List[str]
    filters_passed: List[str]

@dataclass
class TierStatistics:
    """Statistics for a specific tier"""
    tier: int
    total_signals: int
    wins: int
    losses: int
    break_evens: int
    win_rate: float
    avg_return: float
    avg_profit: float
    avg_loss: float
    total_return: float
    max_drawdown: float
    avg_duration_hours: float
    best_trade: float
    worst_trade: float
    avg_rr_ratio: float  # Average Risk:Reward ratio

@dataclass
class SymbolStatistics:
    """Statistics for a specific symbol"""
    symbol: str
    tier_stats: Dict[int, TierStatistics]
    total_signals: int
    overall_win_rate: float

class Backtester:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.analyzer = SmartMoneyAnalyzer()
        self.data_fetcher = DataFetcher()
        self.results: List[BacktestResult] = []
        self.debug_events: List[Dict[str, str]] = []
        self.debug_signal_rows: List[Dict] = []

    @staticmethod
    def _compute_rsi(series: pd.Series, period: int = 14) -> float:
        if len(series) < period + 1:
            return 50.0
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = (avg_gain / (avg_loss.replace(0, np.nan))).iloc[-1]
        if pd.isna(rs) or rs == 0:
            return 50.0
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    @staticmethod
    def _compute_macd(series: pd.Series) -> Tuple[float, float]:
        if len(series) < 35:
            return 0.0, 0.0
        ema12 = series.ewm(span=12, adjust=False).mean()
        ema26 = series.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        return float(macd_line.iloc[-1]), float(signal_line.iloc[-1])

    @staticmethod
    def _compute_ema(series: pd.Series, span: int = 20) -> float:
        if len(series) < span:
            return float(series.iloc[-1])
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1])
        
    async def run_backtest(self, symbols: Union[str, List[str]], start_date: str, end_date: str, debug_signals: bool = False) -> Dict[str, SymbolStatistics]:
        """
        Run comprehensive backtest on specified symbols and date range
        
        Args:
            symbols: Single symbol (str) or list of trading symbols (e.g., 'BTCUSDT' or ['BTCUSDT', 'ETHUSDT'])
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            debug_signals: Whether to enable debug mode for signals
            
        Returns:
            Dictionary with symbol statistics
        """
        # Convert symbols to list if single string
        if isinstance(symbols, str):
            symbols = [symbols]
            
        self.logger.info(f"Starting backtest for {len(symbols)} symbols from {start_date} to {end_date}")
        
        # Convert date strings to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Enable debug signals if requested
        if debug_signals:
            try:
                self.analyzer.enable_debug_mode()
                print("[DEBUG] Rejected signal tracking enabled (signals_rejected.csv)")
            except Exception:
                pass
        
        # Enable raw pattern capture for backtests
        try:
            self.analyzer.enable_raw_pattern_capture()
        except Exception:
            pass
        
        # Store results per symbol
        symbol_results = {}
        all_results = []
        
        for symbol in symbols:
            self.logger.info(f"Backtesting {symbol}...")
            self.results = []  # Reset results for each symbol
            await self._backtest_symbol(symbol, start_dt, end_dt)
            
            # Calculate statistics for this symbol
            tier_stats = self._calculate_tier_statistics()
            
            # Calculate overall stats for symbol
            total_signals = sum(stats.total_signals for stats in tier_stats.values())
            total_wins = sum(stats.wins for stats in tier_stats.values())
            overall_win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
            
            symbol_results[symbol] = SymbolStatistics(
                symbol=symbol,
                tier_stats=tier_stats,
                total_signals=total_signals,
                overall_win_rate=overall_win_rate
            )
            
            # Add symbol to results for CSV export
            for result in self.results:
                all_results.append(result)
            
            # Persist rejected signals for this symbol and clear buffer in debug mode
            if debug_signals and getattr(self.analyzer, 'rejected_signals', None):
                try:
                    save_rejected_signals(self.analyzer.rejected_signals, 'signals_rejected.csv')
                    self.analyzer.rejected_signals = []
                except Exception as e:
                    self.logger.error(f"Failed to append rejected signals for {symbol}: {e}")
        
        # Store all results for CSV export
        self.results = all_results
        
        # Display results
        self._display_multi_symbol_results(symbol_results, start_date, end_date)
        
        # Also print win rate per tier from calculated stats
        try:
            print("===== Win Rate by Tier =====")
            combined = self._calculate_combined_statistics(symbol_results)
            for tier in [1,2,3]:
                stats = combined[tier]
                print(f"Tier {tier}: {stats.win_rate:.2f}%")
        except Exception:
            pass
        
        # Save unified debug CSV (accepted + rejected)
        try:
            import csv
            with open('signals_debug.csv', 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=['symbol','datetime','tier','entry','SL','TP','result','RR'])
                writer.writeheader()
                for row in self.debug_signal_rows:
                    writer.writerow(row)
            print("[DEBUG] signals_debug.csv written")
        except Exception as e:
            self.logger.error(f"Failed to write signals_debug.csv: {e}")
        
        # Explicit note in logs when debug/raw mode is active
        if getattr(Config, 'DEBUG_MODE', False):
            print("⚠️ DEBUG MODE: All raw signals are being saved without filtering.")
        
        # Save raw patterns to CSV
        try:
            if getattr(self.analyzer, 'raw_patterns', None):
                import csv
                with open('patterns_raw.csv', 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=["symbol","timeframe","pattern","direction","timestamp","price"])
                    writer.writeheader()
                    writer.writerows(self.analyzer.raw_patterns)
                print("[DEBUG] Raw patterns saved to patterns_raw.csv")
        except Exception as e:
            self.logger.error(f"Failed to save raw patterns CSV: {e}")
        
        return symbol_results
    
    async def _backtest_symbol(self, symbol: str, start_date: datetime, end_date: datetime):
        """Backtest a single symbol"""
        try:
            # Load historical data for all required timeframes
            timeframes_data = await self._load_historical_data(symbol, start_date, end_date)
            
            if not timeframes_data:
                self.logger.warning(f"No data available for {symbol}")
                return
            
            # Get the base timeframe (M15) for iteration
            m15_data = timeframes_data.get('15m')
            if m15_data is None or m15_data.empty:
                self.logger.warning(f"No M15 data for {symbol}")
                return
            
            self.logger.info(f"Processing {len(m15_data)} M15 candles for {symbol}")
            
            # Single analysis point (last available M15 candle)
            i = len(m15_data) - 1
            current_timestamp = m15_data.index[i]
            current_price = m15_data.iloc[i]['close']
            
            # Prepare timeframe data up to current point
            current_timeframes = {}
            for tf, df in timeframes_data.items():
                if df is not None and not df.empty:
                    # Get data up to current timestamp
                    tf_data = df[df.index <= current_timestamp]
                    if len(tf_data) > 10:  # Minimal data requirement
                        current_timeframes[tf] = tf_data
                
            try:
                print(f"[DEBUG] Starting analysis for {symbol} with available timeframes: {list(current_timeframes.keys())}")
            except Exception:
                pass
            
            # Soften filters and collect debug conditions (RSI/MACD/EMA)
            try:
                close_series = current_timeframes['15m']['close'] if '15m' in current_timeframes else m15_data['close']
                rsi = self._compute_rsi(close_series, period=10)  # softer period
                macd, macd_signal = self._compute_macd(close_series)
                ema_fast = self._compute_ema(close_series, span=9)
                ema_slow = self._compute_ema(close_series, span=21)
                cond_rows = [
                    {"symbol": symbol, "timeframe": "15m", "condition": "RSI(10)", "value": f"{rsi:.2f}"},
                    {"symbol": symbol, "timeframe": "15m", "condition": "MACD", "value": f"{macd:.5f}"},
                    {"symbol": symbol, "timeframe": "15m", "condition": "MACD_SIGNAL", "value": f"{macd_signal:.5f}"},
                    {"symbol": symbol, "timeframe": "15m", "condition": "EMA9", "value": f"{ema_fast:.6f}"},
                    {"symbol": symbol, "timeframe": "15m", "condition": "EMA21", "value": f"{ema_slow:.6f}"}
                ]
                for row in cond_rows:
                    self.debug_events.append({
                        "symbol": row["symbol"],
                        "timeframe": row["timeframe"],
                        "condition": row["condition"],
                        "result": row["value"]
                    })
                    try:
                        print(f"[DEBUG] {row['symbol']} {row['timeframe']} | {row['condition']} = {row['value']}")
                    except Exception:
                        pass
            except Exception:
                pass
                
            # Generate signal using current analyzer
            signal = await self._generate_signal_at_timestamp(
                symbol, current_price, current_timeframes, current_timestamp
            )
            
            accepted = False
            result = None
            if signal:
                # Test the signal outcome
                result = await self._test_signal_outcome(
                    signal, timeframes_data, current_timestamp
                )
                if result:
                    self.results.append(result)
                    accepted = True

            # Record unified debug row for this timestamp
            try:
                tier = signal.tier if signal else 0
                tier_name = signal.tier_name if signal else "Rejected"
                self.debug_signal_rows.append({
                    'symbol': symbol,
                    'datetime': current_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'tier': tier,
                    'entry': round(current_price, 6),
                    'SL': round(signal.stop_loss, 6) if signal else '',
                    'TP': round(signal.take_profit, 6) if signal else '',
                    'result': result.outcome if (result and accepted) else 'REJECTED',
                    'RR': ''
                })
                print(f"[DEBUG] {symbol} 15m | signal {'accepted' if accepted else 'rejected'} | tier={tier}")
            except Exception:
                pass
            
            try:
                print(f"[DEBUG] Finished analysis for {symbol}")
            except Exception:
                pass
                        
        except Exception as e:
            self.logger.error(f"Error backtesting {symbol}: {e}")
    
    async def _load_historical_data(self, symbol: str, start_date: datetime, end_date: datetime) -> Dict[str, pd.DataFrame]:
        """Load historical data for all required timeframes"""
        timeframes = ['1d', '4h', '1h', '15m', '5m']
        timeframes_data = {}
        
        # Convert symbol format if needed (e.g., BTCUSDT -> BTC/USDT)
        if '/' not in symbol:
            if symbol.endswith('USDT'):
                base = symbol[:-4]
                symbol_formatted = f"{base}/USDT"
            else:
                symbol_formatted = symbol
        else:
            symbol_formatted = symbol
        
        for tf in timeframes:
            try:
                # Calculate how much data we need
                # For daily: start_date - 200 days for indicators
                # For smaller timeframes: start_date - appropriate periods
                if tf == '1d':
                    data_start = start_date - timedelta(days=200)
                elif tf == '4h':
                    data_start = start_date - timedelta(days=50)
                elif tf == '1h':
                    data_start = start_date - timedelta(days=15)
                elif tf == '15m':
                    data_start = start_date - timedelta(days=5)
                else:  # 5m
                    data_start = start_date - timedelta(days=3)
                
                self.logger.info(f"Loading {tf} data for {symbol_formatted} from {data_start.strftime('%Y-%m-%d')}")
                
                df = await self.data_fetcher.get_historical_data(
                    symbol_formatted, tf, data_start, end_date + timedelta(days=1)
                )
                
                if df is not None and not df.empty:
                    timeframes_data[tf] = df
                    self.logger.info(f"Loaded {len(df)} {tf} candles for {symbol_formatted}")
                    try:
                        print(f"[DEBUG] Loaded {len(df)} candles | Symbol={symbol_formatted} | Timeframe={tf}")
                    except Exception:
                        pass
                else:
                    self.logger.warning(f"No {tf} data for {symbol_formatted}")
                    try:
                        print(f"[DEBUG] No candle data found for {symbol_formatted} {tf}")
                    except Exception:
                        pass
                    
            except Exception as e:
                self.logger.error(f"Error loading {tf} data for {symbol_formatted}: {e}")
                
        return timeframes_data
    
    async def _generate_signal_at_timestamp(self, symbol: str, current_price: float, 
                                          timeframes_data: Dict[str, pd.DataFrame],
                                          timestamp: pd.Timestamp) -> Optional[SmartMoneySignal]:
        """Generate signal using the current analyzer logic"""
        try:
            # Map timeframe keys to analyzer expected format
            timeframe_mapping = {
                '1d': 'D1',
                '4h': 'H4', 
                '1h': 'H1',
                '15m': 'M15',
                '5m': 'M5'
            }
            
            valid_timeframes = {}
            for tf_key, tf_name in timeframe_mapping.items():
                if tf_key in timeframes_data:
                    valid_timeframes[tf_name] = timeframes_data[tf_key]
            
            # Calculate volatility from recent data
            if 'H4' in valid_timeframes:
                recent_data = valid_timeframes['H4'].tail(20)
                volatility = recent_data['close'].pct_change().std()
            else:
                volatility = 0.02  # Default volatility
            
            # Use the multi-timeframe signal generation
            signal = self.analyzer._generate_mtf_signal(
                valid_timeframes=valid_timeframes,
                symbol=symbol,
                current_price=current_price,
                volatility=volatility,
                daily_analysis=self.analyzer._analyze_timeframe(valid_timeframes.get('D1'), 'D1') if 'D1' in valid_timeframes else None,
                h4_analysis=self.analyzer._analyze_timeframe(valid_timeframes.get('H4'), 'H4') if 'H4' in valid_timeframes else None,
                h1_analysis=self.analyzer._analyze_timeframe(valid_timeframes.get('H1'), 'H1') if 'H1' in valid_timeframes else None,
                m15_analysis=self.analyzer._analyze_timeframe(valid_timeframes.get('M15'), 'M15') if 'M15' in valid_timeframes else None,
                m5_analysis=self.analyzer._analyze_timeframe(valid_timeframes.get('M5'), 'M5') if 'M5' in valid_timeframes else None
            )
            
            if signal:
                # Update timestamp to actual data timestamp
                signal.timestamp = timestamp
                
            return signal
            
        except Exception as e:
            self.logger.debug(f"Error generating signal for {symbol} at {timestamp}: {e}")
            return None
    
    async def _test_signal_outcome(self, signal: SmartMoneySignal, 
                                 timeframes_data: Dict[str, pd.DataFrame],
                                 entry_timestamp: pd.Timestamp) -> Optional[BacktestResult]:
        """Test the outcome of a signal"""
        try:
            # Use M5 data for precise exit detection
            m5_data = timeframes_data.get('5m')
            if m5_data is None or m5_data.empty:
                return None
            
            # Get future data after signal
            future_data = m5_data[m5_data.index > entry_timestamp].copy()
            if future_data.empty:
                return None
            
            entry_price = signal.entry_price
            stop_loss = signal.stop_loss
            take_profit = signal.take_profit
            
            outcome = 'BE'  # Default to break-even
            exit_price = entry_price
            exit_timestamp = entry_timestamp
            
            # Check each subsequent candle
            for timestamp, row in future_data.iterrows():
                high = row['high']
                low = row['low']
                close = row['close']
                
                if signal.signal_type == 'BUY':
                    # Check if stop loss hit first
                    if low <= stop_loss:
                        outcome = 'LOSS'
                        exit_price = stop_loss
                        exit_timestamp = timestamp
                        break
                    # Check if take profit hit
                    elif high >= take_profit:
                        outcome = 'WIN'
                        exit_price = take_profit
                        exit_timestamp = timestamp
                        break
                        
                else:  # SELL
                    # Check if stop loss hit first  
                    if high >= stop_loss:
                        outcome = 'LOSS'
                        exit_price = stop_loss
                        exit_timestamp = timestamp
                        break
                    # Check if take profit hit
                    elif low <= take_profit:
                        outcome = 'WIN'
                        exit_price = take_profit
                        exit_timestamp = timestamp
                        break
                
                # Timeout after 7 days if no exit
                if (timestamp - entry_timestamp).total_seconds() > 7 * 24 * 3600:
                    exit_price = close
                    exit_timestamp = timestamp
                    break
            
            # Calculate returns
            if signal.signal_type == 'BUY':
                return_pct = ((exit_price - entry_price) / entry_price) * 100
                profit_loss = exit_price - entry_price
            else:
                return_pct = ((entry_price - exit_price) / entry_price) * 100
                profit_loss = entry_price - exit_price
            
            # Account for leverage
            return_pct *= signal.leverage
            
            duration_hours = (exit_timestamp - entry_timestamp).total_seconds() / 3600
            
            return BacktestResult(
                timestamp=entry_timestamp,
                symbol=signal.symbol,
                signal_type=signal.signal_type,
                tier=signal.tier,
                tier_name=signal.tier_name,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                leverage=signal.leverage,
                outcome=outcome,
                exit_price=exit_price,
                exit_timestamp=exit_timestamp,
                return_pct=return_pct,
                profit_loss=profit_loss,
                duration_hours=duration_hours,
                matched_concepts=signal.matched_concepts.copy(),
                filters_passed=signal.filters_passed.copy()
            )
            
        except Exception as e:
            self.logger.error(f"Error testing signal outcome: {e}")
            return None
    
    def _calculate_tier_statistics(self) -> Dict[int, TierStatistics]:
        """Calculate statistics for each tier"""
        tier_stats = {}
        
        for tier in [1, 2, 3]:
            tier_results = [r for r in self.results if r.tier == tier]
            
            if not tier_results:
                tier_stats[tier] = TierStatistics(
                    tier=tier,
                    total_signals=0,
                    wins=0,
                    losses=0,
                    break_evens=0,
                    win_rate=0.0,
                    avg_return=0.0,
                    avg_profit=0.0,
                    avg_loss=0.0,
                    total_return=0.0,
                    max_drawdown=0.0,
                    avg_duration_hours=0.0,
                    best_trade=0.0,
                    worst_trade=0.0,
                    avg_rr_ratio=0.0
                )
                continue
            
            wins = [r for r in tier_results if r.outcome == 'WIN']
            losses = [r for r in tier_results if r.outcome == 'LOSS']
            break_evens = [r for r in tier_results if r.outcome == 'BE']
            
            total_signals = len(tier_results)
            win_count = len(wins)
            loss_count = len(losses)
            be_count = len(break_evens)
            
            win_rate = (win_count / total_signals * 100) if total_signals > 0 else 0
            
            returns = [r.return_pct for r in tier_results]
            avg_return = np.mean(returns) if returns else 0
            total_return = sum(returns)
            
            avg_profit = np.mean([w.return_pct for w in wins]) if wins else 0
            avg_loss = np.mean([l.return_pct for l in losses]) if losses else 0
            
            # Calculate max drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = cumulative_returns - running_max
            max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
            
            avg_duration = np.mean([r.duration_hours for r in tier_results]) if tier_results else 0
            best_trade = max(returns) if returns else 0
            worst_trade = min(returns) if returns else 0
            
            # Calculate average RR ratio
            rr_ratios = []
            for r in tier_results:
                if r.signal_type == 'BUY':
                    risk = r.entry_price - r.stop_loss
                    reward = r.take_profit - r.entry_price
                else:
                    risk = r.stop_loss - r.entry_price
                    reward = r.entry_price - r.take_profit
                if risk > 0:
                    rr_ratios.append(reward / risk)
            
            avg_rr_ratio = np.mean(rr_ratios) if rr_ratios else 0
            
            tier_stats[tier] = TierStatistics(
                tier=tier,
                total_signals=total_signals,
                wins=win_count,
                losses=loss_count,
                break_evens=be_count,
                win_rate=win_rate,
                avg_return=avg_return,
                avg_profit=avg_profit,
                avg_loss=avg_loss,
                total_return=total_return,
                max_drawdown=max_drawdown,
                avg_duration_hours=avg_duration,
                best_trade=best_trade,
                worst_trade=worst_trade,
                avg_rr_ratio=avg_rr_ratio
            )
        
        return tier_stats
    
    def _display_multi_symbol_results(self, symbol_results: Dict[str, SymbolStatistics], 
                                     start_date: str, end_date: str):
        """Display multi-symbol backtest results in console"""
        print(f"\n===== Backtest Summary ({start_date} → {end_date}) =====")
        
        # Display results for each symbol
        for symbol, symbol_stats in symbol_results.items():
            print(f"[{symbol}]")
            for tier in [1, 2, 3]:
                stats = symbol_stats.tier_stats[tier]
                tier_emojis = {1: "🔥", 2: "⚡️", 3: "⚠️"}
                
                if stats.total_signals > 0:
                    print(f"Tier {tier} {tier_emojis[tier]}: {stats.win_rate:.0f}% winrate ({stats.wins}/{stats.total_signals}) | Avg RR: {stats.avg_rr_ratio:.1f}")
                else:
                    print(f"Tier {tier} {tier_emojis[tier]}: No signals")
            print("")
        
        # Calculate and display total statistics across all symbols
        total_stats = self._calculate_combined_statistics(symbol_results)
        
        print("===== TOTAL (All Symbols) =====")
        for tier in [1, 2, 3]:
            stats = total_stats[tier]
            tier_emojis = {1: "🔥", 2: "⚡️", 3: "⚠️"}
            
            if stats.total_signals > 0:
                print(f"Tier {tier} {tier_emojis[tier]}: {stats.win_rate:.0f}% winrate ({stats.wins}/{stats.total_signals})")
            else:
                print(f"Tier {tier} {tier_emojis[tier]}: No signals")
        
        print("="*55)
        print(f"📁 Detailed results saved to: signals_backtest.csv")
        print("="*55)
    
    def _calculate_combined_statistics(self, symbol_results: Dict[str, SymbolStatistics]) -> Dict[int, TierStatistics]:
        """Calculate combined statistics across all symbols"""
        combined_stats = {}
        
        for tier in [1, 2, 3]:
            total_signals = 0
            total_wins = 0
            total_losses = 0
            total_break_evens = 0
            
            for symbol_stats in symbol_results.values():
                tier_stat = symbol_stats.tier_stats[tier]
                total_signals += tier_stat.total_signals
                total_wins += tier_stat.wins
                total_losses += tier_stat.losses
                total_break_evens += tier_stat.break_evens
            
            win_rate = (total_wins / total_signals * 100) if total_signals > 0 else 0
            
            combined_stats[tier] = TierStatistics(
                tier=tier,
                total_signals=total_signals,
                wins=total_wins,
                losses=total_losses,
                break_evens=total_break_evens,
                win_rate=win_rate,
                avg_return=0,  # Not calculated for combined stats
                avg_profit=0,
                avg_loss=0,
                total_return=0,
                max_drawdown=0,
                avg_duration_hours=0,
                best_trade=0,
                worst_trade=0,
                avg_rr_ratio=0
            )
        
        return combined_stats
    
    def _save_enhanced_csv_report(self):
        """Save enhanced backtest results to CSV"""
        if not self.results:
            self.logger.warning("No results to save")
            return
        
        filename = "signals_backtest.csv"
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Enhanced CSV format: symbol, datetime, tier, entry, SL, TP, result, RR
            fieldnames = [
                'symbol', 'datetime', 'tier', 'entry', 'SL', 'TP', 'result', 'RR'
            ]
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in self.results:
                # Calculate RR ratio
                if result.signal_type == 'BUY':
                    risk = result.entry_price - result.stop_loss
                    reward = result.take_profit - result.entry_price
                else:
                    risk = result.stop_loss - result.entry_price
                    reward = result.entry_price - result.take_profit
                
                rr_ratio = round(reward / risk, 2) if risk > 0 else 0
                
                row = {
                    'symbol': result.symbol,
                    'datetime': result.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'tier': result.tier,
                    'entry': round(result.entry_price, 6),
                    'SL': round(result.stop_loss, 6),
                    'TP': round(result.take_profit, 6),
                    'result': result.outcome,
                    'RR': rr_ratio
                }
                writer.writerow(row)
        
        self.logger.info(f"Enhanced backtest results saved to {filename}")

# Async wrapper function for CLI usage
async def run_backtest_async(symbols: Union[str, List[str]], start_date: str, end_date: str, debug_signals: bool = False) -> Dict[str, SymbolStatistics]:
    """Async wrapper for running backtest"""
    backtester = Backtester()
    return await backtester.run_backtest(symbols, start_date, end_date, debug_signals)

def run_backtest(symbols: Union[str, List[str]], start_date: str, end_date: str) -> Dict[str, SymbolStatistics]:
    """
    Synchronous wrapper for running backtest
    
    Args:
        symbols: Single symbol (str) or list of trading symbols
        start_date: Start date in 'YYYY-MM-DD' format  
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        Dictionary with symbol statistics
    """
    return asyncio.run(run_backtest_async(symbols, start_date, end_date))
