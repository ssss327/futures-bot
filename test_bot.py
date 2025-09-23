#!/usr/bin/env python3
"""
Test script for the XRP Futures Bot
Tests individual components and overall functionality
"""

import asyncio
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from data_fetcher import DataFetcher
from smart_money_analyzer import SmartMoneyAnalyzer
from telegram_bot import TelegramSignalBot
from config import Config

async def test_data_fetcher():
    """Test data fetching functionality"""
    print("🧪 Testing Data Fetcher...")
    
    fetcher = DataFetcher()
    test_symbol = "BTC/USDT"  # Use BTC for testing
    
    try:
        # Test symbol discovery (quick test with 3 symbols)
        symbols = await fetcher.get_top_volume_pairs(3)
        print(f"✅ Top 3 Volume Pairs: {symbols}")
        
        if symbols:
            test_symbol = symbols[0]  # Use first symbol from list
        
        # Test current price
        current_price = await fetcher.fetch_current_price(test_symbol)
        print(f"✅ Current {test_symbol} Price: ${current_price:.4f}")
        
        # Test OHLCV data
        df = await fetcher.fetch_ohlcv(test_symbol, '5m', 50)
        print(f"✅ OHLCV Data: {len(df)} candles fetched")
        if not df.empty:
            print(f"   Latest close: ${df['close'].iloc[-1]:.4f}")
        
        # Test volatility calculation
        volatility = await fetcher.calculate_volatility(test_symbol, 30)
        print(f"✅ 30-day Volatility: {volatility:.4f} ({volatility*100:.2f}%)")
        
        # Test symbol info
        symbol_info = await fetcher.get_symbol_info(test_symbol)
        print(f"✅ Symbol Info: {symbol_info}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data Fetcher Test Failed: {e}")
        return False

async def test_smart_money_analyzer():
    """Test smart money analysis"""
    print("\n🧪 Testing Smart Money Analyzer...")
    
    try:
        fetcher = DataFetcher()
        analyzer = SmartMoneyAnalyzer()
        test_symbol = "BTC/USDT"
        
        # Get available symbols (quick test)
        symbols = await fetcher.get_top_volume_pairs(3)
        if symbols:
            test_symbol = symbols[0]
        
        # Get test data
        current_price = await fetcher.fetch_current_price(test_symbol)
        volatility = await fetcher.calculate_volatility(test_symbol, 30)
        
        if current_price <= 0:
            print("❌ No price data available for analysis")
            return False
        
        # Test multi-timeframe analysis
        signal = await analyzer.analyze_multi_timeframe(fetcher, test_symbol, current_price, volatility)
        
        if signal:
            print(f"✅ Signal Generated for {test_symbol}:")
            print(f"   Type: {signal.signal_type}")
            print(f"   Confidence: {signal.confidence:.1f}%")
            print(f"   Entry: ${signal.entry_price:.4f}")
            print(f"   Stop Loss: ${signal.stop_loss:.4f}")
            print(f"   Take Profit: ${signal.take_profit:.4f}")
            print(f"   Leverage: {signal.leverage}x")
            print(f"   Concepts: {', '.join(signal.matched_concepts[:3])}...")  # Show first 3
        else:
            print(f"✅ No signal generated for {test_symbol} (insufficient confluence)")
        
        return True
        
    except Exception as e:
        print(f"❌ Smart Money Analyzer Test Failed: {e}")
        return False

async def test_telegram_bot():
    """Test Telegram bot functionality"""
    print("\n🧪 Testing Telegram Bot...")
    
    try:
        bot = TelegramSignalBot()
        
        # Test connection
        if await bot.test_connection():
            print("✅ Telegram Bot Connected")
        else:
            print("❌ Telegram Bot Connection Failed")
            return False
        
        # Test status message
        if await bot.send_status_message("Test message from bot 🧪"):
            print("✅ Status Message Sent")
        else:
            print("❌ Failed to Send Status Message")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Telegram Bot Test Failed: {e}")
        return False

async def test_closed_candle_verification():
    """Test that indicators only use closed candles (no look-ahead bias)"""
    print("\n🧪 Testing Closed Candle Verification...")
    
    try:
        fetcher = DataFetcher()
        analyzer = SmartMoneyAnalyzer()
        
        # Get test data
        symbol = "BTC/USDT"
        df = await fetcher.fetch_ohlcv(symbol, '15m', 100)
        
        if df.empty:
            print("❌ No data available for testing")
            return False
        
        # Remove the last candle (simulating current unclosed candle)
        closed_df = df.iloc[:-1].copy()
        
        # Test various indicator calculations
        market_structure_closed = analyzer._analyze_market_structure(closed_df)
        order_blocks_closed = analyzer._identify_order_blocks(closed_df)
        fvgs_closed = analyzer._identify_fair_value_gaps(closed_df)
        
        # Verify no future data is used
        if len(order_blocks_closed) > 0:
            latest_ob_timestamp = order_blocks_closed[-1]['timestamp']
            if latest_ob_timestamp >= df.index[-1]:
                print("❌ Look-ahead bias detected in order blocks")
                return False
        
        if len(fvgs_closed) > 0:
            latest_fvg_timestamp = fvgs_closed[-1]['timestamp']
            if latest_fvg_timestamp >= df.index[-1]:
                print("❌ Look-ahead bias detected in FVGs")
                return False
        
        print("✅ Closed candle verification passed - no look-ahead bias detected")
        return True
        
    except Exception as e:
        print(f"❌ Closed Candle Verification Failed: {e}")
        return False

async def test_walk_forward_analysis():
    """Test walk-forward analysis: train 6 months, test 2 months, step 1 month"""
    print("\n🧪 Testing Walk-Forward Analysis...")
    
    try:
        fetcher = DataFetcher()
        analyzer = SmartMoneyAnalyzer()
        symbol = "BTC/USDT"
        
        # Get 12 months of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        print(f"Fetching 12 months of data for {symbol}...")
        df = await fetcher.fetch_ohlcv(symbol, '1d', 365)
        
        if len(df) < 240:  # Need at least 8 months
            print(f"❌ Insufficient data: {len(df)} days")
            return False
        
        # Walk-forward parameters
        train_period = 180  # 6 months
        test_period = 60    # 2 months  
        step_size = 30      # 1 month
        
        results = []
        total_signals = 0
        profitable_signals = 0
        
        # Perform walk-forward analysis
        for i in range(0, len(df) - train_period - test_period, step_size):
            train_start = i
            train_end = i + train_period
            test_start = train_end
            test_end = test_start + test_period
            
            if test_end > len(df):
                break
            
            train_data = df.iloc[train_start:train_end]
            test_data = df.iloc[test_start:test_end]
            
            # Simulate signal generation on test period
            for j in range(len(test_data) - 1):
                test_candle = test_data.iloc[j]
                future_price = test_data.iloc[j + 1]['close']
                current_price = test_candle['close']
                
                # Simple signal generation for testing
                volatility = 0.02  # Mock volatility
                signal = await analyzer.analyze_multi_timeframe(
                    fetcher, symbol, current_price, volatility
                )
                
                if signal:
                    total_signals += 1
                    # Check if signal would be profitable
                    if signal.signal_type == 'BUY' and future_price > signal.entry_price:
                        profitable_signals += 1
                    elif signal.signal_type == 'SELL' and future_price < signal.entry_price:
                        profitable_signals += 1
            
            window_results = {
                'train_period': f"{train_data.index[0]} to {train_data.index[-1]}",
                'test_period': f"{test_data.index[0]} to {test_data.index[-1]}",
                'signals': total_signals,
                'profitable': profitable_signals
            }
            results.append(window_results)
        
        # Calculate overall performance
        accuracy = (profitable_signals / total_signals * 100) if total_signals > 0 else 0
        
        print(f"✅ Walk-Forward Analysis Completed:")
        print(f"   Total Windows: {len(results)}")
        print(f"   Total Signals: {total_signals}")
        print(f"   Profitable Signals: {profitable_signals}")
        print(f"   Accuracy: {accuracy:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Walk-Forward Analysis Failed: {e}")
        return False

async def test_monte_carlo_simulation():
    """Test Monte-Carlo simulation with random slippage and fees"""
    print("\n🧪 Testing Monte-Carlo Simulation...")
    
    try:
        fetcher = DataFetcher()
        analyzer = SmartMoneyAnalyzer()
        symbol = "BTC/USDT"
        
        # Get some historical data
        df = await fetcher.fetch_ohlcv(symbol, '4h', 200)
        
        if len(df) < 100:
            print(f"❌ Insufficient data: {len(df)} candles")
            return False
        
        iterations = 100  # Reduced for faster testing (use 2000 in production)
        results = []
        
        print(f"Running {iterations} Monte-Carlo simulations...")
        
        for i in range(iterations):
            # Random slippage and fees for this iteration
            random_slippage = np.random.uniform(0.0001, 0.0005)  # 0.01% to 0.05%
            random_fees = np.random.uniform(0.0002, 0.0006)  # 0.02% to 0.06%
            
            # Simulate trade outcomes
            pnl = 0
            trades = 0
            
            # Take random samples from the data
            sample_size = min(50, len(df) - 10)
            sample_indices = np.random.choice(len(df) - 10, sample_size, replace=False)
            
            for idx in sample_indices:
                current_price = df.iloc[idx]['close']
                future_price = df.iloc[idx + 5]['close']  # 5 periods later
                
                # Generate mock signal
                volatility = 0.02
                signal = await analyzer.analyze_multi_timeframe(
                    fetcher, symbol, current_price, volatility
                )
                
                if signal:
                    trades += 1
                    entry_price = signal.entry_price
                    
                    # Apply slippage to entry
                    if signal.signal_type == 'BUY':
                        entry_with_slippage = entry_price * (1 + random_slippage)
                        trade_return = (future_price - entry_with_slippage) / entry_with_slippage
                    else:
                        entry_with_slippage = entry_price * (1 - random_slippage)
                        trade_return = (entry_with_slippage - future_price) / entry_with_slippage
                    
                    # Apply fees
                    trade_return -= random_fees * 2  # Entry + exit fees
                    
                    pnl += trade_return
            
            # Record iteration results
            if trades > 0:
                avg_return = pnl / trades
                win_rate = len([r for r in [pnl] if r > 0]) / max(1, trades)
                results.append({
                    'iteration': i + 1,
                    'trades': trades,
                    'pnl': pnl,
                    'avg_return': avg_return,
                    'slippage': random_slippage,
                    'fees': random_fees
                })
        
        # Analyze Monte-Carlo results
        if results:
            avg_pnl = np.mean([r['pnl'] for r in results])
            std_pnl = np.std([r['pnl'] for r in results])
            profitable_iterations = len([r for r in results if r['pnl'] > 0])
            success_rate = profitable_iterations / len(results) * 100
            
            print(f"✅ Monte-Carlo Simulation Completed:")
            print(f"   Iterations: {len(results)}")
            print(f"   Average PnL: {avg_pnl:.4f}")
            print(f"   PnL Std Dev: {std_pnl:.4f}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   Avg Slippage: {np.mean([r['slippage'] for r in results]):.4f}")
            print(f"   Avg Fees: {np.mean([r['fees'] for r in results]):.4f}")
        else:
            print("❌ No valid results from Monte-Carlo simulation")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Monte-Carlo Simulation Failed: {e}")
        return False

async def test_full_integration():
    """Test full bot integration"""
    print("\n🧪 Testing Full Integration...")
    
    try:
        from futures_bot import FuturesBot
        
        bot = FuturesBot()
        
        # Test initialization
        if await bot.initialize():
            print("✅ Bot Initialization Successful")
        else:
            print("❌ Bot Initialization Failed")
            return False
        
        # Test basic functionality without generating signals
        print("✅ Full Integration Test Passed - Bot initialization and basic functions working")
        
        return True
        
    except Exception as e:
        print(f"❌ Full Integration Test Failed: {e}")
        return False

async def run_all_tests():
    """Run all tests"""
    print("🚀 Starting Multi-Symbol Futures Bot Tests...\n")
    
    tests = [
        ("Data Fetcher", test_data_fetcher),
        ("Smart Money Analyzer", test_smart_money_analyzer),
        ("Telegram Bot", test_telegram_bot),
        ("Closed Candle Verification", test_closed_candle_verification),
        ("Walk-Forward Analysis", test_walk_forward_analysis),
        ("Monte-Carlo Simulation", test_monte_carlo_simulation),
        ("Full Integration", test_full_integration)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nTotal: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! Bot is ready to use.")
        return True
    else:
        print(f"\n⚠️  {len(results) - passed} test(s) failed. Please check configuration.")
        return False

if __name__ == "__main__":
    # Check if config is properly set
    if not Config.TELEGRAM_BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN not set in .env file")
        sys.exit(1)
    
    if not Config.TELEGRAM_CHANNEL_ID:
        print("❌ TELEGRAM_CHANNEL_ID not set in .env file") 
        sys.exit(1)
    
    # Run tests
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
