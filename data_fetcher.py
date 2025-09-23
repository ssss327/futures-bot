import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Optional, Tuple
from config import Config

class DataFetcher:
    def __init__(self):
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET_KEY,
            'sandbox': False,
            'enableRateLimit': True,
        })
        self._markets = None
        self._usdt_pairs = None
        self._data_cache = {}  # FIXED: Add data caching to reduce API calls
        self._cache_duration = 60  # Cache data for 60 seconds
    
    def _normalize_symbol(self, symbol: str) -> str:
        """FIXED: Standardize symbol format for consistent API calls"""
        # Convert various formats to Binance futures format: BTC/USDT:USDT
        if ':' not in symbol:
            # Convert BTC/USDT to BTC/USDT:USDT
            if symbol.endswith('/USDT'):
                return f"{symbol}:USDT"
            # Convert BTCUSDT to BTC/USDT:USDT
            elif symbol.endswith('USDT') and '/' not in symbol:
                base = symbol.replace('USDT', '')
                return f"{base}/USDT:USDT"
        # Already in correct format
        return symbol
    
    async def get_all_usdt_pairs(self) -> List[str]:
        """Get all USDT trading pairs available on Binance futures"""
        try:
            if self._usdt_pairs is None:
                if self._markets is None:
                    self._markets = await asyncio.get_event_loop().run_in_executor(
                        None, self.exchange.load_markets
                    )
                
                # Filter for USDT futures pairs only
                self._usdt_pairs = [
                    symbol for symbol, market in self._markets.items()
                    if (symbol.endswith(':USDT') and  # Futures format: BTC/USDT:USDT
                        market.get('type') == 'swap' and  # Perpetual futures
                        market.get('active', True))
                ]
                
                # Remove stablecoins and low-volume pairs
                excluded_bases = {'USDC', 'BUSD', 'TUSD', 'USDD', 'USDP'}
                self._usdt_pairs = [
                    pair for pair in self._usdt_pairs 
                    if pair.split('/')[0] not in excluded_bases
                ]
                
            return self._usdt_pairs
        except Exception as e:
            print(f"Error fetching USDT pairs: {e}")
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT']  # Fallback to major futures pairs
    
    async def get_top_volume_pairs(self, limit: int = 50) -> List[str]:
        """Get top trading pairs by 24h volume"""
        try:
            all_pairs = await self.get_all_usdt_pairs()
            
            # Get 24h tickers for volume data
            tickers = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_tickers
            )
            
            # Filter and sort by volume
            pair_volumes = []
            for futures_pair in all_pairs:
                # Convert futures format (BTC/USDT:USDT) to ticker format (BTC/USDT)
                ticker_symbol = futures_pair.replace(':USDT', '')
                
                if ticker_symbol in tickers and tickers[ticker_symbol].get('quoteVolume'):
                    volume = tickers[ticker_symbol]['quoteVolume']
                    pair_volumes.append((futures_pair, volume))
            
            # Sort by volume descending and return top pairs
            pair_volumes.sort(key=lambda x: x[1], reverse=True)
            return [pair for pair, volume in pair_volumes[:limit]]
            
        except Exception as e:
            print(f"Error fetching top volume pairs: {e}")
            # Return major futures pairs as fallback
            return ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'XRP/USDT:USDT', 'ADA/USDT:USDT', 'SOL/USDT:USDT', 
                   'AVAX/USDT:USDT', 'DOGE/USDT:USDT', 'MATIC/USDT:USDT', 'DOT/USDT:USDT', 'UNI/USDT:USDT']
    
    async def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 1000) -> pd.DataFrame:
        """FIXED: Fetch OHLCV data with caching to reduce API overhead"""
        try:
            # FIXED: Standardize symbol format (ensure :USDT suffix for futures)
            normalized_symbol = self._normalize_symbol(symbol)
            
            # FIXED: Check cache first to reduce API calls
            cache_key = f"{normalized_symbol}_{timeframe}_{limit}"
            current_time = datetime.now()
            
            if (cache_key in self._data_cache and 
                (current_time - self._data_cache[cache_key]['timestamp']).total_seconds() < self._cache_duration):
                return self._data_cache[cache_key]['data'].copy()
            
            # Fetch fresh data
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_ohlcv, normalized_symbol, timeframe, None, limit
            )
            
            if not ohlcv:
                return pd.DataFrame()
                
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Cache the result
            self._data_cache[cache_key] = {
                'data': df.copy(),
                'timestamp': current_time
            }
            
            return df
        except Exception as e:
            print(f"Error fetching OHLCV data for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_current_price(self, symbol: str) -> float:
        """FIXED: Fetch current price with symbol normalization and caching"""
        try:
            # FIXED: Standardize symbol format
            normalized_symbol = self._normalize_symbol(symbol)
            
            # FIXED: Use async execution and caching for price
            cache_key = f"price_{normalized_symbol}"
            current_time = datetime.now()
            
            if (cache_key in self._data_cache and 
                (current_time - self._data_cache[cache_key]['timestamp']).total_seconds() < 30):  # 30s cache for prices
                return self._data_cache[cache_key]['data']
            
            ticker = await asyncio.get_event_loop().run_in_executor(
                None, self.exchange.fetch_ticker, normalized_symbol
            )
            price = ticker['last'] if ticker and 'last' in ticker else 0.0
            
            # Cache the price
            self._data_cache[cache_key] = {
                'data': price,
                'timestamp': current_time
            }
            
            return price
        except Exception as e:
            print(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    async def fetch_orderbook(self, symbol: str, limit: int = 100) -> Dict:
        """Fetch orderbook data for liquidity analysis"""
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            print(f"Error fetching orderbook for {symbol}: {e}")
            return {}
    
    async def calculate_volatility(self, symbol: str, days: int = 250) -> float:
        """FIXED: Calculate historical volatility with caching and proper async"""
        try:
            # FIXED: Standardize symbol format
            normalized_symbol = self._normalize_symbol(symbol)
            
            # FIXED: Use caching for volatility (cache for 1 hour)
            cache_key = f"volatility_{normalized_symbol}_{days}"
            current_time = datetime.now()
            
            if (cache_key in self._data_cache and 
                (current_time - self._data_cache[cache_key]['timestamp']).total_seconds() < 3600):  # 1 hour cache
                return self._data_cache[cache_key]['data']
            
            # Fetch daily data for volatility calculation
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            ohlcv = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.exchange.fetch_ohlcv,
                normalized_symbol, 
                '1d', 
                int(start_time.timestamp() * 1000),
                days
            )
            
            if not ohlcv:
                return 0.02  # Default 2% volatility
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['returns'] = df['close'].pct_change()
            volatility = df['returns'].std() * np.sqrt(365)  # FIXED: Proper annualization
            
            result = volatility if volatility > 0 else 0.02
            
            # Cache the result
            self._data_cache[cache_key] = {
                'data': result,
                'timestamp': current_time
            }
            
            return result
        except Exception as e:
            print(f"Error calculating volatility for {symbol}: {e}")
            return 0.02
    
    async def get_symbol_info(self, symbol: str) -> Dict:
        """Get symbol information for position sizing"""
        try:
            if self._markets is None:
                self._markets = await asyncio.get_event_loop().run_in_executor(
                    None, self.exchange.load_markets
                )
            symbol_info = self._markets.get(symbol, {})
            return {
                'min_notional': symbol_info.get('limits', {}).get('cost', {}).get('min', 10),
                'price_precision': symbol_info.get('precision', {}).get('price', 4),
                'amount_precision': symbol_info.get('precision', {}).get('amount', 2),
                'tick_size': symbol_info.get('info', {}).get('tickSize', '0.0001')
            }
        except Exception as e:
            print(f"Error getting symbol info for {symbol}: {e}")
            return {
                'min_notional': 10,
                'price_precision': 4,
                'amount_precision': 2,
                'tick_size': '0.0001'
            }
    
    async def get_historical_data(self, symbol: str, timeframe: str, 
                                start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for backtesting
        
        Args:
            symbol: Trading symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1d', '4h', '1h', '15m', '5m')
            start_date: Start date for data
            end_date: End date for data
            
        Returns:
            DataFrame with OHLCV data indexed by timestamp
        """
        try:
            normalized_symbol = self._normalize_symbol(symbol)
            
            # Convert dates to milliseconds
            start_ms = int(start_date.timestamp() * 1000)
            end_ms = int(end_date.timestamp() * 1000)
            
            # Fetch data in chunks to handle large date ranges
            all_data = []
            current_start = start_ms
            
            # Determine chunk size based on timeframe
            timeframe_limits = {
                '1d': 500,   # Max 500 daily candles
                '4h': 500,   # Max 500 4h candles  
                '1h': 1000,  # Max 1000 1h candles
                '15m': 1000, # Max 1000 15m candles
                '5m': 1000   # Max 1000 5m candles
            }
            
            limit = timeframe_limits.get(timeframe, 500)
            
            while current_start < end_ms:
                try:
                    ohlcv = await asyncio.get_event_loop().run_in_executor(
                        None,
                        self.exchange.fetch_ohlcv,
                        normalized_symbol,
                        timeframe,
                        current_start,
                        limit
                    )
                    
                    if not ohlcv:
                        break
                        
                    all_data.extend(ohlcv)
                    
                    # Update start time for next chunk
                    if len(ohlcv) < limit:
                        break  # No more data available
                    
                    # Get timestamp of last candle
                    last_timestamp = ohlcv[-1][0]
                    current_start = last_timestamp + 1
                    
                    # Add delay to avoid rate limits
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    print(f"Error fetching historical data chunk: {e}")
                    break
            
            if not all_data:
                print(f"[DEBUG] No candle data found for {symbol} {timeframe}")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Convert timestamp to datetime and set as index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Remove duplicates and sort
            df = df[~df.index.duplicated(keep='first')]
            df.sort_index(inplace=True)
            
            # Filter to exact date range
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            print(f"[DEBUG] Loaded {len(df)} candles | Symbol={symbol} | Timeframe={timeframe}")
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol} {timeframe}: {e}")
            return None