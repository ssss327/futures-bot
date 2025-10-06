import asyncio
import logging
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from typing import List, Optional

from config import Config
from telegram_bot import TelegramSignalBot
from smart_money_analyzer import SmartMoneyAnalyzer, SmartMoneySignal
from ws_data_hub import WebSocketDataHub
from trend_utils import detect_trend_and_consolidation
from binance_vision_loader import BinanceVisionLoader


class FuturesBot:
    def __init__(self):
        """Initialize the FuturesBot with all necessary components"""
        self.logger = logging.getLogger(__name__)
        
        # Initialize exchange
        self.exchange = ccxt.binance({
            'apiKey': Config.BINANCE_API_KEY,
            'secret': Config.BINANCE_SECRET_KEY,
            'sandbox': False,
            'options': {
                'defaultType': 'future'  # Use futures market
            },
            'enableRateLimit': True,
            'verbose': False  # Reduce output noise
        })
        
        # Initialize components
        self.telegram_bot = TelegramSignalBot()
        self.analyzer = SmartMoneyAnalyzer(config=Config)
        self.ws_hub = WebSocketDataHub(max_candles=1500)
        self.vision = BinanceVisionLoader()
        
        # Tracking variables
        self.last_analysis_time: Optional[datetime] = None
        self.total_signals_sent = 0

    async def run(self):
        """Main bot loop - runs analysis every UPDATE_INTERVAL_MINUTES"""
        self.logger.info("🚀 FuturesBot starting...")
        
        # Initialize connections
        if not await self.initialize():
            self.logger.error("❌ Failed to initialize. Exiting.")
            return
            
        self.logger.info(f"✅ FuturesBot initialized. Analysis every {Config.UPDATE_INTERVAL_MINUTES} minutes.")
        
        try:
            while True:
                if self.should_run_analysis():
                    await self.analyze_and_signal()
                    self.last_analysis_time = datetime.now()
                
                # Wait 1 minute before checking again
                await asyncio.sleep(60)
                
        except KeyboardInterrupt:
            self.logger.info("🛑 Bot stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Unexpected error in main loop: {e}")
        finally:
            await self.stop()

    async def initialize(self) -> bool:
        """Initialize exchange connection and test Telegram bot"""
        try:
            # Test exchange connection
            await self.exchange.load_markets()
            self.logger.info("✅ Binance Futures connection established")

            # Subscribe to key timeframes for trend and signals
            symbols = await self.get_top_symbols()
            for sym in symbols:
                # live signals timeframe (15m default)
                await self.ws_hub.subscribe(sym, "15m")
                # trend filters
                await self.ws_hub.subscribe(sym, "4h")
                await self.ws_hub.subscribe(sym, "1d")

            # Seed WS buffers with recent history (non-blocking)
            asyncio.create_task(self._seed_historical_data(symbols))

            # Test Telegram connection
            if not await self.telegram_bot.test_connection():
                self.logger.error("❌ Telegram connection failed")
                return False

            self.logger.info("✅ Telegram bot connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization error: {e}")
            return False

    async def _seed_historical_data(self, symbols):
        """Seed historical data in background"""
        try:
            import pandas as pd
            from datetime import datetime, timedelta
            end_dt = datetime.utcnow()
            start_dt_1d = end_dt - timedelta(days=400)
            start_dt_4h = end_dt - timedelta(days=120)
            start_dt_15m = end_dt - timedelta(days=15)
            for sym in symbols:
                for tf, start_dt in [("1d", start_dt_1d), ("4h", start_dt_4h), ("15m", start_dt_15m)]:
                    try:
                        df = await asyncio.wait_for(self.vision.load_range(sym, tf, start_dt, end_dt), timeout=10)
                        if df is not None and not df.empty:
                            rows = [[int(ts.value/1_000_000), float(r.open), float(r.high), float(r.low), float(r.close), float(r.volume)] for ts, r in df.iterrows()]
                            self.ws_hub.seed_buffer(sym, tf, rows[-500:])
                            self.logger.info(f"Seeded {len(rows[-500:])} candles for {sym} {tf}")
                    except asyncio.TimeoutError:
                        self.logger.warning(f"Timeout seeding {sym} {tf}")
                    except Exception as e:
                        self.logger.debug(f"Failed to seed {sym} {tf}: {e}")
        except Exception as e:
            self.logger.debug(f"Seed history failed: {e}")

    async def _analyze_symbol(self, symbol: str) -> List[SmartMoneySignal]:
        """Analyze a single symbol and return valid signals"""
        try:
            # Fetch market data (ensure enough for analysis)
            ohlcv_data = await self.fetch_klines(symbol, timeframe=Config.DEFAULT_TF if hasattr(Config, 'DEFAULT_TF') else "15m", limit=200)
            if not ohlcv_data or len(ohlcv_data) < 100:
                self.logger.debug(f"Skipping {symbol}: insufficient data ({len(ohlcv_data) if ohlcv_data else 0} candles)")
                return []

            # Trend filter using 1D and 4H (temporarily disabled for debugging)
            d1 = self.ws_hub.get_buffer(symbol, "1d")
            h4 = self.ws_hub.get_buffer(symbol, "4h")
            if not d1 or not h4 or len(d1) < 200 or len(h4) < 200:
                # wait for sufficient history
                await self.ws_hub.wait_for_update(symbol, "4h", timeout=2)
                d1 = self.ws_hub.get_buffer(symbol, "1d")
                h4 = self.ws_hub.get_buffer(symbol, "4h")
            if not d1 or not h4:
                return []
            # Temporarily disable trend filter for debugging
            # import pandas as pd
            # df_d1 = pd.DataFrame(d1, columns=['timestamp','open','high','low','close','volume'])
            # df_4h = pd.DataFrame(h4, columns=['timestamp','open','high','low','close','volume'])
            # trend, is_range = detect_trend_and_consolidation(df_d1, df_4h)
            # if is_range:
            #     return []  # skip consolidation

            self.logger.info(f"Analyzing {symbol} with {len(ohlcv_data)} candles")

            # Analyze with SmartMoneyAnalyzer
            symbol_signals = self.analyzer.analyze(symbol, ohlcv_data)
            self.logger.info(f"Found {len(symbol_signals)} raw signals for {symbol}")
            
            # Filter valid signals
            valid_signals = []
            for signal in symbol_signals:
                if self.is_valid_signal(signal):
                    valid_signals.append(signal)
                else:
                    # Calculate RR for debugging
                    if signal.signal_type == 'BUY':
                        risk = signal.entry_price - signal.stop_loss
                        reward = signal.take_profit - signal.entry_price
                    else:
                        risk = signal.stop_loss - signal.entry_price
                        reward = signal.entry_price - signal.take_profit
                    rr = reward / risk if risk > 0 else 0
                    self.logger.info(f"Signal rejected for {signal.symbol}: RR={rr:.2f}, entry={signal.entry_price:.4f}, sl={signal.stop_loss:.4f}, tp={signal.take_profit:.4f}")
            
            return valid_signals
            
        except Exception as e:
            self.logger.debug(f"Error analyzing {symbol}: {e}")
            return []

    def should_run_analysis(self) -> bool:
        """Check if enough time has passed to run analysis"""
        if not self.last_analysis_time:
            return True
        elapsed = (datetime.now() - self.last_analysis_time).total_seconds() / 60
        return elapsed >= Config.UPDATE_INTERVAL_MINUTES

    async def get_top_symbols(self) -> List[str]:
        """Get top USDT futures pairs for analysis"""
        try:
            # Since exchange is configured with defaultType='future', 
            # we can directly use major trading pairs
            major_pairs = [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'MATIC/USDT', 'LTC/USDT',
                'ATOM/USDT', 'LINK/USDT', 'UNI/USDT', 'ICP/USDT', 'FIL/USDT',
                'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'BCH/USDT', 'ALGO/USDT'
            ]
            
            # Filter out excluded symbols and limit to max (reduced for speed)
            available_symbols = [
                symbol for symbol in major_pairs 
                if symbol not in Config.EXCLUDED_SYMBOLS
            ][:5]  # Only analyze top 5 symbols for speed
            
            self.logger.info(f"📊 Selected {len(available_symbols)} major futures pairs for analysis")
            return available_symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols: {e}")
            # Fallback to basic symbols
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

    async def fetch_klines(self, symbol: str, timeframe: str = "15m", limit: int = 200) -> Optional[List[List[float]]]:
        """Fetch OHLCV candles from WS buffer with CCXT fallback."""
        try:
            buf = self.ws_hub.get_buffer(symbol, timeframe)
            if not buf or len(buf) < limit:
                # wait briefly for updates if buffer not filled yet
                await self.ws_hub.wait_for_update(symbol, timeframe, timeout=2)
                buf = self.ws_hub.get_buffer(symbol, timeframe)
            
            if buf and len(buf) >= limit:
                return buf[-limit:]
            
            # Fallback to CCXT if WS buffer is empty
            self.logger.debug(f"WS buffer empty for {symbol} {timeframe}, using CCXT fallback")
            ohlcv = await self.exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            return ohlcv
        except Exception as e:
            self.logger.debug(f"Error fetching klines for {symbol}: {e}")
            return None

    async def analyze_and_signal(self):
        """Main analysis loop: fetch data, analyze with SmartMoneyAnalyzer, send signals"""
        start_time = datetime.now()
        self.logger.info("🔍 Starting market analysis...")
        
        signals_found = []
        signals_sent = []
        
        try:
            # Get symbols to analyze
            symbols = await self.get_top_symbols()
            if not symbols:
                self.logger.warning("⚠️ No symbols found for analysis")
                return
            
            # Analyze symbols in parallel (faster)
            import asyncio
            tasks = []
            for symbol in symbols:
                task = asyncio.create_task(self._analyze_symbol(symbol))
                tasks.append(task)
            
            # Wait for all analyses to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Collect valid signals from all results
            for result in results:
                if isinstance(result, list):
                    signals_found.extend(result)
                elif isinstance(result, Exception):
                    self.logger.debug(f"Symbol analysis failed: {result}")
            
            self.logger.info(f"Collected {len(signals_found)} valid signals from parallel analysis")

            # Sort signals by tier and quality (tier 1 first, then by score if available)
            signals_found.sort(key=lambda s: (s.tier, -len(s.matched_concepts)))
            
            # Send signals to Telegram
            for signal in signals_found:
                if len(signals_sent) >= Config.MAX_SIGNALS_PER_DAY:
                    break
                    
                try:
                    if await self.telegram_bot.send_signal(signal):
                        signals_sent.append(signal)
                        self.total_signals_sent += 1
                except Exception as e:
                    self.logger.error(f"Failed to send signal for {signal.symbol}: {e}")
            
            # Log results
            analysis_duration = (datetime.now() - start_time).total_seconds()
            self.logger.info(
                f"✅ Analysis complete: {len(signals_found)} signals found, "
                f"{len(signals_sent)} sent to Telegram "
                f"(Total sent today: {self.telegram_bot.signals_sent_today}) "
                f"in {analysis_duration:.1f}s"
            )
            
            # Ensure minimum signals warning
            if len(signals_found) < Config.MIN_SIGNALS_PER_RUN:
                self.logger.warning(
                    f"⚠️ Only {len(signals_found)} signals found, "
                    f"expected at least {Config.MIN_SIGNALS_PER_RUN}"
                )

        except Exception as e:
            self.logger.error(f"❌ Error in analyze_and_signal: {e}")

    def is_valid_signal(self, signal: SmartMoneySignal) -> bool:
        """Validate signal before sending"""
        try:
            # Basic validation
            if not signal.symbol or not signal.signal_type:
                self.logger.info(f"Signal rejected: missing symbol or type")
                return False
            
            if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0:
                self.logger.info(f"Signal rejected: invalid prices - entry={signal.entry_price}, sl={signal.stop_loss}, tp={signal.take_profit}")
                return False
            
            # Check age (signals should be fresh) - more lenient for live signals
            if signal.timestamp:
                age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
                # Allow up to 60 minutes for live analysis signals
                if age_minutes > 60:
                    self.logger.info(f"Signal rejected: too old - {age_minutes:.1f} minutes")
                    return False
            
            # Check risk/reward ratio
            if signal.signal_type == 'BUY':
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit
            
            if risk <= 0:
                self.logger.info(f"Signal rejected: invalid risk - {risk}")
                return False
                
            rr_ratio = reward / risk
            # Temporarily lower threshold for debugging
            min_rr = 1.0
            if rr_ratio < min_rr:
                self.logger.info(f"Signal rejected: RR too low - {rr_ratio:.2f} < {min_rr}")
                return False
            
            self.logger.info(f"Signal ACCEPTED: {signal.symbol} {signal.signal_type} RR={rr_ratio:.2f}")
            return True
            
        except Exception as e:
            self.logger.info(f"Signal rejected: validation error - {e}")
            return False

    async def stop(self):
        """Cleanup resources"""
        try:
            await self.exchange.close()
            self.logger.info("✅ FuturesBot stopped")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")