import asyncio
import logging
from datetime import datetime, timedelta
import ccxt.async_support as ccxt
from typing import List, Optional

from config import Config
from telegram_bot import TelegramSignalBot
from signal_models import SmartMoneySignal
from ws_data_hub import WebSocketDataHub
from qudo_smc_strategy import QudoSMCStrategy


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
        self.qudo_strategy = QudoSMCStrategy()  # Only Qudo strategy now
        self.ws_hub = WebSocketDataHub(max_candles=1500)
        
        # Tracking variables
        self.last_analysis_time: Optional[datetime] = None
        self.total_signals_sent = 0
        self.recent_signal_symbols = {}  # Track {symbol: timestamp} to prevent duplicates

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
                # Qudo strategy timeframes: 4H (HTF), 15m (MTF), 1m (LTF)
                await self.ws_hub.subscribe(sym, "4h")
                await self.ws_hub.subscribe(sym, "15m")
                await self.ws_hub.subscribe(sym, "1m")

            # Test Telegram connection
            if not await self.telegram_bot.test_connection():
                self.logger.error("❌ Telegram connection failed")
                return False

            self.logger.info("✅ Telegram bot connection established")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Initialization error: {e}")
            return False

    async def _analyze_symbol(self, symbol: str) -> List[SmartMoneySignal]:
        """Analyze a single symbol using Qudo SMC strategy"""
        try:
            # Fetch all required timeframes for Qudo strategy
            buf_4h = self.ws_hub.get_buffer(symbol, "4h")
            buf_15m = self.ws_hub.get_buffer(symbol, "15m")
            buf_1m = self.ws_hub.get_buffer(symbol, "1m")
            
            # Fallback to CCXT if buffers not filled
            if not buf_4h or len(buf_4h) < 50:
                ohlcv_4h = await self.exchange.fetch_ohlcv(symbol, timeframe="4h", limit=100)
                buf_4h = ohlcv_4h if ohlcv_4h else []
            
            if not buf_15m or len(buf_15m) < 100:
                ohlcv_15m = await self.exchange.fetch_ohlcv(symbol, timeframe="15m", limit=200)
                buf_15m = ohlcv_15m if ohlcv_15m else []
            
            if not buf_1m or len(buf_1m) < 50:
                ohlcv_1m = await self.exchange.fetch_ohlcv(symbol, timeframe="1m", limit=100)
                buf_1m = ohlcv_1m if ohlcv_1m else []
            
            # Check if we have enough data
            if len(buf_4h) < 50 or len(buf_15m) < 100 or len(buf_1m) < 50:
                self.logger.debug(f"Skipping {symbol}: insufficient data (4h:{len(buf_4h)}, 15m:{len(buf_15m)}, 1m:{len(buf_1m)})")
                return []
            
            # Convert to DataFrames
            import pandas as pd
            df_4h = pd.DataFrame(buf_4h, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_15m = pd.DataFrame(buf_15m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df_1m = pd.DataFrame(buf_1m, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            df_4h['timestamp'] = pd.to_datetime(df_4h['timestamp'], unit='ms')
            df_15m['timestamp'] = pd.to_datetime(df_15m['timestamp'], unit='ms')
            df_1m['timestamp'] = pd.to_datetime(df_1m['timestamp'], unit='ms')
            
            self.logger.debug(f"Analyzing {symbol} (4h:{len(df_4h)}, 15m:{len(df_15m)}, 1m:{len(df_1m)})")
            
            # Run Qudo strategy analysis
            qudo_signal = self.qudo_strategy.analyze(df_4h, df_15m, df_1m)
            
            if qudo_signal is None:
                return []
            
            # Convert QudoSignal to SmartMoneySignal
            smc_signal = SmartMoneySignal(
                symbol=symbol,
                signal_type=qudo_signal.direction,
                entry_price=qudo_signal.entry_price,
                stop_loss=qudo_signal.stop_loss,
                take_profit=qudo_signal.take_profit,
                leverage=5,  # Default leverage
                timestamp=datetime.now(),
                tier=1,  # Qudo signals are high quality
                matched_concepts=[
                    f"HTF:{qudo_signal.htf_context}",
                    f"LIQ:{qudo_signal.liquidity_grabbed}",
                    f"BOS:confirmed",
                    f"POI:{qudo_signal.poi.poi_type}",
                    "LTF:CHoCH"
                ],
                filters_passed=["QudoSMC"]
            )
            
            self.logger.info(f"🎯 Qudo setup: {symbol} {qudo_signal.direction} @ {qudo_signal.entry_price:.4f} (SL:{qudo_signal.stop_loss:.4f}, TP:{qudo_signal.take_profit:.4f})")
            
            # Validate signal
            if self.is_valid_signal(smc_signal):
                return [smc_signal]
            else:
                return []
            
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
                'ADA/USDT', 'AVAX/USDT', 'DOT/USDT', 'LTC/USDT', 'ATOM/USDT',
                'ATOM/USDT', 'LINK/USDT', 'UNI/USDT', 'ICP/USDT', 'FIL/USDT',
                'TRX/USDT', 'ETC/USDT', 'XLM/USDT', 'BCH/USDT', 'ALGO/USDT'
            ]
            
            # Filter out excluded symbols and limit to top pairs
            available_symbols = [
                symbol for symbol in major_pairs 
                if symbol not in Config.EXCLUDED_SYMBOLS
            ][:10]  # Analyze top 10 liquid pairs for quality setups
            
            self.logger.info(f"📊 Selected {len(available_symbols)} major futures pairs for analysis")
            return available_symbols
            
        except Exception as e:
            self.logger.error(f"❌ Error getting symbols: {e}")
            # Fallback to basic symbols
            return ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']

    async def analyze_and_signal(self):
        """Main analysis loop: analyze symbols with Qudo strategy and send signals"""
        start_time = datetime.now()
        self.logger.info("🔍 Starting market analysis...")
        
        # Clean up old symbol tracking (remove entries older than cooldown period)
        current_time = datetime.now()
        expired_symbols = [
            sym for sym, ts in self.recent_signal_symbols.items()
            if (current_time - ts).total_seconds() / 3600 > Config.SIGNAL_COOLDOWN_HOURS
        ]
        for sym in expired_symbols:
            del self.recent_signal_symbols[sym]
        
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

            # Filter out duplicate symbols (keep only one signal per symbol per cooldown period)
            filtered_signals = []
            current_time = datetime.now()
            
            for signal in signals_found:
                symbol = signal.symbol
                
                # Check if we recently signaled this symbol
                if symbol in self.recent_signal_symbols:
                    last_signal_time = self.recent_signal_symbols[symbol]
                    hours_since = (current_time - last_signal_time).total_seconds() / 3600
                    
                    if hours_since < Config.SIGNAL_COOLDOWN_HOURS:
                        self.logger.info(f"Skipping {symbol}: signaled {hours_since:.1f}h ago (cooldown: {Config.SIGNAL_COOLDOWN_HOURS}h)")
                        continue
                
                filtered_signals.append(signal)
            
            self.logger.info(f"After cooldown filter: {len(filtered_signals)} unique signals")
            signals_found = filtered_signals

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
                        # Mark this symbol as recently signaled
                        self.recent_signal_symbols[signal.symbol] = datetime.now()
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
            
            # Note: Qudo strategy is selective, fewer signals = higher quality

        except Exception as e:
            self.logger.error(f"❌ Error in analyze_and_signal: {e}")

    def is_valid_signal(self, signal: SmartMoneySignal) -> bool:
        """Validate signal before sending"""
        try:
            # Basic validation
            if not signal.symbol or not signal.signal_type:
                return False
            
            if signal.entry_price <= 0 or signal.stop_loss <= 0 or signal.take_profit <= 0:
                return False
            
            # Check age (signals should be fresh)
            if signal.timestamp:
                age_minutes = (datetime.now() - signal.timestamp).total_seconds() / 60
                if age_minutes > 60:
                    return False
            
            # Check risk/reward ratio
            if signal.signal_type == 'BUY':
                risk = signal.entry_price - signal.stop_loss
                reward = signal.take_profit - signal.entry_price
            else:
                risk = signal.stop_loss - signal.entry_price
                reward = signal.entry_price - signal.take_profit
            
            if risk <= 0:
                return False
                
            rr_ratio = reward / risk
            if rr_ratio < Config.MIN_RR_RATIO:
                self.logger.debug(f"Signal rejected: {signal.symbol} RR={rr_ratio:.2f} < {Config.MIN_RR_RATIO}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.debug(f"Validation error: {e}")
            return False

    async def stop(self):
        """Cleanup resources"""
        try:
            await self.ws_hub.stop()
            await self.exchange.close()
            self.logger.info("✅ FuturesBot stopped")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")