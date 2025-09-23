import os
from dotenv import load_dotenv
import ccxt

load_dotenv()


class Config:
    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID')

    # Exchange Configuration
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')

    # Bot Configuration
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', 30))  # 30 minutes default
    LEVERAGE_MIN = int(os.getenv('LEVERAGE_MIN', 1))
    LEVERAGE_MAX = int(os.getenv('LEVERAGE_MAX', 10))
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', 2))

    # FIXED: Smart Money Configuration - Practical thresholds
    SMC_LOOKBACK_PERIODS = 100
    ORDER_BLOCK_THRESHOLD = 1.5
    FVG_MIN_SIZE = 0.05
    LIQUIDITY_THRESHOLD = 1.3
    MARKET_STRUCTURE_PERIODS = 20

    # Risk Management
    DEFAULT_STOP_LOSS_PERCENTAGE = 2.0
    DEFAULT_TAKE_PROFIT_RATIO = 2.0
    VOLATILITY_CALCULATION_DAYS = 250
    MAX_SIGNAL_AGE_MINUTES = 30

    # Multi-Symbol Configuration
    MAX_SYMBOLS_TO_MONITOR = int(os.getenv('MAX_SYMBOLS_TO_MONITOR', 50))  # Top X symbols by volume
    SYMBOL_SCAN_INTERVAL_MINUTES = int(os.getenv('SYMBOL_SCAN_INTERVAL_MINUTES', 30))
    CONCURRENT_ANALYSIS_LIMIT = int(os.getenv('CONCURRENT_ANALYSIS_LIMIT', 10))

    # Symbol filtering
    EXCLUDED_SYMBOLS = os.getenv('EXCLUDED_SYMBOLS', 'USDC/USDT,BUSD/USDT,TUSD/USDT').split(',')
    MIN_24H_VOLUME_USDT = float(os.getenv('MIN_24H_VOLUME_USDT', 10000000))

    # Enhanced Signal Accuracy Parameters
    CONFLUENCE_THRESHOLD = 20
    MIN_TIMEFRAMES_FOR_SIGNAL = 5
    REQUIRE_D1_ALIGNMENT = True
    MIN_RR_RATIO = 3.0
    VOLUME_CONFIRM_MULTIPLIER = 1.5
    MAX_SIGNALS_PER_DAY = 6
    SLIPPAGE_PERCENT = 0.0003
    MAKER_TAKER_FEES = 0.0004
    MIN_FILL_LIQUIDITY = 5000000

    # Weighted Scoring
    WEIGHT_D1_BIAS = 6
    WEIGHT_H4_BOS_OB = 5
    WEIGHT_VOLUME_SPIKE_BOS = 4
    WEIGHT_M15_OB_REJECTION = 3
    WEIGHT_M5_MOMENTUM = 2

    # Tiered Signal System
    MIN_SIGNALS_PER_RUN = 10

    # Debug flags
    DEBUG_MODE = True
    SAVE_REJECTED_SIGNALS = True

    # -----------------------------
    # AUTOLOAD SYMBOLS SECTION
    # -----------------------------
    AUTOLOAD_SYMBOLS = True  # если True → бот сам подгрузит список фьючерсных пар

    @staticmethod
    def get_symbols():
        if Config.AUTOLOAD_SYMBOLS:
            exchange = ccxt.binance({
                "options": {"defaultType": "future"}
            })
            markets = exchange.load_markets()

            usdt_pairs = [
                s.replace("/", "")  # убираем слэш для вида BTCUSDT
                for s in markets.keys()
                if s.endswith("/USDT") and s not in Config.EXCLUDED_SYMBOLS
            ]

            return usdt_pairs[:Config.MAX_SYMBOLS_TO_MONITOR]
        else:
            # fallback если автозагрузка выключена
            return ["BTCUSDT", "ETHUSDT"]
