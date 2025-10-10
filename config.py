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
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', 30))
    LEVERAGE_MIN = int(os.getenv('LEVERAGE_MIN', 1))
    LEVERAGE_MAX = int(os.getenv('LEVERAGE_MAX', 10))
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', 2))

    # Qudo SMC Strategy Configuration
    QUDO_HTF_LOOKBACK = 50  # 4H bars for Order Flow detection
    QUDO_MTF_LOOKBACK = 200  # 15m bars for liquidity and setup
    QUDO_LTF_LOOKBACK = 100  # 1m bars for CHoCH confirmation

    # Symbol Filtering
    EXCLUDED_SYMBOLS = os.getenv('EXCLUDED_SYMBOLS', 'USDC/USDT,BUSD/USDT,TUSD/USDT').split(',')
    MIN_24H_VOLUME_USDT = float(os.getenv('MIN_24H_VOLUME_USDT', 10000000))

    # Signal Quality Parameters
    MIN_RR_RATIO = 1.5  # Minimum risk/reward for Qudo signals
    MAX_SIGNALS_PER_DAY = 6
    SIGNAL_COOLDOWN_HOURS = 4  # Don't signal same token within 4 hours
    SLIPPAGE_PERCENT = 0.0003
    MAKER_TAKER_FEES = 0.0004

    # Debug Flags
    DEBUG_MODE = True
    SAVE_REJECTED_SIGNALS = True
