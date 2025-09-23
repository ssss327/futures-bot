#!/usr/bin/env python3
"""
Smart Money Futures Bot Launcher

Simply launches the FuturesBot which will:
- Analyze the market every UPDATE_INTERVAL_MINUTES
- Generate signals using SmartMoneyAnalyzer
- Send signals to Telegram
- Log only essential information (no symbol list spam)
"""

import asyncio
import logging
import sys
from futures_bot import FuturesBot


def setup_logging():
    """Setup clean logging with minimal output"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("futures_bot.log")
        ]
    )
    
    # Reduce noise from external libraries
    logging.getLogger('ccxt').setLevel(logging.ERROR)
    logging.getLogger('urllib3').setLevel(logging.ERROR)
    logging.getLogger('telegram').setLevel(logging.WARNING)
    logging.getLogger('asyncio').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)


async def main():
    """Main entry point"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🚀 Starting Smart Money Futures Bot...")
    
    try:
        bot = FuturesBot()
        await bot.run()
    except KeyboardInterrupt:
        logger.info("🛑 Bot stopped by user (Ctrl+C)")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[STOPPED] Bot manually interrupted.")
    except Exception as e:
        print(f"[ERROR] Failed to start bot: {e}")
        sys.exit(1)