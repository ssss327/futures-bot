#!/usr/bin/env python3
"""
Qudo SMC Futures Bot Launcher

Launches the FuturesBot which will:
- Analyze the market every UPDATE_INTERVAL_MINUTES
- Generate signals using Qudo SMC Strategy (HTF→MTF→LTF)
- Send high-quality setups to Telegram
- Log essential information only
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
    
    logger.info("🚀 Starting Qudo SMC Futures Bot...")
    
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