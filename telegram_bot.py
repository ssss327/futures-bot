import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError
from typing import Optional
from config import Config
from signal_models import SmartMoneySignal

class TelegramSignalBot:
    def __init__(self):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.channel_id = Config.TELEGRAM_CHANNEL_ID
        self.logger = logging.getLogger(__name__)
        self.signals_sent_today = 0
        self.last_reset_date = datetime.now().date()
        
    def _reset_daily_counter_if_needed(self):
        """Reset daily signal counter if date changed"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.signals_sent_today = 0
            self.last_reset_date = current_date
            self.logger.info(f"Daily signal counter reset for {current_date}")
    
    async def send_signal(self, signal: SmartMoneySignal) -> bool:
        """Send trading signal to Telegram channel with tier-based filtering and daily limits"""
        try:
            # Reset daily counter if needed
            self._reset_daily_counter_if_needed()
            
            # Only send signals from active tiers (1-3) - no percentage filtering
            # Tier filtering is handled in futures_bot.py
            
            # Check daily signal limit
            if self.signals_sent_today >= Config.MAX_SIGNALS_PER_DAY:
                self.logger.info(f"Daily signal limit reached ({Config.MAX_SIGNALS_PER_DAY}). Signal not sent.")
                return False
            
            message = self._format_signal_message(signal)
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=message,
                parse_mode='HTML'
            )
            
            # Increment counter only after successful send
            self.signals_sent_today += 1
            self.logger.info(f"TIER {signal.tier} signal sent successfully: {signal.signal_type} ({self.signals_sent_today}/{Config.MAX_SIGNALS_PER_DAY} today)")
            return True
            
        except TelegramError as e:
            self.logger.error(f"Failed to send signal: {e}")
            return False
    
    async def send_status_message(self, message: str) -> bool:
        """Send status/info message to channel"""
        try:
            await self.bot.send_message(
                chat_id=self.channel_id,
                text=f"🤖 <b>Bot Status:</b> {message}",
                parse_mode='HTML'
            )
            return True
        except TelegramError as e:
            self.logger.error(f"Failed to send status message: {e}")
            return False
    
    def _format_signal_message(self, signal: SmartMoneySignal) -> str:
        """Format Qudo SMC signal for Telegram message"""
        
        # Determine signal emoji and color
        if signal.signal_type == 'BUY':
            signal_emoji = "🟢"
            direction_emoji = "📈"
        else:
            signal_emoji = "🔴"
            direction_emoji = "📉"
        
        # Calculate risk/reward ratio
        if signal.signal_type == 'BUY':
            risk = signal.entry_price - signal.stop_loss
            reward = signal.take_profit - signal.entry_price
        else:
            risk = signal.stop_loss - signal.entry_price
            reward = signal.entry_price - signal.take_profit
        
        rr_ratio = reward / risk if risk > 0 else 0
        
        # Format Qudo components
        concepts_text = "\n".join([f"• {concept}" for concept in signal.matched_concepts])
        
        # Determine appropriate decimal places based on price
        if signal.entry_price >= 100:
            price_decimals = 2
        elif signal.entry_price >= 1:
            price_decimals = 4
        else:
            price_decimals = 6
        
        # Create message
        message = f"""
{signal_emoji} <b>QUDO SMC SETUP</b> {signal_emoji}

🎯 <b>{signal.symbol}</b> - {signal.signal_type}
⚡ <b>Leverage:</b> {signal.leverage}x

💰 <b>Trade Setup:</b>
🎯 Entry: ${signal.entry_price:.{price_decimals}f}
🛑 Stop Loss: ${signal.stop_loss:.{price_decimals}f}
🎁 Take Profit: ${signal.take_profit:.{price_decimals}f}
📈 Risk/Reward: 1:{rr_ratio:.2f}

🧠 <b>Qudo Components:</b>
{concepts_text}

⏰ <b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
🕐 <b>Timeframes:</b> 4H (HTF) → 15m (MTF) → 1m (LTF)

<i>⚠️ Qudo SMC Strategy - High quality setups only. This is not financial advice.</i>
"""
        
        return message.strip()
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection"""
        try:
            bot_info = await self.bot.get_me()
            self.logger.info(f"Bot connected successfully: @{bot_info.username}")
            return True
        except TelegramError as e:
            self.logger.error(f"Bot connection failed: {e}")
            return False
