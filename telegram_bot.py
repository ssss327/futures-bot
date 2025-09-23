import asyncio
import logging
from datetime import datetime, timedelta
from telegram import Bot
from telegram.error import TelegramError
from typing import Optional
from config import Config
from smart_money_analyzer import SmartMoneySignal

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
        """Format trading signal for Telegram message"""
        
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
        
        # Format matched concepts
        concepts_text = "\n".join([f"• {concept}" for concept in signal.matched_concepts])
        
        # Get base asset from symbol (e.g., BTC from BTC/USDT)
        base_asset = signal.symbol.split('/')[0]
        
        # Determine appropriate decimal places based on price
        if signal.entry_price >= 100:
            price_decimals = 2
        elif signal.entry_price >= 1:
            price_decimals = 4
        else:
            price_decimals = 6
        
        # Get tier information
        tier_emojis = {1: "🔥", 2: "⚡️", 3: "⚠️"}
        tier_names = {1: "High Confidence", 2: "Medium Confidence", 3: "Low Confidence"}
        tier_emoji = tier_emojis.get(signal.tier, "❓")
        tier_name = tier_names.get(signal.tier, "Unknown")
        
        # Tier-specific display (loosened filters note)
        tier_requirements = "⚠️ Loosened filters active – expect more signals for testing."
        
        # Create message
        message = f"""
{tier_emoji} <b>TIER {signal.tier} ({tier_name})</b> {tier_emoji}

🎯 <b>Symbol:</b> {signal.symbol}
{direction_emoji} <b>Direction:</b> {signal.signal_type}
⚡ <b>Leverage:</b> {signal.leverage}x (0.25 Kelly)

💰 <b>Trade Setup:</b>
🎯 Entry: ${signal.entry_price:.{price_decimals}f}
🛑 Stop Loss: ${signal.stop_loss:.{price_decimals}f}
🎁 Take Profit 1 (40%): ${signal.entry_price + (signal.take_profit - signal.entry_price) * 0.33:.{price_decimals}f}
🎁 Take Profit 2 (30%): ${signal.take_profit:.{price_decimals}f}
📈 Trail (30%): Break-even + 1 ATR
📈 Risk/Reward: 1:{rr_ratio:.2f}

{tier_emoji} <b>TIER {signal.tier} REQUIREMENTS MET:</b>
{tier_requirements}

🧠 <b>Signal Components:</b>
{concepts_text}

⏰ <b>Time:</b> {signal.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}
🕐 <b>Hierarchy:</b> D1 (Primary) → H4 (Structure) → M15/M5 (Confirmation)

<i>⚠️ Tiered signal system ensures minimum {Config.MIN_SIGNALS_PER_RUN} signals per run. Loosened filters are active for testing. This is not financial advice.</i>
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
