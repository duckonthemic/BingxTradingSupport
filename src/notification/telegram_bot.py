"""
Telegram Bot Client.
Handles sending alerts to Telegram channels/groups.
"""

import asyncio
import logging
from typing import Optional
from datetime import datetime

from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError, RetryAfter

from ..config import config
from ..storage.redis_client import RedisClient

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Telegram notification handler.
    Sends formatted alerts with rate limiting and cooldown management.
    """
    
    def __init__(self, redis_client: RedisClient):
        self.bot_token = config.telegram.bot_token
        self.chat_id = config.telegram.chat_id
        self.redis = redis_client
        self.cooldown_seconds = config.timing.alert_cooldown
        
        self._bot: Optional[Bot] = None
        self._enabled = bool(self.bot_token and self.chat_id)
        
        if not self._enabled:
            logger.warning("⚠️ Telegram not configured - alerts will be logged only")
    
    async def connect(self):
        """Initialize Telegram bot."""
        if self._enabled:
            self._bot = Bot(token=self.bot_token)
            try:
                me = await self._bot.get_me()
                logger.info(f"🤖 Telegram bot connected: @{me.username}")
            except TelegramError as e:
                logger.error(f"❌ Telegram connection failed: {e}")
                self._enabled = False
    
    async def disconnect(self):
        """Close bot connection."""
        if self._bot:
            await self._bot.close()
            self._bot = None
    
    async def _send_message(self, message: str) -> bool:
        """
        Internal method to send message to Telegram.
        Handles rate limiting and errors.
        """
        if not self._enabled or not self._bot:
            # Log message if Telegram not configured
            logger.info(f"📝 [LOG ONLY]\n{message[:200]}...")
            return True
        
        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
            return True
            
        except RetryAfter as e:
            logger.warning(f"⏳ Rate limited, retry after {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            return await self._send_message(message)
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Send failed: {e}")
            return False
    
    async def _send_message_with_buttons(self, message: str, buttons: list) -> bool:
        """
        Send message with inline keyboard buttons.
        """
        from telegram import InlineKeyboardMarkup
        
        if not self._enabled or not self._bot:
            logger.info(f"📝 [LOG ONLY]\n{message[:200]}...")
            return True
        
        try:
            keyboard = InlineKeyboardMarkup(buttons)
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard,
                disable_web_page_preview=True
            )
            return True
            
        except RetryAfter as e:
            logger.warning(f"⏳ Rate limited, retry after {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            return await self._send_message_with_buttons(message, buttons)
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Send with buttons failed: {e}")
            return False
    
    @property
    def is_enabled(self) -> bool:
        return self._enabled
    
    async def pin_message(self, message_id: int) -> bool:
        """Pin a message in the chat."""
        if not self._enabled or not self._bot:
            logger.info(f"📌 [LOG ONLY] Would pin message {message_id}")
            return True
        
        try:
            await self._bot.pin_chat_message(
                chat_id=self.chat_id,
                message_id=message_id,
                disable_notification=True
            )
            return True
        except TelegramError as e:
            logger.error(f"❌ Failed to pin message: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Pin message error: {e}")
            return False

    async def send_message(self, message: str) -> Optional[int]:
        """
        Send message and return message_id.
        
        Returns:
            message_id if successful, None otherwise
        """
        if not self._enabled or not self._bot:
            logger.info(f"📝 [LOG ONLY]\n{message[:200]}...")
            return None
        
        try:
            sent = await self._bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.HTML,
                disable_web_page_preview=True
            )
            return sent.message_id
            
        except RetryAfter as e:
            logger.warning(f"⏳ Rate limited, retry after {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            return await self.send_message(message)
            
        except TelegramError as e:
            logger.error(f"❌ Telegram error: {e}")
            return None
    
    async def send_photo_with_id(self, photo_bytes: bytes, caption: str, buttons: list = None) -> Optional[int]:
        """
        Send photo with caption and return message_id.
        
        Returns:
            message_id if successful, None otherwise
        """
        from telegram import InlineKeyboardMarkup
        import io
        
        if not self._enabled or not self._bot:
            logger.info(f"📸 [LOG ONLY] Photo with caption:\n{caption[:200]}...")
            return None
        
        try:
            keyboard = InlineKeyboardMarkup(buttons) if buttons else None
            sent = await self._bot.send_photo(
                chat_id=self.chat_id,
                photo=io.BytesIO(photo_bytes),
                caption=caption,
                parse_mode=ParseMode.HTML,
                reply_markup=keyboard
            )
            return sent.message_id
            
        except RetryAfter as e:
            logger.warning(f"⏳ Rate limited, retry after {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            return await self.send_photo_with_id(photo_bytes, caption, buttons)
            
        except TelegramError as e:
            logger.error(f"❌ Send photo error: {e}")
            return None
    
    async def reply_to_message(self, message_id: int, text: str) -> Optional[int]:
        """
        Reply to a specific message.
        
        Args:
            message_id: ID of message to reply to
            text: Reply text
            
        Returns:
            New message_id if successful, None otherwise
        """
        if not self._enabled or not self._bot:
            logger.info(f"↩️ [LOG ONLY] Reply to {message_id}:\n{text[:200]}...")
            return None
        
        try:
            sent = await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode=ParseMode.HTML,
                reply_to_message_id=message_id,
                disable_web_page_preview=True
            )
            return sent.message_id
            
        except RetryAfter as e:
            logger.warning(f"⏳ Rate limited, retry after {e.retry_after}s")
            await asyncio.sleep(e.retry_after)
            return await self.reply_to_message(message_id, text)
            
        except TelegramError as e:
            # If original message deleted, send as new message
            if "message not found" in str(e).lower():
                logger.warning(f"⚠️ Original message {message_id} not found, sending as new")
                return await self.send_message(text)
            logger.error(f"❌ Reply error: {e}")
            return None
