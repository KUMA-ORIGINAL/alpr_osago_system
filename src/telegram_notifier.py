import os
import logging
import httpx
import asyncio
from typing import Optional

logger = logging.getLogger(__name__)


class TelegramNotifier:
    def __init__(self, timeout: float = 10.0):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID')
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        self.timeout = timeout
        self.client: Optional[httpx.AsyncClient] = None

        if self.bot_token and self.chat_id:
            logger.info("Telegram нотификатор инициализирован")
        else:
            logger.warning("Telegram не настроен")

    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def send_message(self, message: str, disable_notification: bool = False) -> bool:
        """Отправка сообщения в Telegram через HTTPX"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram не настроен, сообщение не отправлено")
            return False

        try:
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML',
                'disable_notification': disable_notification
            }

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(self.api_url, json=payload)

                if response.status_code == 200:
                    logger.info("Уведомление отправлено в Telegram")
                    return True
                else:
                    error_data = response.json()
                    logger.error(f"Ошибка отправки в Telegram: {response.status_code} - {error_data}")
                    return False

        except httpx.TimeoutException:
            logger.error("Таймаут при отправке сообщения в Telegram")
            return False
        except httpx.NetworkError as e:
            logger.error(f"Ошибка сети при отправке в Telegram: {e}")
            return False
        except httpx.HTTPError as e:
            logger.error(f"HTTP ошибка при отправке в Telegram: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при отправке в Telegram: {e}")
            return False

    async def send_message_with_retry(self,
                                      message: str,
                                      max_retries: int = 3,
                                      backoff_factor: float = 1.0) -> bool:
        """Отправка сообщения с повторными попытками и экспоненциальной backoff задержкой"""
        for attempt in range(max_retries):
            success = await self.send_message(message)

            if success:
                return True
            elif attempt < max_retries - 1:
                wait_time = backoff_factor * (2 ** attempt)
                logger.warning(f"Попытка {attempt + 1} не удалась. Повтор через {wait_time:.1f} сек...")
                await asyncio.sleep(wait_time)

        logger.error(f"Не удалось отправить сообщение после {max_retries} попыток")
        return False

    async def send_photo(self, photo_url: str, caption: str = "") -> bool:
        """Отправка фото в Telegram"""
        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram не настроен, фото не отправлено")
            return False

        try:
            payload = {
                'chat_id': self.chat_id,
                'photo': photo_url,
                'caption': caption,
                'parse_mode': 'HTML'
            }

            api_url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(api_url, json=payload)

                if response.status_code == 200:
                    logger.info("Фото отправлено в Telegram")
                    return True
                else:
                    error_data = response.json()
                    logger.error(f"Ошибка отправки фото в Telegram: {response.status_code} - {error_data}")
                    return False

        except Exception as e:
            logger.error(f"Ошибка при отправке фото в Telegram: {e}")
            return False

    async def check_connection(self) -> bool:
        """Проверка подключения к Telegram API"""
        if not self.bot_token:
            return False

        try:
            api_url = f"https://api.telegram.org/bot{self.bot_token}/getMe"

            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(api_url)
                return response.status_code == 200

        except Exception as e:
            logger.error(f"Ошибка проверки подключения к Telegram: {e}")
            return False


# Пример использования
async def main():
    # Простое использование
    notifier = TelegramNotifier()

    # Отправка сообщения
    success = await notifier.send_message("🚨 Обнаружено нарушение!")

    # Отправка с повторными попытками
    success = await notifier.send_message_with_retry(
        "Важное сообщение",
        max_retries=3,
        backoff_factor=1.5
    )

    # Использование как контекстный менеджер
    async with TelegramNotifier() as notifier_ctx:
        await notifier_ctx.send_message("Сообщение через контекстный менеджер")

    # Проверка подключения
    if await notifier.check_connection():
        print("Подключение к Telegram работает")
    else:
        print("Проблемы с подключением к Telegram")


if __name__ == "__main__":
    asyncio.run(main())