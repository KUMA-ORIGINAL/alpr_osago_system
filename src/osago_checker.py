import asyncio
import logging
from typing import Dict, Any, Optional
import mechanicalsoup

logger = logging.getLogger(__name__)


class OSAGOChecker:
    def __init__(self):
        self.browser = mechanicalsoup.StatefulBrowser()
        self.base_url = "https://strahovanie.kg/osago/check"
        self.request_timeout = 30
        self.max_retries = 3
        self.retry_delay = 2

        # Настройка браузера
        self.browser.set_user_agent(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )

    async def check_osago(self, plate_number: str) -> bool:
        """
        Проверить наличие ОСАГО по номеру автомобиля

        Args:
            plate_number: Номер автомобиля (например, "01KG123ABC")

        Returns:
            bool: True если ОСАГО есть, False если нет или ошибка
        """
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Проверка ОСАГО для номера {plate_number} (попытка {attempt + 1})")

                # Нормализуем номер
                normalized_plate = self.normalize_plate_number(plate_number)

                # Выполняем проверку в отдельном потоке чтобы не блокировать event loop
                result = await asyncio.get_event_loop().run_in_executor(
                    None, self._check_osago_sync, normalized_plate
                )

                logger.info(f"Результат проверки {plate_number}: ОСАГО {'есть' if result else 'нет'}")
                return result

            except Exception as e:
                logger.error(f"Ошибка при проверке ОСАГО для {plate_number} (попытка {attempt + 1}): {e}")

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay)
                else:
                    logger.error(f"Все попытки проверки ОСАГО для {plate_number} завершились ошибкой")
                    return False

    def _check_osago_sync(self, plate_number: str) -> bool:
        """Синхронная проверка ОСАГО (выполняется в отдельном потоке)"""
        try:
            # Открываем страницу проверки
            logger.debug(f"Открываем страницу проверки для {plate_number}")
            page = self.browser.open(self.base_url, timeout=self.request_timeout)

            if not page.ok:
                logger.error(f"Не удалось открыть страницу проверки: HTTP {page.status_code}")
                return False

            # Ищем форму
            form = self.browser.page.select_one("form.FormCheck_root__5EGkn")
            if not form:
                logger.error("Форма проверки не найдена на странице")
                return False

            # Находим поле ввода номера
            input_field = form.select_one("input[name='vehicle_gov_plate']")
            if not input_field:
                logger.error("Поле ввода номера не найдено в форме")
                return False

            # Заполняем форму
            self.browser["vehicle_gov_plate"] = plate_number

            # Отправляем форму
            logger.debug(f"Отправляем форму с номером {plate_number}")
            result_page = self.browser.submit(form, page.url)

            if not result_page.ok:
                logger.error(f"Ошибка при отправке формы: HTTP {result_page.status_code}")
                return False

            # Парсим результат
            return self._parse_result_page(result_page, plate_number)

        except mechanicalsoup.LinkNotFoundError as e:
            logger.error(f"Элемент не найден на странице: {e}")
            return False
        except Exception as e:
            logger.error(f"Неожиданная ошибка при проверке ОСАГО: {e}")
            return False

    def _parse_result_page(self, result_page, plate_number: str) -> bool:
        """Парсинг страницы с результатом проверки"""
        try:
            # Ищем элемент с результатом
            status_element = result_page.soup.select_one(".CheckStatus_title__Y31qh")

            if not status_element:
                logger.warning(f"Элемент результата не найден для номера {plate_number}")
                return False

            status_text = status_element.get_text(strip=True).lower()
            logger.debug(f"Текст статуса для {plate_number}: {status_text}")

            # Проверяем наличие ОСАГО по тексту статуса
            if "застрахован" in status_text:
                return True
            elif "не найден" in status_text:
                return False
            else:
                logger.warning(f"Неизвестный статус ОСАГО: {status_text}")
                return False

        except Exception as e:
            logger.error(f"Ошибка при парсинге результата для {plate_number}: {e}")
            return False

    def normalize_plate_number(self, plate_number: str) -> str:
        """
        Нормализация номера автомобиля

        Args:
            plate_number: Исходный номер

        Returns:
            str: Нормализованный номер
        """
        # Удаляем пробелы, дефисы, приводим к верхнему регистру
        normalized = plate_number.upper().replace(' ', '').replace('-', '')

        # Дополнительная валидация формата
        if len(normalized) < 5 or len(normalized) > 12:
            logger.warning(f"Номер {plate_number} имеет нестандартную длину: {len(normalized)}")

        return normalized

    async def check_multiple_plates(self, plates: list) -> Dict[str, bool]:
        """
        Проверить несколько номеров

        Args:
            plates: Список номеров автомобилей

        Returns:
            Dict[str, bool]: Словарь с результатами проверки
        """
        results = {}

        for plate in plates:
            results[plate] = await self.check_osago(plate)
            # Небольшая пауза между запросами чтобы не нагружать сервер
            await asyncio.sleep(1)

        return results

    def get_detailed_info(self, plate_number: str) -> Optional[Dict[str, Any]]:
        """
        Получить подробную информацию о полисе (если найден)

        Args:
            plate_number: Номер автомобиля

        Returns:
            Optional[Dict]: Подробная информация или None если ошибка
        """
        try:
            normalized_plate = self.normalize_plate_number(plate_number)

            # Открываем страницу
            page = self.browser.open(self.base_url, timeout=self.request_timeout)
            if not page.ok:
                return None

            # Заполняем и отправляем форму
            form = self.browser.page.select_one("form.FormCheck_root__5EGkn")
            if not form:
                return None

            self.browser["vehicle_gov_plate"] = normalized_plate
            result_page = self.browser.submit(form, page.url)

            if not result_page.ok:
                return None

            return self._parse_detailed_info(result_page, normalized_plate)

        except Exception as e:
            logger.error(f"Ошибка при получении детальной информации для {plate_number}: {e}")
            return None

    def _parse_detailed_info(self, result_page, plate_number: str) -> Dict[str, Any]:
        """Парсинг детальной информации о полисе"""
        info = {
            "plate": plate_number,
            "has_osago": False,
            "status": "unknown",
            "details": {}
        }

        try:
            # Основной статус
            status_element = result_page.soup.select_one(".CheckStatus_title__Y31qh")
            if status_element:
                status_text = status_element.get_text(strip=True)
                info["status"] = status_text
                info["has_osago"] = "застрахован" in status_text.lower()

            # Детальная информация
            info_rows = result_page.soup.select(".CheckInfo_infoRow__cUKmx")
            for row in info_rows:
                paragraphs = row.find_all("p")
                if len(paragraphs) == 2:
                    key = paragraphs[0].get_text(strip=True)
                    value = paragraphs[1].get_text(strip=True)
                    info["details"][key] = value

            # Информация о базах
            base_rows = result_page.soup.select(".CheckInfo_rowItem__tYuv7")
            for row in base_rows:
                base_text = row.get_text(strip=True)
                info["details"][f"base_{len(info['details'])}"] = base_text

        except Exception as e:
            logger.error(f"Ошибка при парсинге детальной информации: {e}")

        return info

    async def close(self):
        """Закрыть браузер и освободить ресурсы"""
        try:
            self.browser.close()
            logger.info("Браузер OSAGO checker закрыт")
        except Exception as e:
            logger.error(f"Ошибка при закрытии браузера: {e}")

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


# Пример использования
async def main():
    """Пример использования OSAGOChecker"""
    checker = OSAGOChecker()

    try:
        # Проверка одного номера
        result = await checker.check_osago("01KG120AAA")
        print(f"ОСАГО для 01KG120AAA: {'ЕСТЬ' if result else 'НЕТ'}")

        # Проверка нескольких номеров
        plates = ["01KG120AAA", "01KG999BBB", "02KG123CCC"]
        results = await checker.check_multiple_plates(plates)

        for plate, has_osago in results.items():
            print(f"{plate}: {'ЕСТЬ' if has_osago else 'НЕТ'}")

        # Детальная информация
        detailed_info = checker.get_detailed_info("01KG120AAA")
        if detailed_info:
            print(f"Детальная информация: {detailed_info}")

    finally:
        await checker.close()


if __name__ == "__main__":
    asyncio.run(main())