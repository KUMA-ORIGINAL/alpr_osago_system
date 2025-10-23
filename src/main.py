import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import List, Dict, Any
import cv2
from fastapi import FastAPI
from contextlib import asynccontextmanager
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from alpr_processor import ALPRProcessor
from osago_checker import OSAGOChecker
from telegram_notifier import TelegramNotifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALPROSAGOSystem:
    def __init__(self):
        self.alpr = ALPRProcessor()
        self.osago_checker = OSAGOChecker()
        self.telegram_notifier = TelegramNotifier()

        # Инициализация Redis с обработкой ошибок
        self.redis_client = None
        self.redis_available = False
        self.init_redis()

        # Временный кэш в памяти, если Redis недоступен
        self.memory_cache = {}

        self.camera_urls = [url.strip() for url in os.getenv('CAMERA_RTSP_URLS', '').split(',') if url.strip()]
        self.check_interval = int(os.getenv('CHECK_INTERVAL', 60))

        logger.info(f"Загружено {len(self.camera_urls)} камер: {self.camera_urls}")

    def init_redis(self):
        """Инициализация Redis с обработкой ошибок подключения"""
        try:
            redis_host = os.getenv('REDIS_HOST', 'localhost')
            redis_port = int(os.getenv('REDIS_PORT', 6379))

            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True
            )

            # Тестируем подключение
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis подключен успешно")

        except (RedisConnectionError, Exception) as e:
            logger.warning(f"Redis недоступен: {e}. Используется in-memory кэш")
            self.redis_available = False

    def get_cached_result(self, plate_number: str) -> tuple[bool, bool]:
        """Получить результат из кэша (Redis или memory)"""
        cache_key = f"plate:{plate_number}"

        if self.redis_available:
            try:
                cached_result = self.redis_client.get(cache_key)
                if cached_result is not None:
                    return True, cached_result == 'true'
            except RedisConnectionError:
                self.redis_available = False
                logger.warning("Потеряно соединение с Redis, переключаемся на memory кэш")

        # Fallback to memory cache
        if plate_number in self.memory_cache:
            cache_data = self.memory_cache[plate_number]
            if time.time() - cache_data['timestamp'] < self.check_interval:
                return True, cache_data['has_osago']

        return False, False

    def set_cached_result(self, plate_number: str, has_osago: bool):
        """Сохранить результат в кэш"""
        cache_key = f"plate:{plate_number}"

        if self.redis_available:
            try:
                self.redis_client.setex(cache_key, self.check_interval, 'true' if has_osago else 'false')
            except RedisConnectionError:
                self.redis_available = False
                logger.warning("Ошибка записи в Redis, используем memory кэш")

        # Fallback to memory cache
        self.memory_cache[plate_number] = {
            'has_osago': has_osago,
            'timestamp': time.time()
        }

        # Очистка старых записей в memory cache
        current_time = time.time()
        self.memory_cache = {
            plate: data for plate, data in self.memory_cache.items()
            if current_time - data['timestamp'] < self.check_interval
        }

    async def process_camera(self, camera_url: str, camera_name: str):
        """Обработка одной камеры"""
        reconnect_attempts = 0
        max_reconnect_attempts = 5

        while reconnect_attempts < max_reconnect_attempts:
            cap = None
            try:
                cap = cv2.VideoCapture(camera_url)

                if not cap.isOpened():
                    logger.error(f"Не удалось открыть камеру {camera_name}: {camera_url}")
                    reconnect_attempts += 1
                    await asyncio.sleep(10)
                    continue

                logger.info(f"Успешно подключились к камере {camera_name}")
                reconnect_attempts = 0
                frame_count = 0
                last_processing_time = 0

                while True:
                    ret, frame = cap.read()

                    if not ret:
                        logger.warning(f"Потеряно соединение с камерой {camera_name}")
                        break

                    current_time = time.time()
                    # Обрабатываем кадры с интервалом (1 кадр в секунду для оптимизации)
                    if current_time - last_processing_time >= 1.0:
                        plates = self.alpr.detect_plates(frame)

                        for plate_info in plates:
                            await self.process_detected_plate(plate_info, camera_name)

                        last_processing_time = current_time
                        frame_count += 1

                    await asyncio.sleep(0.01)  # Небольшая пауза

            except Exception as e:
                logger.error(f"Ошибка в камере {camera_name}: {e}")
                reconnect_attempts += 1

            finally:
                if cap is not None:
                    cap.release()

            if reconnect_attempts < max_reconnect_attempts:
                wait_time = min(30, 5 * reconnect_attempts)
                logger.info(f"Повторное подключение к {camera_name} через {wait_time} сек...")
                await asyncio.sleep(wait_time)

        logger.error(f"Превышено максимальное количество попыток подключения к {camera_name}")

    async def process_detected_plate(self, plate_info: Dict[str, Any], camera_name: str):
        """Обработка обнаруженного номерного знака"""
        plate_number = plate_info['plate']
        confidence = plate_info['confidence']

        # Пропускаем номера с низкой уверенностью
        if confidence < 0.7:
            return

        # Нормализуем номер
        plate_number = plate_number.replace(' ', '').replace('-', '').upper()

        # Пропускаем слишком короткие или некорректные номера
        if len(plate_number) < 5 or len(plate_number) > 12:
            return

        logger.info(f"Обнаружен номер: {plate_number} (уверенность: {confidence:.2f})")

        found_in_cache, has_osago = self.get_cached_result(plate_number)

        if found_in_cache:
            logger.info(f"Номер {plate_number} из кэша: ОСАГО {'есть' if has_osago else 'нет'}")
        else:
            has_osago = await self.osago_checker.check_osago(plate_number)

            # Сохраняем в кэш
            self.set_cached_result(plate_number, has_osago)

            logger.info(f"Номер {plate_number} проверен: ОСАГО {'есть' if has_osago else 'нет'}")

        # Если ОСАГО нет - отправляем уведомление
        if not has_osago:
            await self.send_violation_notification(plate_number, camera_name, plate_info)
        else:
            logger.info(f"Номер {plate_number} пропущен (есть ОСАГО)")

    async def send_violation_notification(self, plate_number: str, camera_name: str, plate_info: Dict[str, Any]):
        """Отправка уведомления о нарушении"""
        message = (
            f"🚨 Нарушение! Обнаружен автомобиль без ОСАГО\n"
            f"📷 Камера: {camera_name}\n"
            f"🚗 Номер: {plate_number}\n"
            f"⏰ Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"🎯 Уверенность: {plate_info['confidence']:.2f}"
        )

        # Сохраняем в лог
        violation_data = {
            'plate': plate_number,
            'camera': camera_name,
            'timestamp': datetime.now().isoformat(),
            'confidence': plate_info['confidence']
        }

        self.save_violation_log(violation_data)

        # Отправляем в Telegram
        await self.telegram_notifier.send_message_with_retry(message)

    def save_violation_log(self, data: Dict[str, Any]):
        """Сохранение лога нарушений"""
        log_file = "/app/data/violations.json"
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        try:
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"Ошибка сохранения лога: {e}")

    async def start_monitoring(self):
        """Запуск мониторинга всех камер"""
        tasks = []
        for i, camera_url in enumerate(self.camera_urls):
            camera_name = f"Camera_{i + 1}"
            task = asyncio.create_task(self.process_camera(camera_url, camera_name))
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)


# Создаем экземпляр системы
system = ALPROSAGOSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Запуск системы ALPR OSAGO...")
    asyncio.create_task(system.start_monitoring())
    yield
    # Shutdown
    logger.info("Остановка системы ALPR OSAGO...")


app = FastAPI(title="ALPR OSAGO System", lifespan=lifespan)


@app.get("/health")
async def health_check():
    redis_status = "connected" if system.redis_available else "disconnected"
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "cameras_count": len(system.camera_urls),
        "redis": redis_status,
        "memory_cache_size": len(system.memory_cache)
    }


@app.get("/stats")
async def get_stats():
    return {
        "cameras_count": len(system.camera_urls),
        "redis_available": system.redis_available,
        "memory_cache_size": len(system.memory_cache)
    }


@app.get("/violations")
async def get_violations(limit: int = 10):
    """Получить последние нарушения"""
    try:
        with open("/app/data/violations.json", "r", encoding='utf-8') as f:
            lines = f.readlines()[-limit:]
            return [json.loads(line) for line in lines]
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
