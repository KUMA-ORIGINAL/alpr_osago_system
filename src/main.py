import asyncio
import json
import logging
import os
import time
from datetime import datetime
from typing import Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import redis
from redis.exceptions import ConnectionError as RedisConnectionError

from alpr_processor import ALPRProcessor
from osago_checker import OSAGOChecker
from telegram_notifier import TelegramNotifier
from easy4ip_client import Easy4ipClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ALPROSAGOSystem:
    def __init__(self):
        self.alpr = ALPRProcessor()
        self.osago_checker = OSAGOChecker()
        self.telegram_notifier = TelegramNotifier()

        # Redis
        self.redis_client = None
        self.redis_available = False
        self.init_redis()

        self.memory_cache: dict[str, dict] = {}

        # камеры могут задаваться напрямую, либо будут получены через Easy4ip
        raw_urls = os.getenv('CAMERA_RTSP_URLS', '')
        self.camera_urls = [url.strip() for url in raw_urls.split(',') if url.strip()]
        self.check_interval = int(os.getenv('CHECK_INTERVAL', 60))

        logger.info(f"Инициализация ALPR OSAGO System")
        logger.info(f"Камер в конфигурации: {len(self.camera_urls)}")

    def init_redis(self):
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
            self.redis_client.ping()
            self.redis_available = True
            logger.info("Redis подключен успешно")
        except Exception as e:
            logger.warning(f"Redis недоступен: {e}, используется память")

    async def load_camera_urls_from_easy4ip(self):
        """Получить URL‑ы камер через Easy4ip, если прямые ссылки не заданы"""
        if self.camera_urls:
            return  # заданы вручную

        app_id = os.getenv("EASY4IP_APP_ID")
        secret = os.getenv("EASY4IP_APP_SECRET")
        if not app_id or not secret:
            logger.error("EASY4IP_* переменные не заданы и CAMERA_RTSP_URLS пусто")
            return

        easy4ip = Easy4ipClient()
        token = await easy4ip.get_access_token()
        devices = os.getenv("EASY4IP_DEVICE_IDS", "")
        ids = [d.strip() for d in devices.split(",") if d.strip()]

        for dev_id in ids:
            try:
                url = await easy4ip.get_or_create_live_url(token, dev_id, "0")
                if url:
                    self.camera_urls.append(url)
                    logger.info(f"Для устройства {dev_id} получен live URL")
            except Exception as e:
                logger.warning(f"Ошибка Easy4ip для {dev_id}: {e}")

    #  остальной код – как в твоей версии (get_cached_result, set_cached_result, process_camera, и т.д.)
    #  всё без изменений ↓↓↓

    def get_cached_result(self, plate_number: str) -> tuple[bool, bool]:
        cache_key = f"plate:{plate_number}"
        if self.redis_available:
            try:
                v = self.redis_client.get(cache_key)
                if v is not None:
                    return True, v == 'true'
            except RedisConnectionError:
                self.redis_available = False
                logger.warning("Потеряно соединение с Redis")
        d = self.memory_cache.get(plate_number)
        if d and time.time() - d['timestamp'] < self.check_interval:
            return True, d['has_osago']
        return False, False

    def set_cached_result(self, plate_number: str, has_osago: bool):
        cache_key = f"plate:{plate_number}"
        if self.redis_available:
            try:
                self.redis_client.setex(cache_key, self.check_interval, 'true' if has_osago else 'false')
            except RedisConnectionError:
                self.redis_available = False
        self.memory_cache[plate_number] = {'has_osago': has_osago, 'timestamp': time.time()}

    async def process_camera(self, camera_url: str, camera_name: str):
        reconnect_attempts = 0
        while reconnect_attempts < 5:
            cap = None
            try:
                cap = cv2.VideoCapture(camera_url)
                if not cap.isOpened():
                    raise RuntimeError("камера не открылась")
                logger.info(f"{camera_name}: поток открыт")
                last_time = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        raise RuntimeError("потеряно соединение")
                    now = time.time()
                    if now - last_time >= 1:
                        plates = self.alpr.detect_plates(frame)
                        for p in plates:
                            await self.process_detected_plate(p, camera_name)
                        last_time = now
                    await asyncio.sleep(0.01)
            except Exception as e:
                logger.error(f"{camera_name}: {e}")
                reconnect_attempts += 1
                await asyncio.sleep(min(30, 5 * reconnect_attempts))
            finally:
                if cap:
                    cap.release()
        logger.error(f"{camera_name}: превышено кол‑во попыток.")

    async def process_detected_plate(self, plate_info: Dict[str, Any], camera_name: str):
        plate = plate_info['plate'].replace(' ', '').replace('-', '').upper()
        conf = plate_info['confidence']
        if conf < 0.7 or len(plate) < 5:
            return
        cached, has_osago = self.get_cached_result(plate)
        if not cached:
            has_osago = await self.osago_checker.check_osago(plate)
            self.set_cached_result(plate, has_osago)
        if not has_osago:
            await self.send_violation_notification(plate, camera_name, plate_info)

    async def send_violation_notification(self, plate, camera_name, info):
        msg = (
            f"🚨 Нарушение! (нет ОСАГО)\n"
            f"Камера: {camera_name}\n"
            f"Номер: {plate}\n"
            f"Время: {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            f"Уверенность: {info['confidence']:.2f}"
        )
        self.save_violation_log({
            'plate': plate,
            'camera': camera_name,
            'timestamp': datetime.now().isoformat(),
            'confidence': info['confidence']
        })
        await self.telegram_notifier.send_message_with_retry(msg)

    def save_violation_log(self, data: Dict[str, Any]):
        path = "/app/data/violations.json"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')

    async def start_monitoring(self):
        await self.load_camera_urls_from_easy4ip()  # 🔹 добавлено
        if not self.camera_urls:
            logger.error("Нет доступных камер для мониторинга")
            return
        tasks = [asyncio.create_task(self.process_camera(url, f"Camera_{i+1}"))
                 for i, url in enumerate(self.camera_urls)]
        await asyncio.gather(*tasks, return_exceptions=True)


# FastAPI часть
system = ALPROSAGOSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("ALPR‑система запускается")
    asyncio.create_task(system.start_monitoring())
    yield
    logger.info("ALPR‑система остановлена")

app = FastAPI(title="ALPR OSAGO System", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Разрешить все источники
    allow_credentials=True,
    allow_methods=["*"],          # Разрешить все HTTP-методы
    allow_headers=["*"],          # Разрешить все заголовки
)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "redis": "connected" if system.redis_available else "disconnected",
        "cameras": len(system.camera_urls),
        "memory_cache": len(system.memory_cache)
    }

@app.get("/api/violations/")
async def get_violations(limit: int = 10):
    path = "/app/data/violations.json"
    try:
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()[-limit:]
            return [json.loads(x) for x in lines]
    except FileNotFoundError:
        return []


@app.post("/api/detect-number/")
async def detect_number(frame: UploadFile = File(...)):
    """
    Принять снимок (изображение), распознать ВСЕ номера и проверить ОСАГО по каждому.
    Возвращает JSON со списком результатов.
    """
    try:
        # читаем байты загруженного файла
        image_bytes = await frame.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Невозможно декодировать изображение"}

        # Распознаём номера (может вернуть несколько)
        plates = system.alpr.detect_plates(img)
        if not plates:
            return {"results": [], "message": "Номера не найдены"}

        results = []
        strong_confidence_found = False  # флаг для уверенных номеров

        for p in plates:
            # очистим и нормализуем текст
            plate = p["plate"].replace(" ", "").replace("-", "").upper()
            conf = float(p.get("confidence", 0))

            if conf >= 0.7 and len(plate) >= 5:
                strong_confidence_found = True
                # Проверяем кеш ОСАГО
                cached, has_osago = system.get_cached_result(plate)
                if not cached:
                    has_osago = await system.osago_checker.check_osago(plate)
                    system.set_cached_result(plate, has_osago)

                results.append({
                    "plate": plate,
                    "confidence": conf,
                    "has_osago": has_osago
                })

        # Если нет номеров с достаточной уверенностью
        if not strong_confidence_found:
            return {"results": [], "message": "Номера не найдены"}

        return {"results": results}

    except Exception as e:
        logger.exception("Ошибка при распознавании номера")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
