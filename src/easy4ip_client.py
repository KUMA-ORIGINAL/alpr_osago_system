import asyncio
import os
import uuid
import time
import hashlib

import cv2
import httpx
import logging

logger = logging.getLogger(__name__)


class Easy4ipClient:
    """
    Обёртка вокруг Easy4ip API: получение токена, создание live и извлечение url.
    """
    def __init__(self):
        self.app_id = os.getenv("EASY4IP_APP_ID", "")
        self.app_secret = os.getenv("EASY4IP_APP_SECRET", "")
        self.data_center = os.getenv("EASY4IP_DATA_CENTER", "sg")

    def _sign(self, timestamp: int, nonce: str) -> str:
        raw = f"time:{timestamp},nonce:{nonce},appSecret:{self.app_secret}"
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    async def _request(self, endpoint: str, params: dict):
        timestamp = int(time.time())
        nonce = str(uuid.uuid4())
        sign = self._sign(timestamp, nonce)

        body = {
            "system": {
                "ver": "1.0",
                "appId": self.app_id,
                "sign": sign,
                "time": timestamp,
                "nonce": nonce,
            },
            "id": str(uuid.uuid4()),
            "params": params or {},
        }

        url = f"https://openapi-{self.data_center}.easy4ip.com/openapi/{endpoint}"
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(url, json=body)
            r.raise_for_status()
            return r.json()

    async def get_access_token(self) -> str:
        data = await self._request("accessToken", {})
        result = data.get("result", {})
        print(result)
        if result.get("code") != "0":
            raise RuntimeError(result.get("msg", f"Ошибка токена: {result!r}"))
        token = result["data"]["accessToken"]
        logger.info("Easy4ip токен получен")
        return token

    async def get_or_create_live_url(self, token: str, device_id: str, channel_id: str="0") -> str:
        """Создаёт поток, если нет, и возвращает hls/live url."""
        # пробуем создать
        bind = await self._request("bindDeviceLive", {
            "token": token,
            "deviceId": device_id,
            "channelId": channel_id,
            "streamId": 1,
            "liveMode": "proxy"
        })
        result = bind.get("result", {})
        code = result.get("code")
        if code == "0":
            data = result["data"]
        elif code == "LV1001":
            # уже существует — получим инфо
            info = await self._request("getLiveStreamInfo", {
                "token": token, "deviceId": device_id, "channelId": channel_id
            })
            result = info.get("result", {})
            if result.get("code") != "0":
                raise RuntimeError(result.get("msg", "getLiveStreamInfo error"))
            data = result["data"]["streams"][0]
        else:
            raise RuntimeError(result.get("msg", "bindDeviceLive error"))
        return data.get("hls") or data.get("rtmp") or ""


async def get_stream_url(device_id: str):
    """
    Получаем HLS или RTMP URL через Easy4ipClient.
    """
    client = Easy4ipClient()
    token = await client.get_access_token()
    url = await client.get_or_create_live_url(token, device_id)
    return url


async def main():
    # Читаем переменные окружения
    device_ids = '8L07D68PAZ5A24C'

    # Берём первый ID (если переменных несколько через запятую)
    device_id = device_ids.split(",")[0].strip()

    # Получаем потоковый URL
    stream_url = await get_stream_url(device_id)
    if not stream_url:
        print("❌ Не удалось получить потоковый URL")
        return

    print(f"🎥 Открываю поток: {stream_url}")

    # подключаемся к потоку через OpenCV
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        print("⚠️ Ошибка: не удалось открыть поток.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⛔ Поток прервался или закончился")
            break

        cv2.imshow("Easy4ip Live Stream", frame)

        # выход по клавише "q"
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())