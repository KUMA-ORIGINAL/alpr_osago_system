import cv2
import numpy as np
import re
import logging
from typing import List, Dict
from fast_alpr import ALPR

logger = logging.getLogger(__name__)


class ALPRProcessor:
    def __init__(self, use_gpu: bool = False):
        """Инициализация ALPR и OCR для левой части"""
        logger.info("Инициализация ALPR процессора...")

        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )

        logger.info("✅ ALPR процессор и OCR региона (PaddleOCR) инициализированы")

    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """Основное распознавание"""
        results = self.alpr.predict(frame)
        plates = []

        for result in results:
            plate_text = result.ocr.text.upper()

            bbox = result.detection.bounding_box
            x1, y1, x2, y2 = map(int, [bbox.x1, bbox.y1, bbox.x2, bbox.y2])
            cropped = frame[y1:y2, x1:x2]

            if cropped.size == 0:
                continue

            h, w = cropped.shape[:2]
            left_crop = cropped[:, : int(w * 0.35)]

            region_text = self.alpr.ocr.predict(left_crop).text

            print('region_text', region_text)

            if plate_text[0].isdigit():
                prefix = region_text.split("KG")[0]
                digits = re.findall(r"\d", prefix)
                if digits:
                    num = int(digits[-1])
                    if 1 <= num <= 9:
                        region_text = f"0{num}KG"
                    else:
                        region_text = ""
                else:
                    region_text = "01KG"
            else:
                region_text = ""

            full_plate = region_text + plate_text

            plates.append(
                {
                    "plate": full_plate[:10],
                    "region": region_text,
                    "number": plate_text,
                    "confidence": float(result.ocr.confidence),
                    "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                }
            )

            logger.info(
                f"🚗 Регион: {region_text} | Основной номер: {plate_text} | "
                f"Уверенность: {result.ocr.confidence:.1%}"
            )

        return plates


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        processor = ALPRProcessor(use_gpu=False)
        test_image = "cars_photo/test_car.jpg"
        frame = cv2.imread(test_image)

        if frame is None:
            logger.error(f"❌ Не удалось загрузить: {test_image}")
        else:
            logger.info(f"📷 Обработка изображения: {test_image}")

            plates = processor.detect_plates(frame)

            annotated_frame = processor.alpr.draw_predictions(frame)
            output_path = "resul.jpg"
            cv2.imwrite(output_path, annotated_frame)

            if plates:
                print("\n" + "=" * 70)
                print("РЕЗУЛЬТАТЫ РАСПОЗНАВАНИЯ")
                print("=" * 70)

                for i, plate in enumerate(plates, 1):
                    print(f"\n🚗 Номер #{i}:")
                    print(f"  Полный номер: {plate['plate']}")
                    print(f"  Регион: {plate['region']}")
                    print(f"  Номер: {plate['number']}")
                    print(f"  Уверенность: {plate['confidence']:.1%}")
                    print(f"  Координаты: ({plate['bbox']['x1']}, {plate['bbox']['y1']}) - "
                          f"({plate['bbox']['x2']}, {plate['bbox']['y2']})")
            else:
                print("\n❌ Номера не обнаружены")

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}", exc_info=True)
