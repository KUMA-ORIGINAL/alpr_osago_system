from fast_alpr import ALPR
import logging

logger = logging.getLogger(__name__)


class ALPRProcessor:
    def __init__(self):
        self.alpr = ALPR(
            detector_model="yolo-v9-t-384-license-plate-end2end",
            ocr_model="cct-xs-v1-global-model",
        )
        logger.info("ALPR процессор инициализирован")

    def detect_plates(self, frame):
        """Обнаружение номерных знаков в кадре"""
        try:
            results = self.alpr.predict(frame)
            plates = []

            for result in results:
                plate_info = {
                    'plate': result.ocr.text,
                    'confidence': float(result.ocr.confidence),
                    'bbox': {
                        'x1': result.detection.bounding_box.x1,
                        'y1': result.detection.bounding_box.y1,
                        'x2': result.detection.bounding_box.x2,
                        'y2': result.detection.bounding_box.y2
                    }
                }
                plates.append(plate_info)

            return plates
        except Exception as e:
            logger.error(f"Ошибка распознавания номеров: {e}")
            return []
