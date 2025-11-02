import cv2
import numpy as np
import easyocr
import re
import logging
from typing import List, Dict, Optional
from fast_alpr import ALPR, BaseOCR, OcrResult, ALPRResult

logger = logging.getLogger(__name__)


class EasyOCRPlateRecognizer(BaseOCR):
    """OCR на базе EasyOCR для киргизских номерных знаков"""

    def __init__(self, gpu: bool = False):
        """
        Инициализация EasyOCR

        Args:
            gpu: Использовать GPU (требует CUDA)
        """
        logger.info("Инициализация EasyOCR...")
        try:
            # Инициализируем EasyOCR с английским языком
            # detail=0 возвращает только текст без координат
            self.reader = easyocr.Reader(
                ['en'],
                gpu=gpu,
                verbose=False
            )
            logger.info(f"✅ EasyOCR инициализирован (GPU: {gpu})")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации EasyOCR: {e}")
            raise

    def preprocess_plate(self, plate_img: np.ndarray) -> np.ndarray:
        """Предобработка изображения номера"""
        # Конвертируем в оттенки серого
        if len(plate_img.shape) == 3:
            gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_img

        # Увеличиваем размер для лучшего распознавания
        scale_factor = 2
        height, width = gray.shape
        resized = cv2.resize(
            gray,
            (width * scale_factor, height * scale_factor),
            interpolation=cv2.INTER_CUBIC
        )

        # Улучшаем контраст с CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(resized)

        # Убираем шум
        denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)

        # Конвертируем обратно в BGR для EasyOCR
        return cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    def predict(self, cropped_plate: np.ndarray) -> Optional[OcrResult]:
        """
        Распознавание текста на номерном знаке

        Args:
            cropped_plate: Вырезанное изображение номера

        Returns:
            OcrResult или None
        """
        if cropped_plate is None or cropped_plate.size == 0:
            return None

        try:
            # Предобработка
            processed = self.preprocess_plate(cropped_plate)

            # Распознавание с EasyOCR
            # detail=1 возвращает [координаты, текст, уверенность]
            results = self.reader.readtext(
                processed,
                detail=1,
                paragraph=False,
                allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            )

            if not results:
                # Пробуем на оригинальном изображении
                results = self.reader.readtext(
                    cropped_plate,
                    detail=1,
                    paragraph=False,
                    allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
                )

            if not results:
                return None

            # Собираем все распознанные фрагменты
            texts = []
            confidences = []

            for bbox, text, conf in results:
                # Очищаем текст
                cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
                if cleaned:
                    texts.append(cleaned)
                    confidences.append(conf)

            if not texts:
                return None

            # Объединяем текст
            plate_text = ''.join(texts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            # Проверяем минимальную длину
            if len(plate_text) >= 5:
                logger.debug(f"OCR результат: '{plate_text}' (conf: {avg_confidence:.2f})")
                return OcrResult(text=plate_text, confidence=float(avg_confidence))

            return None

        except Exception as e:
            logger.error(f"❌ Ошибка EasyOCR: {e}", exc_info=True)
            return None


class CustomALPR(ALPR):
    """Расширенный ALPR с улучшенной обработкой"""

    def predict(self, frame: np.ndarray | str) -> List[ALPRResult]:
        if isinstance(frame, str):
            img = cv2.imread(frame)
            if img is None:
                raise ValueError(f"Не удалось загрузить изображение: {frame}")
        else:
            img = frame

        plate_detections = self.detector.predict(img)
        alpr_results: List[ALPRResult] = []

        for detection in plate_detections:
            bbox = detection.bounding_box
            x1, y1 = max(bbox.x1, 0), max(bbox.y1, 0)
            x2, y2 = min(bbox.x2, img.shape[1]), min(bbox.y2, img.shape[0])

            # Увеличенный отступ для захвата всего номера
            padding = 15
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img.shape[1], x2 + padding)
            y2 = min(img.shape[0], y2 + padding)

            cropped_plate = img[y1:y2, x1:x2]

            if cropped_plate.size == 0:
                continue

            ocr_result = self.ocr.predict(cropped_plate)

            if ocr_result:
                alpr_results.append(ALPRResult(detection=detection, ocr=ocr_result))

        return alpr_results


class ALPRProcessor:
    def __init__(self, use_gpu: bool = False):
        """
        Инициализация процессора ALPR с EasyOCR

        Args:
            use_gpu: Использовать GPU для EasyOCR (требует CUDA)
        """
        try:
            logger.info("Инициализация ALPR процессора...")

            self.alpr = CustomALPR(
                detector_model="yolo-v9-t-384-license-plate-end2end",
                ocr=EasyOCRPlateRecognizer(gpu=use_gpu)
            )

            logger.info("✅ ALPR процессор успешно инициализирован")
        except Exception as e:
            logger.error(f"❌ Ошибка инициализации ALPR: {e}")
            raise

    def detect_plates(self, frame: np.ndarray) -> List[Dict]:
        """
        Обнаружение и распознавание номерных знаков
        """
        try:
            results = self.alpr.predict(frame)
            plates = []

            for result in results:
                plate_text = result.ocr.text

                # Попытка исправить формат — вставляем "KG" после 2 цифр, если его нет
                plate_text = self._normalize_plate_format(plate_text)

                # Проверяем формат
                if self._validate_plate_format(plate_text):
                    plate_info = {
                        'plate': plate_text,
                        'region': plate_text[:2],
                        'number': plate_text[4:],  # пропускаем '01KG'
                        'confidence': float(result.ocr.confidence),
                        'bbox': {
                            'x1': result.detection.bounding_box.x1,
                            'y1': result.detection.bounding_box.y1,
                            'x2': result.detection.bounding_box.x2,
                            'y2': result.detection.bounding_box.y2
                        }
                    }
                    plates.append(plate_info)

                    logger.info(
                        f"🚗 Распознан: {plate_text} | "
                        f"Регион: {plate_text[:2]} | "
                        f"Уверенность: {result.ocr.confidence:.1%}"
                    )
                else:
                    logger.warning(f"⚠️ Некорректный формат: '{plate_text}'")

            return plates

        except Exception as e:
            logger.error(f"❌ Ошибка распознавания: {e}", exc_info=True)
            return []

    def _normalize_plate_format(self, plate_text: str) -> str:
        """
        Попытка восстановить правильный формат киргизского номера (01KG564ABF)
        даже если OCR перепутал порядок цифр и букв.
        """
        if not plate_text:
            return plate_text

        # Убираем возможные лишние пробелы и KG
        text = re.sub(r'[^A-Z0-9]', '', plate_text.upper())
        text = text.replace('KG', '')

        # Проверяем, есть ли хотя бы 2 цифры и 3 буквы
        digits = re.findall(r'\d', text)
        letters = re.findall(r'[A-Z]', text)

        if len(digits) < 5 or len(letters) < 3:
            return plate_text  # вернуть как есть, если данных мало

        region = ''.join(digits[:2])
        num_part = ''.join(digits[2:5])
        letter_part = ''.join(letters[-3:])  # последние 3 буквы обычно правильные

        normalized = f'{region}KG{num_part}{letter_part}'
        return normalized

    def _validate_plate_format(self, plate_text: str) -> bool:
        """
        Проверка формата киргизского номера:
        2 цифры региона + KG + 3 цифры + 3 буквы.
        Пример: 01KG564ABF
        """
        if not plate_text:
            return False

        # Разрешаем только заглавные буквы
        plate_text = plate_text.upper().strip()

        pattern = r'^[0-9]{2}KG[0-9]{3}[A-Z]{3}$'
        if not re.match(pattern, plate_text):
            return False

        # Проверяем диапазон региона
        try:
            region_num = int(plate_text[:2])
            return 1 <= region_num <= 9
        except ValueError:
            return False

    def draw_plates(self, frame: np.ndarray, plates: List[Dict]) -> np.ndarray:
        """
        Отрисовка обнаруженных номеров на изображении

        Args:
            frame: Исходное изображение
            plates: Список распознанных номеров

        Returns:
            Изображение с отрисованными номерами
        """
        result_frame = frame.copy()

        for plate in plates:
            bbox = plate['bbox']
            x1, y1 = int(bbox['x1']), int(bbox['y1'])
            x2, y2 = int(bbox['x2']), int(bbox['y2'])

            # Рисуем рамку
            color = (0, 255, 0)  # Зеленый
            thickness = 2
            cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, thickness)

            # Добавляем текст
            text = f"{plate['plate']} ({plate['confidence']:.0%})"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.8
            font_thickness = 2

            # Фон для текста
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, font_thickness)
            cv2.rectangle(
                result_frame,
                (x1, y1 - text_height - 10),
                (x1 + text_width, y1),
                color,
                -1
            )

            # Текст
            cv2.putText(
                result_frame,
                text,
                (x1, y1 - 5),
                font,
                font_scale,
                (0, 0, 0),
                font_thickness
            )

        return result_frame


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Инициализация (gpu=True если есть CUDA)
        processor = ALPRProcessor(use_gpu=False)

        # Загрузка тестового изображения
        test_image = "test_car.jpg"
        frame = cv2.imread(test_image)

        if frame is None:
            logger.error(f"❌ Не удалось загрузить: {test_image}")
        else:
            logger.info(f"📷 Обработка изображения: {test_image}")

            # Распознавание номеров
            plates = processor.detect_plates(frame)

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

                # Сохранение результата с отрисовкой
                result_frame = processor.draw_plates(frame, plates)
                output_path = "result.jpg"
                cv2.imwrite(output_path, result_frame)
                logger.info(f"✅ Результат сохранен: {output_path}")

            else:
                print("\n❌ Номера не обнаружены")

    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}", exc_info=True)
