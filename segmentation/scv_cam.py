import cv2
import numpy as np
import time
import os
from datetime import datetime

class RoadFieldDetector:
    def __init__(self):
        # Цветовые диапазоны в HSV
        self.lower_field = np.array([35, 40, 40])
        self.upper_field = np.array([85, 255, 255])
        self.lower_road = np.array([0, 0, 100])
        self.upper_road = np.array([179, 50, 220])
        self.kernel = np.ones((5,5), np.uint8)
        
        # Настройки логов
        self.log_dir = "segmentation/segmentation"
        os.makedirs(self.log_dir, exist_ok=True)
        self.setup_new_log()
        
        # Статистика
        self.frame_count = 0
        self.start_time = time.time()
        self.video_writer = None
        self.video_initialized = False

    def init_video_writer(self, frame_shape, fps=20):
        """Инициализация видео writer"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join(self.log_dir, f"segmentation_{timestamp}.mp4")
        
        # Определяем кодек и создаем VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frame_shape[:2]
        
        self.video_writer = cv2.VideoWriter(
            video_path, 
            fourcc, 
            fps,
            (width, height)
        )
        self.video_initialized = True
        print(f"Video recording started: {video_path}")
    
    def setup_new_log(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"{self.log_dir}/detection_{timestamp}.txt"
        with open(self.log_path, 'w') as f:
            f.write("Time(ms)\tFrame\tClass\tConfidence\tFieldArea\tRoadArea\n")
    
    def write_log_entry(self, time_ms, frame_num, class_id, confidence, field_area, road_area):
        with open(self.log_path, 'a') as f:
            f.write(f"{time_ms}\t{frame_num}\t{class_id}\t{confidence:.2f}\t{field_area}\t{road_area}\n")
    
    def process_frame(self, frame):
        self.frame_count += 1
        current_time_ms = int((time.time() - self.start_time) * 1000)
        
        # Основная обработка
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Создание масок
        mask_field = cv2.inRange(hsv, self.lower_field, self.upper_field)
        mask_road = cv2.inRange(hsv, self.lower_road, self.upper_road)
        
        # Морфологические операции
        mask_field = cv2.morphologyEx(mask_field, cv2.MORPH_CLOSE, self.kernel)
        mask_road = cv2.morphologyEx(mask_road, cv2.MORPH_OPEN, self.kernel)
        
        # Расчет площадей
        field_area = cv2.countNonZero(mask_field)
        road_area = cv2.countNonZero(mask_road)
        total_area = frame.shape[0] * frame.shape[1]
        
        # Определение класса
        if road_area > 0.2 * total_area:  # Если дорога занимает >10% площади
            class_id = 1
            confidence = road_area / (road_area + field_area + 1e-5)
        else:
            class_id = 0
            confidence = field_area / (field_area + road_area + 1e-5)
        
        # Логирование
        self.write_log_entry(current_time_ms, self.frame_count, class_id, confidence, field_area, road_area)
        
        # Визуализация
        result = frame.copy()
        class_text = "ROAD" if class_id == 1 else "FIELD"
        color = (0, 0, 255) if class_id == 1 else (0, 255, 0)
        
        cv2.putText(result, class_text, (20, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        contours = cv2.findContours(mask_road if class_id == 1 else mask_field, 
                                  cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        for cnt in contours:
            if cv2.contourArea(cnt) > 1000:
                cv2.drawContours(result, [cnt], -1, color, 2)
        
        # Добавление временной метки и статистики
        info_text = f"Time: {current_time_ms/1000:.1f}s | Frame: {self.frame_count}"
        cv2.putText(result, info_text, (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        stats_text = f"Field: {field_area} | Road: {road_area}"
        cv2.putText(result, stats_text, (20, 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
         # Инициализация видео writer при первом кадре
        if not self.video_initialized:
            self.init_video_writer(result.shape, fps = cv2.VideoCapture(0).get(cv2.CAP_PROP_FPS))
        
        # Запись кадра в видео
        if self.video_writer is not None:
            self.video_writer.write(result)
        
        return result

def process_camera():
    detector = RoadFieldDetector()
    
    # Открытие камеры (0 - индекс камеры по умолчанию)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Ошибка открытия камеры")
        return
    
    # Получение разрешения камеры
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Обработка видеопотока с камеры")
    print(f"Размер: {width}x{height}, FPS: {fps:.1f}")
    print(f"Логи будут сохранены в: {detector.log_path}")
    
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра с камеры")
            break
        
        # Обработка кадра
        result_frame = detector.process_frame(frame)
        
        # Отображение
        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Освобождение ресурсов
    cap.release()
    print(f"\nОбработка завершена. Логи сохранены в: {detector.log_path}")
    print(f"Всего обработано кадров: {detector.frame_count}")

if __name__ == "__main__":
    process_camera()
