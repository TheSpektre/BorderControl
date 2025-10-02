import cv2
import numpy as np
from collections import deque
import time
import csv
import sys
from datetime import datetime
import os

class OptimizedDroneTracker:
    def __init__(self, panorama_path, frame_interval=5, max_history=100):
        self.panorama = cv2.imread(panorama_path)
        if self.panorama is None:
            raise ValueError(f"Не удалось загрузить панораму: {panorama_path}")
        
        self.frame_interval = max(1, frame_interval)
        self.frame_counter = 0
        self.max_history = max_history
        
        self.detector_0 = cv2.ORB_create(10000)
        self.detector_1 = cv2.ORB_create(nfeatures=5000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        gray_pano = cv2.cvtColor(self.panorama, cv2.COLOR_BGR2GRAY)
        self.pano_kp, self.pano_des = self.detector_0.detectAndCompute(gray_pano, None)
        
        self.positions = deque(maxlen=max_history)
        self.last_position = None
        
        self.init_log_files()
    
    def init_log_files(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "segmentation/tracking_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        self.txt_log_path = os.path.join(log_dir, f"tracking_log_{timestamp}.txt")
        with open(self.txt_log_path, 'w') as f:
            f.write("Time, X Coordinate, Y Coordinate\n")
        
        self.csv_log_path = os.path.join(log_dir, f"tracking_log_{timestamp}.csv")
        with open(self.csv_log_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Timestamp", "X", "Y"])
    
    def log_position(self, position):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        x, y = position
        
        with open(self.txt_log_path, 'a') as f:
            f.write(f"{timestamp}, {x:.2f}, {y:.2f}\n")
        
        with open(self.csv_log_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([timestamp, f"{x:.2f}", f"{y:.2f}"])
    
    def process_frame(self, frame):
        self.frame_counter += 1
        
        if self.frame_counter % self.frame_interval != 0:
            return self.last_position
        
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_kp, frame_des = self.detector_1.detectAndCompute(gray_frame, None)
        
        if frame_des is None or len(frame_kp) < 10:
            print('[Ошибка] Недостаточно ключевых точек')
            return self.last_position
        
        matches = self.matcher.match(self.pano_des, frame_des)
        if len(matches) < 10:
            print('[Предупреждение] Мало совпадений')
            return self.last_position
        
        matches = sorted(matches, key=lambda x: x.distance)[:30]
        src_pts = np.float32([self.pano_kp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
        
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
        if H is None:
            print('[Ошибка] Ошибка гомографии')
            return self.last_position
        
        h, w = frame.shape[:2]
        center = np.array([[w/2, h/2]], dtype=np.float32).reshape(-1,1,2)
        panorama_center = cv2.perspectiveTransform(center, H)[0][0]
        
        self.positions.append(panorama_center)
        self.last_position = panorama_center
        self.log_position(panorama_center)
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Позиция: X={panorama_center[0]:.1f}, Y={panorama_center[1]:.1f}")
        
        return panorama_center
    
    def get_visualization(self, frame, scale=0.3):
        if self.last_position is None:
            return self.panorama.copy()
        
        vis = self.panorama.copy()
        
        # Уменьшенный размер точки с 100 до 10 пикселей
        point_radius = 10
        for i, pos in enumerate(self.positions):
            color = (0, 255, 0) if i == len(self.positions)-1 else (0, 0, 200)
            cv2.circle(vis, (int(pos[0]), int(pos[1])), point_radius, color, -1)
        
        small_frame = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = small_frame.shape[:2]
        vis[10:h+10, 10:w+10] = small_frame
        
        cv2.putText(vis, f"X: {self.last_position[0]:.1f}, Y: {self.last_position[1]:.1f}", 
                   (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return vis

def camera_tracking(panorama_path, output_path=None, frame_interval=5):
    tracker = OptimizedDroneTracker(panorama_path, frame_interval)
    
    # Открытие камеры (0 - индекс камеры по умолчанию)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("[Ошибка] Не удалось открыть камеру")
        return
    
    # Установка разрешения камеры (по желанию)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    writer = None
    if output_path:
        fps = 30 / frame_interval  # Предполагаем 30 FPS с камеры
        size = (tracker.panorama.shape[1], tracker.panorama.shape[0])
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, size)
    
    print("Начало обработки с камеры... (q - выход, p - пауза)")
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("[Ошибка] Не удалось получить кадр с камеры")
                break
            
            position = tracker.process_frame(frame)
            
            if position is not None and tracker.frame_counter % tracker.frame_interval == 0:
                vis = tracker.get_visualization(frame)
                
                if writer is not None:
                    writer.write(vis)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused
            print("Пауза" if paused else "Продолжение")
    
    cap.release()
    if writer is not None:
        writer.release()
    
    print("Обработка завершена")

if __name__ == "__main__":
    panorama = sys.argv[1]   
    
    start_time = time.time()
    camera_tracking(panorama, frame_interval=int(sys.argv[2]))
    elapsed_time = time.time() - start_time
    
    print(f"\nВремя выполнения: {elapsed_time:.2f} сек")
    print(f"Логи сохранены в 'tracking_logs'")
