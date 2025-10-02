import cv2
import numpy as np
import time
import signal
import sys
import os

# Глобальная переменная для обработки прерывания
interrupt_flag = False

def signal_handler(sig, frame):
    global interrupt_flag
    print("\nПрерывание получено, завершаем работу и сохраняем панораму...")
    interrupt_flag = True

def create_panorama_from_camera(camera_id=0, frame_interval=10, roi_expansion=150, output_path="panorama_output.jpg"):
    """Оптимизированная версия для Jetson Orin Nano с поддержкой CUDA. Источник - камера."""
    # Устанавливаем обработчик сигналов для прерывания
    signal.signal(signal.SIGINT, signal_handler)
    
    # Получаем путь к директории скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir == '':
        script_dir = os.getcwd()
    
    # Создаем полные пути для сохранения файлов
    output_path = os.path.join(script_dir, output_path)
    output_dir = os.path.dirname(output_path)
    
    # Проверяем доступность CUDA
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    print(f"CUDA доступен: {use_cuda}")

    # Инициализация видеопотока с камеры
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise ValueError(f"Ошибка открытия камеры {camera_id}!")

    # Настройка параметров для Jetson
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0:
        fps = 20  # Значение по умолчанию
    
    print(f"Размер кадра: {width}x{height}, FPS: {fps}")
    print(f"Файлы будут сохранены в: {script_dir}")

    # Создаем папку для сохранения, если её нет
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Инициализация видео писателей
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Видео с исходной камеры
    camera_video_path = os.path.join(script_dir, f"camera_feed_{timestamp}.avi")
    camera_writer = cv2.VideoWriter(
        camera_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )
    
    # Видео процесса склейки (будет настроен позже)
    stitching_writer = None
    stitching_video_path = None

    # Оптимизированные параметры для Orin Nano
    sift = cv2.SIFT_create(
        nfeatures=17000,  # Уменьшено для экономии ресурсов
        nOctaveLayers=3,
        contrastThreshold=0.04,
        edgeThreshold=10
    )

    # FLANN параметры для ARM
    flann = cv2.FlannBasedMatcher(
        dict(algorithm=1, trees=3),  # Меньше деревьев для ускорения
        dict(checks=30)              # Меньше проверок
    )

    panorama = None
    gray_panorama = None
    last_h_adjusted = None
    frame_count = 0
    start_time = time.time()

    # Преаллокация памяти для часто используемых объектов
    corners_template = np.array([[0, 0], [0, height], [width, height], [width, 0]], dtype=np.float32)

    print("Нажмите Ctrl+C для завершения и сохранения панорамы...")
    
    while cap.isOpened() and not interrupt_flag:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка чтения кадра с камеры")
            break

        # Сохраняем исходный кадр в видео
        camera_writer.write(frame)

        frame_count += 1
        if frame_count % frame_interval != 0:
            continue

        print(f"Обработка кадра {frame_count}...", end='\r')

        # Преобразование в grayscale (оптимизированное)
        if use_cuda:
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            gray_frame = cv2.cuda.cvtColor(gpu_frame, cv2.COLOR_BGR2GRAY).download()
        else:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if panorama is None:
            panorama = frame.copy()
            gray_panorama = gray_frame.copy()
            
            # Инициализируем видео писатель для процесса склейки после создания первого кадра панорамы
            stitching_video_path = os.path.join(script_dir, f"stitching_process_{timestamp}.avi")
            stitching_writer = cv2.VideoWriter(
                stitching_video_path,
                cv2.VideoWriter_fourcc(*'XVID'),
                max(1, fps // frame_interval),  # Понижаем FPS т.к. записываем не каждый кадр, но не меньше 1
                (panorama.shape[1], panorama.shape[0])
            )
            continue

        # ROI обработка с преаллокацией
        if last_h_adjusted is not None:
            warped_corners = cv2.perspectiveTransform(
                corners_template.reshape(-1, 1, 2), 
                last_h_adjusted
            ).reshape(-1, 2)
            
            x_min, y_min = warped_corners.min(axis=0).astype(int) - roi_expansion
            x_max, y_max = warped_corners.max(axis=0).astype(int) + roi_expansion
            
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(panorama.shape[1], x_max)
            y_max = min(panorama.shape[0], y_max)
            
            roi = gray_panorama[y_min:y_max, x_min:x_max]
            kp_pano, des_pano = sift.detectAndCompute(roi, None)
            
            if des_pano is not None:
                for kp in kp_pano:
                    kp.pt = (kp.pt[0] + x_min, kp.pt[1] + y_min)
        else:
            kp_pano, des_pano = sift.detectAndCompute(gray_panorama, None)

        kp_frame, des_frame = sift.detectAndCompute(gray_frame, None)

        # Быстрая проверка ключевых точек
        if des_pano is None or des_frame is None or len(des_pano) < 10 or len(des_frame) < 10:
            print(f"Пропуск кадра {frame_count} - недостаточно ключевых точек")
            continue

        # Ускоренное сопоставление
        matches = flann.knnMatch(des_pano, des_frame, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good_matches) < 8:  # Уменьшенный порог
            print(f"Пропуск кадра {frame_count} - недостаточно совпадений")
            continue

        # Векторизованные операции для ускорения
        src_pts = np.float32([kp_pano[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 3.0)  # Уменьшенный ransacReprojThreshold

        if H is None:
            print(f"Пропуск кадра {frame_count} - не найдена гомография")
            continue

        # Оптимизированное вычисление границ
        warped_corners = cv2.perspectiveTransform(corners_template.reshape(-1, 1, 2), H)
        x_min = int(min(0, warped_corners[:, :, 0].min()))
        y_min = int(min(0, warped_corners[:, :, 1].min()))
        x_max = int(max(panorama.shape[1], warped_corners[:, :, 0].max()))
        y_max = int(max(panorama.shape[0], warped_corners[:, :, 1].max()))

        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        H_adjusted = translation @ H
        last_h_adjusted = H_adjusted

        # Ускоренное преобразование перспективы
        if use_cuda:
            gpu_frame.upload(frame)
            gpu_warped = cv2.cuda.warpPerspective(
                gpu_frame, 
                H_adjusted, 
                (int(x_max - x_min), int(y_max - y_min)))
            warped_frame = gpu_warped.download()
        else:
            warped_frame = cv2.warpPerspective(
                frame, 
                H_adjusted, 
                (int(x_max - x_min), int(y_max - y_min)))

        # Оптимизированное смешивание
        mask = (warped_frame[..., 0] != 0) | (warped_frame[..., 1] != 0) | (warped_frame[..., 2] != 0)
        panorama_adjusted = np.zeros_like(warped_frame)
        panorama_adjusted[-y_min:panorama.shape[0]-y_min, -x_min:panorama.shape[1]-x_min] = panorama

        panorama = np.where(mask[..., None], warped_frame, panorama_adjusted)
        gray_panorama = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)

        # Сохраняем текущее состояние панорамы в видео процесса склейки
        if stitching_writer is not None:
            # Создаем копию для отображения прогресса
            display_panorama = panorama.copy()
            
            # Добавляем информацию о прогрессе
            cv2.putText(display_panorama, f"Frames processed: {frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_panorama, f"Panorama size: {display_panorama.shape[1]}x{display_panorama.shape[0]}", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Записываем кадр в видео процесса склейки
            stitching_writer.write(display_panorama)

    # Завершаем запись видео
    camera_writer.release()
    if stitching_writer is not None:
        stitching_writer.release()
    
    cap.release()
    
    # Финализация и сохранение
    if panorama is not None:
        # Обрезка черных границ
        gray = cv2.cvtColor(panorama, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            panorama = panorama[y:y+h, x:x+w]

        cv2.imwrite(output_path, panorama)
        elapsed_time = time.time() - start_time
        print(f"\nПанорама сохранена как: {output_path}")
        print(f"Видео с камеры сохранено как: {camera_video_path}")
        if stitching_video_path:
            print(f"Видео процесса склейки сохранено как: {stitching_video_path}")
        print(f"Всего кадров обработано: {frame_count}")
        print(f"Общее время: {elapsed_time:.2f} сек")
        print(f"Скорость обработки: {frame_count/elapsed_time:.2f} FPS")
        return panorama
    else:
        print("Не удалось создать панораму")
        return None

if __name__ == "__main__":
    # Используем камеру по умолчанию (0)
    create_panorama_from_camera(camera_id=0, frame_interval=10, roi_expansion=150)