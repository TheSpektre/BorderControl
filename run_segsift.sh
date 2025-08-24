#!/bin/bash

# Скрипт для запуска трекера БПЛА с использованием камеры и панорамного изображения
# Версия с фиксированным путём для логов и именем скрипта ssift_cam.py

# Конфигурационные параметры
PANORAMA_IMAGE="panorama_output.jpg"  # Путь к файлу панорамы
FRAME_INTERVAL=10                     # Интервал обработки кадров
PYTHON_SCRIPT="segmentation/ssift_cam.py"          # Имя Python-скрипта
LOG_DIR="segmentation/tracking_logs"  # Фиксированная директория для логов

# Функция для проверки ошибок
check_error() {
    if [ $1 -ne 0 ]; then
        echo "[ОШИБКА] Процесс завершился с кодом $1"
        exit $1
    fi
}

# Проверяем наличие необходимых файлов
echo "=== Подготовка к запуску трекера БПЛА ==="
echo "Проверка необходимых файлов..."

if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "[ОШИБКА] Основной скрипт $PYTHON_SCRIPT не найден!"
    exit 1
fi

if [ ! -f "$PANORAMA_IMAGE" ]; then
    echo "[ОШИБКА] Файл панорамы $PANORAMA_IMAGE не найден!"
    exit 1
fi

# Проверяем доступность камеры
echo "Проверка доступности камеры..."
if [ ! -e /dev/video0 ] && [ ! -e /dev/v4l/by-path/*video0 ]; then
    echo "[ОШИБКА] Камера /dev/video0 не обнаружена!"
    exit 1
fi

# Проверяем что Python скрипт действительно использует камеру
if ! grep -q "VideoCapture(0)" "$PYTHON_SCRIPT"; then
    echo "[ПРЕДУПРЕЖДЕНИЕ] Скрипт $PYTHON_SCRIPT возможно не настроен на работу с камерой!"
fi

# Запускаем Python-скрипт с параметрами
echo ""
echo "=== Параметры запуска ==="
echo "Скрипт: $PYTHON_SCRIPT"
echo "Панорама: $PANORAMA_IMAGE"
echo "Интервал кадров: $FRAME_INTERVAL"
echo "Директория логов: $LOG_DIR"
echo ""
echo "Запуск трекера БПЛА..."

python3.8 "$PYTHON_SCRIPT" "$PANORAMA_IMAGE" "$FRAME_INTERVAL"

# Проверяем результат выполнения
check_error $? "$PYTHON_SCRIPT"

echo "Трекер успешно завершил работу"
exit 0
