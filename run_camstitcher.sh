#!/bin/bash

# Скрипт запуска camstitcher.py для создания панорамы с камеры
# Панорама сохраняется в /stitcher/panorama_output.jpg

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Функция для вывода сообщений
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}


# Создаем директорию для панорамы
PANORAMA_DIR="/stitcher"
PANORAMA_PATH="$PANORAMA_DIR/panorama_output.jpg"

print_status "Проверяем директорию для панорамы: $PANORAMA_DIR"
if [ ! -d "$PANORAMA_DIR" ]; then
    print_status "Создаем директорию для панорамы..."
    sudo mkdir -p "$PANORAMA_DIR"
    if [ $? -ne 0 ]; then
        print_error "Не удалось создать директорию $PANORAMA_DIR"
        exit 1
    fi
    sudo chmod 777 "$PANORAMA_DIR"
    print_status "Директория создана и права установлены"
else
    print_status "Директория уже существует"
fi

# Проверяем доступность камеры
print_status "Проверяем доступность камеры..."
if [ ! -e "/dev/video0" ]; then
    print_warning "Камера /dev/video0 не найдена. Пытаемся найти доступные камеры..."
    CAM_DEVICES=$(ls /dev/video* 2>/dev/null)
    if [ -z "$CAM_DEVICES" ]; then
        print_error "Не найдено ни одной камеры!"
        exit 1
    else
        print_status "Найдены камеры: $CAM_DEVICES"
        print_status "Будет использована первая доступная камера"
    fi
else
    print_status "Камера /dev/video0 доступна"
fi

# Проверяем зависимости Python
print_status "Проверяем зависимости Python..."
if ! python3 -c "import cv2; import numpy; import signal" 2>/dev/null; then
    print_error "Не найдены необходимые Python библиотеки:"
    print_error "Установите: sudo apt install python3-opencv python3-numpy"
    exit 1
fi

print_status "Все зависимости удовлетворены"
print_status "Запускаем создание панорамы с камеры..."
print_status "Панорама будет сохранена в: $PANORAMA_PATH"
echo ""
print_warning "Для завершения и сохранения панорамы нажмите Ctrl+C"
echo ""

# Запускаем Python скрипт
python ~/stitcher/camstitcher.py

