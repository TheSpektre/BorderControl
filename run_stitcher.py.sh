#!/bin/bash

# Скрипт для запуска создания панорамы с камеры
# Панорама сохраняется в /stitcher/panorama_output.jpg

# Директория для сохранения панорамы
PANORAMA_DIR="~/aii_modules/BorderControl/stitcher"
PANORAMA_PATH="$PANORAMA_DIR/panorama_output.jpg"

# Проверяем существование директории, создаем если нет
if [ ! -d "$PANORAMA_DIR" ]; then
    echo "Создаем директорию для панорамы: $PANORAMA_DIR"
    sudo mkdir -p "$PANORAMA_DIR"
    sudo chmod 777 "$PANORAMA_DIR"
fi

# Переходим в директорию со скриптом (предполагаем, что скрипт в той же папке)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Запуск создания панорамы с камеры..."
echo "Панорама будет сохранена в: $PANORAMA_PATH"
echo "Для завершения нажмите Ctrl+C"

# Запускаем Python скрипт
python camstitcher.py

# Проверяем успешность выполнения
if [ $? -eq 0 ]; then
    echo "Панорама успешно создана и сохранена в: $PANORAMA_PATH"
else
    echo "Ошибка при создании панорамы"
    exit 1
fi
