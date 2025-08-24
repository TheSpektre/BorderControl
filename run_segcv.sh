#!/bin/bash

# Скрипт для запуска детектора дороги/поля с камеры
# Переходим в директорию скрипта (если нужно)
cd "$(dirname "$0")" || exit

# Проверяем существование файла
if [ ! -f "segmentation/scv_cam.py" ]; then
    echo "Ошибка: файл segmentation/scv_cam.py не найден!"
    exit 1
fi

# Проверяем доступность камеры
if [ ! -e /dev/video0 ]; then
    echo "Ошибка: камера /dev/video0 не обнаружена!"
    exit 1
fi

# Запускаем Python-скрипт
echo "Запуск детектора дороги/поля..."
python3.8 segmentation/scv_cam.py

# Проверяем код завершения
if [ $? -ne 0 ]; then
    echo "Ошибка при выполнении scv_cam.py"
    exit 1
fi

echo "Детектор завершил работу"
