# BorderControl
# Запуск модулей обработки видео и панорам

## Структура скриптов

- **`run_camstitcher.sh`** - создание панорамы с камеры
- **`run_segcv.sh`** - детектор дороги/поля
- **`run_segsift.sh`** - трекер БПЛА с панорамой
- **`run_segcvsift.sh`** - параллельный запуск детектора и трекера

## Быстрый запуск

### 1. Создание панорамы
```bash
chmod +x run_camstitcher.sh
./run_camstitcher.sh
```
*Панорама сохраняется в `/stitcher/panorama_output.jpg`*

### 2. Детекция дороги/поля
```bash
chmod +x run_segcv.sh
./run_segcv.sh
```

### 3. Трекинг БПЛА с панорамой
```bash
chmod +x run_segsift.sh
./run_segsift.sh
```

### 4. Параллельный запуск (детекция + трекинг)
```bash
chmod +x run_segcvsift.sh
./run_segcvsift.sh
```

## Требования

Убедитесь, что установлены:
```bash
sudo apt install python3-opencv python3-numpy
```

## Примечания

- Скрипты автоматически проверяют доступность камеры
- Для остановки процессов нажмите `Ctrl+C`
- Логи сохраняются в соответствующих директориях
- Права на запись в папку `/stitcher` устанавливаются автоматически

## Права доступа

При первом запуске скрипты запросят права:
```bash
sudo ./run_camstitcher.sh  # для создания системной директории
```