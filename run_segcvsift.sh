#!/bin/bash

# Параметры запуска
SCV_SCRIPT="segmentation/scv_cam.py"
SSIFT_SCRIPT="segmentation/ssift_cam.py"
PANORAMA_IMAGE="output_panorama.jpg"  # Укажите правильный путь к панораме
LOG_DIR="segmentation/segmentation"
TRACKING_LOG_DIR="segmentation/tracking_logs"

# Создаем директории для логов
mkdir -p "$LOG_DIR" "$TRACKING_LOG_DIR"

# Функция для запуска скрипта с обработкой ошибок
run_script() {
    local script_name=$1
    local log_file=$2
    shift 2
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Запуск $script_name с параметрами: $@" | tee -a "$log_file"
    python3 "$script_name" "$@" 2>&1 | tee -a "$log_file"
    
    exit_code=${PIPESTATUS[0]}
    if [ $exit_code -ne 0 ]; then
        echo "[ОШИБКА] $script_name завершился с кодом $exit_code" | tee -a "$log_file"
        return $exit_code
    fi
    return 0
}

# Запуск скриптов параллельно
(
    run_script "$SCV_SCRIPT" "$LOG_DIR/scv_$(date +%Y%m%d_%H%M%S).log" \
        --camera_id=0 \
        --log_dir="$LOG_DIR"
) &

(
    run_script "$SSIFT_SCRIPT" "$TRACKING_LOG_DIR/ssift_$(date +%Y%m%d_%H%M%S).log" \
        "$PANORAMA_IMAGE" \
        --frame_interval=5 \
        --log_dir="$TRACKING_LOG_DIR"
) &

# Ожидание завершения и обработка ошибок
FAIL=0
for job in $(jobs -p); do
    wait $job || let "FAIL+=1"
done

if [ "$FAIL" -ne 0 ]; then
    echo "[КРИТИЧЕСКАЯ ОШИБКА] Завершено с ошибками ($FAIL подпроцессов)"
    exit 1
fi

echo "Оба скрипта успешно завершены"
exit 0
