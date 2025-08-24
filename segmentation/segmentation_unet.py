import cv2
import numpy as np
import torch
from torchvision import models
from PIL import Image
import torchvision.transforms as T

# Инициализация модели DeepLabV3+ (предобученная на Cityscapes)
model = models.segmentation.deeplabv3_resnet50(pretrained=True).eval().to('cuda')

# Трансформации для входного изображения
transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Классы Cityscapes (дорога = 0, поле/трава = 1, и т.д.)
classes = {
    0: "road",
    1: "field/grass",  # В Cityscapes нет "поля", но есть "трава" (класс 1)
    2: "building",
    # ... (остальные классы не используются)
}

# Функция для постобработки маски
def apply_mask(image, mask, color, alpha=0.5):
    """Накладывает маску на изображение с прозрачностью."""
    for c in range(3):
        image[:, :, c] = np.where(
            mask == 1,
            image[:, :, c] * (1 - alpha) + alpha * color[c],
            image[:, :, c]
        )
    return image

# Захват видео (0 - встроенная камера, или путь к файлу)
cap = cv2.VideoCapture(0)  # Если видео с БПЛА, укажите путь

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Конвертация BGR -> RGB и resize до 640x480 для скорости
    frame = cv2.resize(frame, (640, 480))
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Преобразование и передача на GPU
    input_tensor = transform(img).unsqueeze(0).to('cuda')

    # Inference
    with torch.no_grad():
        output = model(input_tensor)['out'][0]
    output_predictions = output.argmax(0).cpu().numpy()

    # Создаем маски
    road_mask = (output_predictions == 0).astype(np.uint8)
    field_mask = (output_predictions == 1).astype(np.uint8)

    # Накладываем маски на кадр
    frame = apply_mask(frame, road_mask, (0, 255, 0), alpha=0.3)  # Дорога - зеленый
    frame = apply_mask(frame, field_mask, (255, 0, 0), alpha=0.3)  # Поле - синий

    # Вывод текста
    cv2.putText(frame, f"Road: {np.sum(road_mask)} px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Field: {np.sum(field_mask)} px", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Показываем результат
    cv2.imshow("Drone View - Road/Field Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
