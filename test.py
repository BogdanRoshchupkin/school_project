import cv2
from ultralytics import YOLO

# Загрузка модели YOLOv8
model = YOLO('weights/yolo12s.pt')  # Предположим, что используется предобученная версия yolo12s; замените на нужную вариацию

# Инициализация камеры (в данном случае первый доступный видеопоток)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Выполняем детекцию на текущем кадре
    results = model(frame, verbose=False, conf=0.6)[0]

    # Обработка каждого обнаруженного объекта
    for box in results.boxes:
        # Преобразуем координаты ограничивающего прямоугольника из формата xyxy
        coords = box.xyxy.cpu().numpy()[0]  # если используется GPU, переводим в numpy
        x1, y1, x2, y2 = map(int, coords)

        # Рисуем прямоугольник на кадре (цвет: зелёный, толщина: 2 пикселя)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Получаем номер класса, уверенность и формируем метку
        class_id = int(box.cls.cpu().numpy()[0])
        confidence = float(box.conf.cpu().numpy()[0])
        label = f"{model.names[class_id]}: {confidence:.2f}"

        # Рисуем текст (метку) над прямоугольником
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Опционально: отображаем кадры в реальном времени
    cv2.imshow("Detections", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()