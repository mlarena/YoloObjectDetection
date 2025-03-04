import cv2
import os
from ultralytics import YOLO

def detect_objects_in_video(video_path, model_path):
    # Проверка существования файла видео
    if not os.path.isfile(video_path):
        print(f"Ошибка: файл видео '{video_path}' не найден.")
        return

    # Проверка существования файла модели
    if not os.path.isfile(model_path):
        print(f"Ошибка: файл модели '{model_path}' не найден.")
        return

    # Загрузка предварительно обученной модели YOLOv8
    model = YOLO(model_path)

    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)

    # Проверка, удалось ли открыть видеофайл
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        return

    # Чтение и обработка каждого кадра видео
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Изменение размера изображения до определенного значения
        final_wide = 1000
        r = float(final_wide) / frame.shape[1]  # Пропорциональное соотношение ширины к высоте
        dim = (final_wide, int(frame.shape[0] * r))  # Устанавливаем новые размеры
        frame = cv2.resize(frame, dim)  # Уменьшаем изображение до подготовленных размеров

        # Распознавание объектов на кадре
        results = model(frame)

        # Отображение результатов на кадре
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты ограничивающего прямоугольника
                cls = int(box.cls[0])  # Класс объекта
                label = model.model.names[cls]  # Имя класса

                # Рисуем прямоугольник и метку на кадре
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        # Отображение кадра с распознанными объектами
        cv2.imshow('YOLOv8 Object Detection', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Пример использования функции
video_path = 'VIDEO_FOLDER/pillars.mp4'
model_path = 'MODELS_FOLDER/beststolb.pt'
detect_objects_in_video(video_path, model_path)
