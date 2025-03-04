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

        # Распознавание объектов на кадре
        results = model(frame)

        # Отображение результатов на кадре
        annotated_frame = results[0].plot()

        # Отображение кадра с распознанными объектами
        cv2.imshow('YOLOv8 Object Detection', annotated_frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()

# Пример использования функции
video_path = 'VIDEO_FOLGER/pillars.avi'
model_path = 'beststolb.pt'
detect_objects_in_video(video_path, model_path)
