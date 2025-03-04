import cv2
import os
from ultralytics import YOLO

def detect_objects_in_video(video_path, model_path, save_video=True, display_video=True, color_information='GREEN', font_size='average'):
    # Определение цветов
    colors = {
        'RED': (0, 0, 255),
        'GREEN': (0, 255, 0),
        'BLUE': (255, 0, 0),
        'BLACK': (0, 0, 0),
        'WHITE': (255, 255, 255)
    }

    # Определение размеров шрифта
    font_sizes = {
        'small': 0.3,
        'average': 0.4,
        'big': 0.5,
        'extra_large': 0.6
    }

    # Проверка существования файла видео
    if not os.path.isfile(video_path):
        print(f"Ошибка: файл видео '{video_path}' не найден.")
        return

    # Проверка существования файла модели
    if not os.path.isfile(model_path):
        print(f"Ошибка: файл модели '{model_path}' не найден.")
        return

    # Загрузка предварительно обученной модели YOLO
    model = YOLO(model_path)

    # Открытие видеофайла
    cap = cv2.VideoCapture(video_path)

    # Проверка, удалось ли открыть видеофайл
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        return

    # Получение информации о видео
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Изменение размера изображения до определенного значения
    final_wide = 1000
    r = float(final_wide) / width  # Пропорциональное соотношение ширины к высоте
    dim = (final_wide, int(height * r))  # Устанавливаем новые размеры

    # Подготовка к сохранению видео
    if save_video:
        output_video_path = f'RESULT_VIDEO/result_{os.path.basename(video_path)}'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, dim)

    # Словарь для подсчета количества уникальных объектов каждого класса
    class_unique_counts = {}

    # Чтение и обработка каждого кадра видео
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Изменение размера изображения
        frame = cv2.resize(frame, dim)

        # Распознавание объектов на кадре с трекингом
        results = model.track(frame, iou=0.4, conf=0.5, persist=True, imgsz=608, verbose=False, tracker="botsort.yaml")

        # Отображение результатов на кадре
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Координаты ограничивающего прямоугольника
                cls = int(box.cls[0])  # Класс объекта
                label = model.model.names[cls]  # Имя класса
                track_id = int(box.id[0]) if box.id is not None else 'N/A'  # ID трека

                # Добавляем уникальный track_id в словарь для данного класса
                if track_id != 'N/A':
                    if label not in class_unique_counts:
                        class_unique_counts[label] = set()
                    class_unique_counts[label].add(track_id)

                # Формирование метки с ID трека
                label_with_id = f"{label} ID:{track_id}"

                # Рисуем прямоугольник и метку на кадре только если track_id не равен 'N/A'
                if track_id != 'N/A':
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label_with_id, (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

        # Отображение количества уникальных объектов каждого класса в левом верхнем углу
        y_offset = 15
        for class_name, unique_ids in class_unique_counts.items():
            count = len(unique_ids)
            cv2.putText(frame, f'{class_name}: {count}', (10, y_offset), cv2.FONT_HERSHEY_COMPLEX, font_sizes[font_size], colors[color_information], 1)
            y_offset += 15

        # Сохранение кадра в видеофайл
        if save_video:
            out.write(frame)

        # Отображение кадра с распознанными объектами
        if display_video:
            cv2.imshow('YOLO Object Detection', frame)

        # Выход по нажатию клавиши 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождение ресурсов
    cap.release()
    if save_video:
        out.release()
    cv2.destroyAllWindows()

# Пример использования функции
video_path = 'VIDEO_FOLDER/pillars.mp4'
model_path = 'MODELS_FOLDER/beststolb.pt'
detect_objects_in_video(video_path, model_path, save_video=False, display_video=True, color_information='RED', font_size='extra_large')
