# Импорт библиотек
from ultralytics import YOLO
import time
import streamlit as st
import cv2
from pytube import YouTube

# Импорт локальных модулей
import settings


# Функция загрузки предобученной модели
def load_model(model_path):
    '''
    Функция загружает модель обнаружения объектов YOLO из указанного model_path.
    Входные данные: model_path (str): путь к файлу модели YOLO.
    Выходные данные: Модель обнаружения объектов YOLO.
    '''
    
    model = YOLO(model_path)
    return model


# отображение трекера
def display_tracker_options():
    display_tracker = st.radio("Display Tracker", ('Yes', 'No'))
    is_display_tracker = True if display_tracker == 'Yes' else False
    if is_display_tracker:
        tracker_type = st.radio("Tracker", ("bytetrack.yaml", "botsort.yaml"))
        return is_display_tracker, tracker_type
    return is_display_tracker, None


# Функция отображения рамки
def _display_detected_frames(conf, model, st_frame, image, is_display_tracking=None, tracker=None):
    '''
    Отображене обнаруженных объектов на изображении/видеокадре, используя модель YOLOv8.

     Аргументы:
     - conf (float): порог уверенности для обнаружения объекта.
     - модель (YoloV8): модель обнаружения объектов YOLov8.
     - st_frame (объект Streamlit): объект Streamlit для отображения обнаруженного видео.
     - image (массив numpy): массив numpy, представляющий видеокадр.
     - is_display_tracking (bool): флаг, указывающий, отображать ли отслеживание объекта (по умолчанию = нет).

     Возврат: None
    '''

    # Изменение размера изображения до стандартного размера
    image = cv2.resize(image, (300, int(300*(9/16))))

    # Отображение отслеживания объекта, если указано
    if is_display_tracking:
        res = model.track(image, conf=conf, persist=True, tracker=tracker)
    else:
        # Прогнозирование объектов на изображении с помощью модели YOLOv8
        res = model.predict(image, conf=conf)

    # Нанесение обнаруженных объектов на видеокадр
    res_plotted = res[0].plot()
    st_frame.image(res_plotted,
                   caption='Detected Video',
                   channels="BGR",
                   use_column_width=True
                   )


def play_stored_video(conf, model):
    '''
    Воспроизведение сохраненного видеофайла. Отслеживание и обнаружение объектов в режиме реального времени с использованием модели обнаружения объектов YOLOv8.

    Параметры:
    - conf: Достоверность модели YOLOv8.
    - model: экземпляр класса YOLOv8, содержащий модель YOLOv8.

     Возврат: None
    '''
    
    source_vid = st.sidebar.selectbox(
        "Выберите видео...", settings.VIDEOS_DICT.keys())

    is_display_tracker, tracker = display_tracker_options()

    with open(settings.VIDEOS_DICT.get(source_vid), 'rb') as video_file:
        video_bytes = video_file.read()
    
    if video_bytes:
        st.video(video_bytes)

    if st.sidebar.button('Detect Video Objects'):
        try:
            vid_cap = cv2.VideoCapture(
                str(settings.VIDEOS_DICT.get(source_vid)))
            st_frame = st.empty()
            while (vid_cap.isOpened()):
                success, image = vid_cap.read()
                if success:
                    _display_detected_frames(conf,
                                             model,
                                             st_frame,
                                             image,
                                             is_display_tracker,
                                             tracker
                                             )
                else:
                    vid_cap.release()
                    break
        except Exception as e:
            st.sidebar.error("Error loading video: " + str(e))


def tags_from_yolo(results):
    for result in results:
        boxes = result.boxes  # Boxes object for bbox outputs
        classes = result.names
    
    detected_classes_num = boxes.cls.unique().tolist()
    classes_det = []
    for i in detected_classes_num:
        classes_det.append(classes[i])
    return classes_det
