# Python In-built packages
from pathlib import Path
import PIL

# Сторонние библиотеки
import streamlit as st

# Локальные модули
import settings
import helper

# Настройка макета страницы
st.set_page_config(
    page_title="Object Detection с использованием YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Наименование
st.title("Object Detection с использованием YOLOv8")

# Меню
st.sidebar.header("Конфигурация")

# Опции (будут активированы по мере добавления моделей)??
# model_type = st.sidebar.radio(
#     "Select Task", ['Detection', 'Segmentation'])

# Порог детекции 
confidence = float(st.sidebar.slider(
    "Выберите порог детекции", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
# if model_type == 'Detection':

model_path = Path(settings.DETECTION_MODEL)

# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)

# Загрузка предобученной модели
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


# выбор единицы контента: изображения/видео 
st.sidebar.header("Выбор единицы контента")
source_radio = st.sidebar.radio(
    "Выберите единицу контента", settings.SOURCES_LIST)

source_img = None

# Если выбрано изображение

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Выберите изображение...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default image",
                         width=50)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image",
                         width=50)
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(
                default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     width=50)
        else:
            if st.sidebar.button('Запустить распознавание'):
                res = model.predict(uploaded_image,
                                    conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot() #[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                st.write(helper.tags_from_yolo(res))
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.data)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")


    with col3:
        if st.sidebar.button('Запустить распознавание'):
            try:
                res = model.predict(uploaded_image,
                                    conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot() #[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                st.write(helper.tags_from_yolo(res))

            except Exception as ex:
                # st.write(ex)
                st.write("No image is uploaded yet!")

elif source_radio == settings.VIDEO:
    helper.play_stored_video(confidence, model)

# elif source_radio == settings.WEBCAM:
#     helper.play_webcam(confidence, model)

# elif source_radio == settings.RTSP:
#     helper.play_rtsp_stream(confidence, model)

# elif source_radio == settings.YOUTUBE:
#     helper.play_youtube_video(confidence, model)

else:
    st.error("Please select a valid source type!")
