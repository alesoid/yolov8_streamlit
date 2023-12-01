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
st.title("Модель присвоения тегов на основе Object Detection и ZeroShot Classification")

# Меню
st.sidebar.header("Конфигурация")

# Порог детекции для Object Detection
confidence = float(st.sidebar.slider( "Выберите порог детекции", 25, 100, 40)) / 100

# Selecting Detection Or Segmentation
# if model_type == 'Detection':

model_path = Path(settings.DETECTION_MODEL)

# elif model_type == 'Segmentation':
#     model_path = Path(settings.SEGMENTATION_MODEL)

# Загрузка предобученной модели YOLO
try:
    model = helper.load_model(model_path)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

# Загрузка предобученной модели и процессора CLIP
try:
    model_CLIP, processor_CLIP = helper.load_clip(settings.MODEL_CLIP)
except Exception as ex:
    st.error(f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)


# выбор единицы контента: изображения/видео 
st.sidebar.header("Выбор единицы контента")
source_radio = st.sidebar.radio(
    "Выберите единицу контента", settings.SOURCES_LIST)

source_img = None

# Если выбрано изображение (IMAGE)

if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader("Выберите изображение...", 
                                          type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    col1, col2, col3, col4 = st.columns(4)
    col3.write("Теги object detection")
    col4.write("Теги classification")
    
    with col1:
        try:
            if source_img is None:
                default_image_path = str(settings.DEFAULT_IMAGE)
                default_image = PIL.Image.open(default_image_path)
                st.image(default_image_path, caption="Default image", use_column_width=True)
            else:
                uploaded_image = PIL.Image.open(source_img)
                st.image(source_img, caption="Uploaded Image", use_column_width=True)
        
        except Exception as ex:
            st.error("Error occurred while opening the image.")
            st.error(ex)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            default_detected_image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',use_column_width=True)
        else:
            if st.sidebar.button('Tags from yolo'):
                res = model.predict(uploaded_image,
                                    conf=confidence)
                boxes = res[0].boxes
                res_plotted = res[0].plot()[:, :, ::-1]
                st.image(res_plotted, caption='Detected Image',
                         use_column_width=True)
                with col3:
                    tags_list = helper.tags_from_yolo(res)
                    for i, tags in enumerate(tags_list):
                        st.write(i, tags)

    with col4:
        if st.sidebar.button('tags from CLIP'):
            classif_list = []
            classes = settings.CLASSES
            for i , key in enumerate(classes):
                classification = classes[key]
                inputs = processor_CLIP(text=classification, images=uploaded_image, return_tensors="pt", padding=True)
                outputs = model_CLIP(**inputs)
                logits_per_image = outputs.logits_per_image # this is the image-text similarity score
                probs = logits_per_image.softmax(dim=1)     # we can take the softmax to get the label probabilities
                res = probs.tolist()
                res_classif = res[0]
                # print(i, key, res_classif)

                # plt.barh(classification, res_classif)
                # plt.show()

                class_lst = []
                #   for i, ver in enumerate(res_classif):
                #     if ver >= 0.5:
                #       class_lst.append(classification[i])
                #   if class_lst == []:
                max_v = np.argmax(res_classif)
                print(max_v)
                class_lst.append(classification[max_v])
                print(class_lst)
                classif_list.append(class_lst[0])

            #   # print(class_lst)
            for i, cl_tags in enumerate(classif_list):
                st.write(i, cl_tags)


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
