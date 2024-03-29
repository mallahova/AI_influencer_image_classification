import streamlit as st
from PIL import Image
import numpy as np
import streamlit as st
import tensorflow as tf
import numpy as np
import tensorflow as tf
import cv2
import json
from keras import layers
import pywt
import joblib

model_cnn = tf.keras.models.load_model("models/4/")
model_lr = joblib.load("model.pkl")
influencer_images = [
    "source/Andrew_Ng.jpg",
    "source/Sam_Altman.jpg",
    "source/Fei-Fei_li.jpg",
    "source/Geoffrey_Hinton.jpg",
    "source/Timnit_Gebru.jpg",
    "source/Yoshua_Bengio.jpg",
]

celebrity_dict_cnn = {
    0: "Andrew Ng",
    1: "Fei-Fei Li",
    2: "Geoffrey Hinton",
    3: "Sam Altman",
    4: "Timnit Gebru ",
    5: "Yoshua Bengio",
}
celebrity_dict_lr = {
    0: "Fei-Fei Li",
    1: "Geoffrey Hinton",
    2: "Yoshua Bengio",
    3: "Sam Altman",
    4: "Timnit Gebru ",
    5: "Andrew Ng",
}
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")


def w2d(img, mode="haar", level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    imArray = np.float32(imArray)
    imArray /= 255
    coeffs = pywt.wavedec2(imArray, mode, level=level)
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0
    imArray_H = pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H = np.uint8(imArray_H)

    return imArray_H


def predict_lr(img_input):
    img_input = np.array(img_input)
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 1:
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img_input[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                img = roi_color
                img = cv2.resize(img, (32, 32))
                img_w2d = w2d(img.reshape(32, 32, 3).astype(np.uint8))
                combined_img = np.vstack(
                    (img.reshape(32 * 32 * 3, 1), img_w2d.reshape(32 * 32, 1))
                )
                predicion = model_lr.predict(combined_img.reshape(1, -1))[0]
                return celebrity_dict_lr[predicion]
            elif len(eyes) == 1:
                return "Only one eye detected"
            else:
                return "No eyes detected"
    elif len(faces) > 1:
        return "More than one face detected"
    else:
        return "Face not detected"


resize_and_rescale = tf.keras.Sequential(
    [layers.experimental.preprocessing.Resizing(256, 256)]
)


def predict_cnn(img_input):
    img_input = np.array(img_input)
    gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) == 1:
        for x, y, w, h in faces:
            roi_gray = gray[y : y + h, x : x + w]
            roi_color = img_input[y : y + h, x : x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                img = roi_color
                img = resize_and_rescale(img)
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model_cnn.predict(img_array)
                predicted_class = celebrity_dict_cnn[(np.argmax(predictions[0]))]
                confidence = round(100 * (np.max(predictions[0])), 2)
                return predicted_class, confidence
            elif len(eyes) == 1:
                return "Only one eye detected", None
            else:
                return "No eyes detected", None
    elif len(faces) > 1:
        return "More than one face detected", None
    else:
        return "Face not detected", None


st.title("AI Influencer Image Classification")
st.write(
    "[Academic Website](https://mallahova.github.io/) | [GitHub](https://github.com/mallahova/AI_influencer_image_classification)"
)


def display_influencer(image_path, name):
    st.image(image_path, width=150, use_column_width="auto", clamp=True)
    st.write(name)


# Create a layout for six columns
col1, col2, col3, col4, col5, col6 = st.columns(6)

# Display images and names in each column
with col1:
    st.image(influencer_images[0], width=150, use_column_width="auto", clamp=True)
    st.write("Andrew Ng")
with col2:
    st.image(influencer_images[1], width=150, use_column_width="auto", clamp=True)
    st.write("Sam Altman")
with col3:
    st.image(influencer_images[2], width=150, use_column_width="auto", clamp=True)
    st.write("Fe-Fei Li")
with col4:
    st.image(influencer_images[3], width=150, use_column_width="auto", clamp=True)
    st.write("Geoffrey Hinton")
with col5:
    st.image(influencer_images[4], width=150, use_column_width="auto", clamp=True)
    st.write("Timnit Gebru")
with col6:
    st.image(influencer_images[5], width=150, use_column_width="auto", clamp=True)
    st.write("Yoshua Bengio")

model_type = st.selectbox(
    "Select model type", ("Convolutional Neural Network", "Logistic Regression")
)
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    if model_type == "Logistic Regression":
        st.write(
            "<span style='font-size:24px'>{}.</span>".format(predict_lr(image)),
            unsafe_allow_html=True,
        )
    else:
        st.write(
            "<span style='font-size:24px'>{}.</span>".format(predict_cnn(image)[0]),
            unsafe_allow_html=True,
        )
