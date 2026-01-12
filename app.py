import streamlit as st
import cv2
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Facial Emotion Recognition")

st.title("😊 Facial Emotion Recognition")

# -----------------------------
# Google Drive config
# -----------------------------
FILE_ID = "1gd8pl9pPY0XZe4IJV8ho-WNHcnrsWFOf"
MODEL_PATH = "full_emotion_model.keras"
GDRIVE_URL = f"https://drive.google.com/drive/folders/1gd8pl9pPY0XZe4IJV8ho-WNHcnrsWFOf"

# -----------------------------
# Download model if not exists
# -----------------------------
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading emotion model..."):
            gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False)
    return load_model(MODEL_PATH)

model = load_emotion_model()

# -----------------------------
# Emotion labels
# -----------------------------
emotion_labels = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise"
]

# -----------------------------
# Face detector
# -----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# -----------------------------
# Preprocessing (48,48,3)
# -----------------------------
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# -----------------------------
# Input
# -----------------------------
img_file = st.camera_input("Take a photo")

if img_file is not None:
    image = cv2.imdecode(
        np.frombuffer(img_file.read(), np.uint8),
        cv2.IMREAD_COLOR
    )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.warning("No face detected")
    else:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]
            face_input = preprocess_face(face)

            preds = model.predict(face_input, verbose=0)
            idx = np.argmax(preds)
            emotion = emotion_labels[idx]
            confidence = preds[0][idx]

            cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(
                image,
                f"{emotion} ({confidence:.2f})",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0,255,0),
                2
            )

        st.image(image, channels="BGR")
