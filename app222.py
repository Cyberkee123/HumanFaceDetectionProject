import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Facial Emotion 123Recognition",
    layout="centered"
)

st.title("😊 Facial Emotion Recognition")

# -----------------------------
# Load model (cached)
# -----------------------------
@st.cache_resource
def load_emotion_model():
    return load_model("full_emotion_model.keras")

model = load_emotion_model()

# -----------------------------
# Emotion labels
# ⚠️ MUST match training folder order
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
# Preprocess face for model
# -----------------------------
def preprocess_face(face_img):
    face_img = cv2.resize(face_img, (48, 48))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img / 255.0
    face_img = np.expand_dims(face_img, axis=0)
    return face_img

# -----------------------------
# Input source
# -----------------------------
option = st.radio(
    "Choose input method:",
    ("Upload Image", "Use Camera")
)

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

elif option == "Use Camera":
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        image_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

# -----------------------------
# Process image
# -----------------------------
if "image" in locals():
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    if len(faces) == 0:
        st.warning("No face detected 😕")
    else:
        for (x, y, w, h) in faces:
            face = image[y:y+h, x:x+w]

            face_input = preprocess_face(face)
            predictions = model.predict(face_input, verbose=0)

            emotion_index = np.argmax(predictions)
            emotion = emotion_labels[emotion_index]
            confidence = predictions[0][emotion_index]

            # Draw bounding box & label
            cv2.rectangle(
                image,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                image,
                f"{emotion} ({confidence:.2f})",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        st.image(image, channels="BGR")

        # Show probability scores
        st.subheader("Emotion Probabilities")
        for i, label in enumerate(emotion_labels):
            st.write(f"{label}: {predictions[0][i]:.3f}")
