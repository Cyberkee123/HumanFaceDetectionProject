import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# -------------------------------
# Load model (cached, safe)
# -------------------------------
@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model(
        "full_emotion_model.keras",
        compile=False
    )
    return model

model = load_emotion_model()

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Human pppp Emotion Detection Web App")
st.write("Upload an image for emotion prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image
    file_bytes

