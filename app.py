import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# --------------------------------
# Load model (cached)
# --------------------------------
@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model("model_keras.h5")
    return model

model = load_emotion_model()

emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']

st.title("Human Emotion Detection Web App")

# --------------------------------
# Image uploader
# --------------------------------
uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # --------------------------------
    # Preprocessing (CRITICAL PART)
    # --------------------------------
    img = np.array(image)

    # Resize to model input size
    img = cv2.resize(img, (48, 48))

    # Normalize
    img = img.astype("float32") / 255.0

    # Ensure shape is (1, 48, 48, 3)
    img = np.expand_dims(img, axis=0)

    # Safety check (prevents your crash)
    if img.ndim != 4:
        st.error(f"Invalid input shape: {img.shape}")
        st.stop()

    # Debug display
    st.write("Input shape to model:", img.shape)

    # --------------------------------
    # Prediction
    # --------------------------------
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    st.success(
        f"Predicted Emotion: **{emotion_labels[predicted_class]}** "
        f"(Confidence: {confidence:.2f})"
    )
