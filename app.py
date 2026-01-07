import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Use caching to load the model once

@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model('model_keras.h5')
    return model

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Human Emotion Detection Web 1 App")
st.write("Upload an image for emotion prediction.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    st.image(img, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for the CNN model (resize, grayscale, normalize, etc.)
    # ... (your specific preprocessing steps from the notebook) ...
    # Example:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(gray_img, (48, 48)) # Adjust size based on your model input
    input_arr = np.expand_dims(np.expand_dims(resized_img, -1), 0) / 255.0

    # Make prediction
    prediction = model.predict(input_arr)
    predicted_emotion = emotion_labels[np.argmax(prediction)]

    st.success(f'Prediction: {predicted_emotion}')
