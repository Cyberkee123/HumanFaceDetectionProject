import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

# 1. Load the model and labels
@st.cache_resource
def load_emotion_model():
    # Matches the saved model name from your notebook
    return tf.keras.models.load_model('full_emotion_model.keras')

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 2. Load Face Detector (Essential for correct cropping)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Human Emotion Detection")
uploaded_file = st.file_uploader("Upload a face image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Convert upload to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    st.image(img_rgb, caption='Uploaded Image', width=300)

    # 3. Preprocessing
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            # Crop to the face only
            roi_gray = gray[y:y+h, x:x+w]
            # Resize to 48x48 (Matches Notebook)
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Normalization: rescale 1./255 (Matches Notebook)
            roi = roi_gray.astype('float32') / 255.0
            
            # Reshape for model: (batch, height, width, channels)
            img_pixels = np.expand_dims(roi, axis=0)
            img_pixels = np.expand_dims(img_pixels, axis=-1)

            # 4. Predict
            predictions = model.predict(img_pixels)
            max_index = np.argmax(predictions[0])
            emotion = emotion_labels[max_index]

            st.subheader(f"Predicted Emotion: {emotion}")
            
            # Show confidence levels
            prob_df = pd.DataFrame({'Emotion': emotion_labels, 'Confidence': predictions[0]})
            st.bar_chart(prob_df.set_index('Emotion'))
    else:
        st.warning("No face detected. Please ensure the face is clear and centered.")
