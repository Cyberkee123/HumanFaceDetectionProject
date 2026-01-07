import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Emotion Detector", layout="centered")

# 2. Load the model and labels
@st.cache_resource
def load_emotion_model():
    # Load the model saved from your notebook
    return tf.keras.models.load_model('full_emotion_model.keras')

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# 3. Load Face Detector
# This is necessary to crop the image so the model only sees the face
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Human Emotion 1 Detection")
st.write("Upload a photo to see the predicted emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read and display image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Detecting face and predicting emotion..."):
        # Convert to grayscale for the face detector and the model
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        # We use a slightly more sensitive scaleFactor to avoid "No face detected" errors
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Focus on the largest detected face
            (x, y, w, h) = faces[0]
            fc = gray_img[y:y+h, x:x+w]
            
            # Preprocessing to match your Notebook
            # Resize to 48x48
            roi = cv2.resize(fc, (48, 48))
            # Normalization (rescale 1./255 as used in your ImageDataGenerator)
            roi = roi.astype('float32') / 255.0
            # Reshape for model input: (1, 48, 48, 1)
            img_pixels = np.expand_dims(roi, axis=0)
            img_pixels = np.expand_dims(img_pixels, axis=-1)

            # Prediction
            prediction = model.predict(img_pixels)
            max_index = np.argmax(prediction)
            predicted_emotion = emotion_labels[max_index]

            with col2:
                st.success(f"**Prediction: {predicted_emotion.upper()}**")
                
                # Show confidence chart
                prob_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Confidence': prediction[0]
                })
                st.bar_chart(prob_df.set_index('Emotion'))
        else:
            # Warning if the Haar Cascade fails
            st.warning("No face detected. Try a photo with better lighting or a clearer front-facing view.")
        


