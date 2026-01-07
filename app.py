import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

# Page configuration
st.set_page_config(page_title="Emotion Detector", layout="centered")

# Use caching to load the model once
@st.cache_resource
def load_emotion_model():
    # Ensure this matches the filename in your notebook
    model = tf.keras.models.load_model('full_emotion_model.keras')
    return model

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Human Emotion Detection")
st.write("Upload a clear photo of a face to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # For display
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Analyzing face and emotion..."):
        # 1. Grayscale conversion
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 2. Face Detection
        faces = face_cascade.detectMultiScale(gray_img, 1.1, 4)
        
        if len(faces) > 0:
            # Take the first face detected
            (x, y, w, h) = faces[0]
            fc = gray_img[y:y+h, x:x+w]
            
            # 3. Resize and Normalize (Matching training notebook)
            roi = cv2.resize(fc, (48, 48))
            img_pixels = np.expand_dims(np.expand_dims(roi, -1), 0)
            img_pixels = img_pixels.astype('float32') / 255.0 # Normalization

            # 4. Prediction
            prediction = model.predict(img_pixels)
            max_index = np.argmax(prediction)
            predicted_emotion = emotion_labels[max_index]

            with col2:
                st.success(f"**Result: {predicted_emotion.upper()}**")
                
                # Create a dataframe for the probability chart
                prob_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Probability': prediction[0]
                })
                st.bar_chart(prob_df.set_index('Emotion'))
        else:
            st.error("No face detected. Please try an image where the face is clearly visible.")

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This model was trained on the FER-2013 dataset to recognize 7 basic human emotions.")
