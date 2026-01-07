import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
import pandas as pd

# 1. Page Configuration
st.set_page_config(page_title="Human Emotion 111 Detector", layout="centered")

# 2. Load the model and labels
@st.cache_resource
def load_emotion_model():
    # Loading the model saved after training
    return tf.keras.models.load_model('model_keras.keras')

model = load_emotion_model()
# Labels matching the FER-2013 dataset structure
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 3. Load Face Detector
# Essential for cropping the face before feeding it to the CNN
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("😊 Human Emotion 1cnn Detection")
st.write("Upload a photo to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Streamlit displays RGB
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(img_rgb, caption='Uploaded Image', use_container_width=True)

    with st.spinner("Analyzing face..."):
        # Face detection is performed on grayscale for better accuracy
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5)
        
        if len(faces) > 0:
            # Sort to find the largest face
            (x, y, w, h) = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            
            # 4. PREPROCESSING (Fixes the ValueError)
            # Crop the face from the RGB image because your model expects 3 channels
            roi_color = img_rgb[y:y+h, x:x+w]
            
            # Resize to 48x48 as required by your model architecture
            roi = cv2.resize(roi_color, (48, 48), interpolation=cv2.INTER_AREA)
            
            # Normalization: rescale 1./255 to match training
            roi = roi.astype('float32') / 255.0
            
            # Reshape to (1, 48, 48, 3) - No extra channel dimension needed for RGB
            img_pixels = np.expand_dims(roi, axis=0)

            # 5. Prediction
            prediction = model.predict(img_pixels)
            max_index = np.argmax(prediction)
            predicted_emotion = emotion_labels[max_index]

            with col2:
                st.success(f"**Prediction: {predicted_emotion.upper()}**")
                
                # Confidence Chart
                prob_df = pd.DataFrame({
                    'Emotion': emotion_labels,
                    'Confidence': prediction[0]
                })
                st.bar_chart(prob_df.set_index('Emotion'))
        else:
            st.warning("No face detected. Please ensure the face is clear and centered.")

