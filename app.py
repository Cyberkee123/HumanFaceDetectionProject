import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Page Config
st.set_page_config(page_title="Emotion Detector", layout="centered")

# 2. Load Model
@st.cache_resource
def load_my_model():
    # Ensure 'full_emotion_model.keras' is in the same folder as this script
    return tf.keras.models.load_model('full_emotion_model.keras')

model = load_my_model()

# 3. Emotion Labels (Ensure these match your training folder order)
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("😊 Facial qqqEmotion Recognition")
st.write("Upload a photo and click the button to see the prediction.")

# 4. File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the image immediately upon upload
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 5. The Predict Button
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing face...'):
            try:
                # --- Preprocessing ---
                # Convert to RGB to ensure 3 channels (Fixes your ValueError)
                img = image.convert('RGB')
                
                # Resize to exactly 48x48
                img = img.resize((48, 48))
                
                # Convert to array and Rescale (0 to 1)
                img_array = np.array(img).astype('float32') / 255.0
                
                # Add Batch Dimension: (1, 48, 48, 3)
                img_tensor = np.expand_dims(img_array, axis=0)
                
                # --- Prediction ---
                prediction = model.predict(img_tensor)
                max_index = np.argmax(prediction[0])
                label = EMOTIONS[max_index]
                confidence = prediction[0][max_index] * 100

                # --- Results Display ---
                st.subheader(f"Result: {label}")
                st.progress(int(confidence))
                st.write(f"Confidence Score: **{confidence:.2f}%**")
                
                # Show all probabilities in a bar chart
                chart_data = dict(zip(EMOTIONS, prediction[0].tolist()))
                st.bar_chart(chart_data)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

else:
    st.info("Please upload an image file to begin.")
