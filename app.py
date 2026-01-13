import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# 1. Page Configuration
st.set_page_config(page_title="Facial Emotion Recognition", layout="centered")

# 2. Load the trained model
@st.cache_resource
def load_model():
    try:
        # Loading the specific model file you provided
        model = tf.keras.models.load_model('full_emotion_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# 3. Define the Emotion Classes
# Ensure these match the subfolders in your training data
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 4. Image Preprocessing Function
def preprocess_image(image):
    # Convert PIL image to RGB to ensure 3 channels (Fixes the ValueError)
    img = np.array(image.convert('RGB'))
    
    # Resize to 48x48 as required by your model's input layer
    img = cv2.resize(img, (48, 48)) 
    
    # Normalize pixels to [0, 1]
    img = img.astype('float32') / 255.0
    
    # Add batch dimension: shape becomes (1, 48, 48, 3)
    img = np.expand_dims(img, axis=0)
    return img

# 5. UI Elements
st.title("Facial Emotion Recognition")
st.write("Upload a face image to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    if model is not None:
        with st.spinner('Analyzing...'):
            try:
                # Preprocess and Predict
                processed_img = preprocess_image(image)
                prediction = model.predict(processed_img)
                
                # Get the emotion with highest probability
                max_index = np.argmax(prediction[0])
                emotion = EMOTIONS[max_index]
                confidence = prediction[0][max_index] * 100

                # Show Result
                st.success(f"Prediction: **{emotion}** ({confidence:.2f}%)")
                
                # Show probability chart
                st.write("### Confidence levels:")
                chart_data = dict(zip(EMOTIONS, prediction[0].tolist()))
                st.bar_chart(chart_data)
            except Exception as e:
                st.error(f"Prediction Error: {e}")
    else:
        st.error("Model 'full_emotion_model.keras' not found in the current directory.")







