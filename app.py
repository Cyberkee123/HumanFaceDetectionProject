import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import gdown
import os

# 1. Configuration and Model Retrieval
# Replace 'YOUR_GOOGLE_DRIVE_FILE_ID' with the actual ID from your shareable link
FILE_ID = '130UmJ7x2YWv4gsyU8JBY07CdW_XPUYj3' 
MODEL_PATH = 'full_emotion_model.keras'

@st.cache_resource
def load_emotion_model():
    """Downloads the model from Google Drive if not present and loads it."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model from Google Drive..."):
            url = f'https://drive.google.com/uc?id={FILE_ID}'
            #url = f'https://drive.google.com/uc?id=130UmJ7x2YWv4gsyU8JBY07CdW_XPUYj3'
            #url = f'https://drive.google.com/file/d/130UmJ7x2YWv4gsyU8JBY07CdW_XPUYj3/view?usp=sharing'
            gdown.download(url, MODEL_PATH, quiet=False)
    
    return load_model(MODEL_PATH)

# Define the classes based on the dataset structure
# Observed labels: Angry, Fear, Happy, Neutral, Sad, 
CLASS_LABELS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# 2. Image Preprocessing
def preprocess_image(image):
    """Prepares the uploaded image for the VGG16-based model."""
    # The model expects 48x48 RGB images based on input_layer_1 metadata
    image = image.resize((48, 48))
    image = image.convert('RGB')
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization
    return img_array

# 3. Streamlit UI
st.title("Facial Emotion Recognition")
st.write("Upload a face image to predict the emotion.")

model = load_emotion_model()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # B) The Predict Emotion Button
    if st.button("Predict Emotion"):
        processed_img = preprocess_image(image)
        prediction = model.predict(processed_img)
        
        # Get the index of the highest probability
        label_index = np.argmax(prediction)
        predicted_emotion = CLASS_LABELS[label_index]
        confidence = prediction[0][label_index] * 100

        st.success(f"Prediction: **{predicted_emotion}** ({confidence:.2f}%)")
        
        # Optional: Show probability chart
        st.bar_chart(dict(zip(CLASS_LABELS, prediction[0])))

