import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image
import gdown
import os

#https://drive.google.com/file/d/1y4OBg-FbUSDff7K9BrsmtUh67udMR9RB/view?usp=sharing
#https://drive.google.com/file/d/1K4cQ0qcvylA1aKuM5iD4iHuKPpjulauT/view?usp=sharing

# 1. Constants
IMG_SIZE = (48, 48)
FILE_ID = '1K4cQ0qcvylA1aKuM5iD4iHuKPpjulauT'
MODEL_PATH = 'full_emotion_model.keras'

# 1. Define the base VGG16
    vgg_base = VGG16(weights=None, include_top=False, input_shape=(48, 48, 3))
    
    # 2. Build the architecture using Functional API
    # This explicitly handles the tensor output to avoid the 'list' error
    x = vgg_base.output
    if isinstance(x, list):
        x = x[0]  # Grab the tensor if it's wrapped in a list
        
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=vgg_base.input, outputs=predictions)
    
    # 3. Load the weights
    try:
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load weights: {e}")
        return None

model = load_emotion_model()


# 2. Define Emotion Labels (Based on your dataset structure)
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

st.title("Facial Emotion 7788Recognition")
st.write("Upload a photo to detect the emotion.")

# 3. Image Upload Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # 4. Pre-processing (Matching the model's training config)
    # The notebook indicates an input shape of 48x48x3
    img = image.resize((48, 48))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0                # Normalize if used during training

    # 5. Prediction
    if st.button('Predict Emotion'):
        prediction = model.predict(img_array)
        max_index = np.argmax(prediction[0])
        predicted_emotion = EMOTIONS[max_index]
        
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
        
        # Display confidence levels
        for i, emotion in enumerate(EMOTIONS):
            st.write(f"{emotion}: {prediction[0][i]:.2f}")


