import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
import numpy as np
from PIL import Image
import gdown
import os

# 1. Constants
IMG_SIZE = (48, 48)
FILE_ID = '1K4cQ0qcvylA1aKuM5iD4iHuKPpjulauT'
MODEL_PATH = 'full_emotion_model.keras'

@st.cache_resource
def load_emotion_model():
    # Download from Drive if the file doesn't exist locally
    if not os.path.exists(MODEL_PATH):
        url = f'https://drive.google.com/uc?id={FILE_ID}'
        gdown.download(url, MODEL_PATH, quiet=False)

    # 1. Define the base VGG16 (No top layers, custom input shape)
    vgg_base = VGG16(weights=None, include_top=False, input_shape=(48, 48, 3))
    
    # 2. Build the architecture using Functional API
    # Logic: Extract the tensor from the list if necessary to avoid the AttributeError
    base_output = vgg_base.output
    if isinstance(base_output, list):
        base_output = base_output[0]
        
    x = Flatten()(base_output)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(6, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=vgg_base.input, outputs=predictions)
    
    # 3. Load the weights into the manual architecture
    try:
        model.load_weights(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading weights: {e}")
        return None

# Initialize Model
model = load_emotion_model()

# 2. Define Emotion Labels
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Suprise']

st.title("Facial Emotion Recognition")
st.write("Upload a photo to detect the emotion.")

# 3. Image Upload Interface
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB') # Ensure 3 channels for VGG16
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    # 4. Pre-processing
    img = image.resize(IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
    img_array = img_array / 255.0                # Normalize

    # 5. Prediction
    if st.button('Predict Emotion'):
        if model is not None:
            prediction = model.predict(img_array)
            max_index = np.argmax(prediction[0])
            predicted_emotion = EMOTIONS[max_index]
            
            st.success(f"Predicted Emotion: **{predicted_emotion}**")
            
            # Display confidence levels
            st.write("### Confidence Scores:")
            cols = st.columns(len(EMOTIONS))
            for i, emotion in enumerate(EMOTIONS):
                cols[i].metric(label=emotion, value=f"{prediction[0][i]:.2f}")
        else:
            st.error("Model not loaded. Please check the weight file path.")



