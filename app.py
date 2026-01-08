import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# -------------------------------
# Load model (cached, safe)
# -------------------------------
@st.cache_resource
def load_emotion_model():
    model = tf.keras.models.load_model(
        "full_emotion_model.keras",
        compile=False
    )
    return model

model = load_emotion_model()

emotion_labels = [
    'angry', 'disgust', 'fear',
    'happy', 'neutral', 'sad', 'surprise'
]

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("Human pppp Emotion Detection (1111 full_emotion_model.keras)Web App")
st.write("Upload an image for emotion prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

#if uploaded_file is not None:
#    # Read image
#    file_bytes

if uploaded_file is not None:
    # --- FIX THE NAMEERROR ---
    # Read the file into a byte array
    content = uploaded_file.read()
    file_bytes = np.asarray(bytearray(content), dtype=np.uint8)
    
    # Decode the bytes into an OpenCV image
    image = cv2.imdecode(file_bytes, 1)

    # Display the original image for the user
    st.image(image, channels="BGR", caption="Uploaded Image")

    # --- PREPROCESSING (Per your Notebook/Proposal) ---
    # 1. Convert to Grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 2. Resize to 48x48 (Matches model input)
    resized_image = cv2.resize(gray_image, (48, 48))
    
    # 3. Reshape and Normalize
    # Adding batch dimension and channel dimension: (1, 48, 48, 1)
    img_array = resized_image.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # 4. Prediction Trigger
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing...'):
            predictions = model.predict(img_array)
            # Find index with highest probability
            max_index = np.argmax(predictions[0])
            label = emotion_labels[max_index]
            confidence = predictions[0][max_index] * 100
            
            st.success(f"Result: {label.upper()} ({confidence:.2f}% confidence)")

