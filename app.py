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
st.title("Human pppp Emotion Detection (22222 full_emotion_model.keras)Web App")
st.write("Upload an image for emotion prediction.")

uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # 1. Convert the file to an OpenCV image
    # Use .read() to get the bytes and np.frombuffer to create an array
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 2. Display the uploaded image to the user
    st.image(image, channels="BGR", caption="Uploaded Image")

    # 3. Preprocess for the model (Grayscale + Resize to 48x48)
    # This matches the training requirements in your notebook
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (48, 48))
    
    # 4. Normalize and Reshape
    # Convert to float32, scale to [0, 1], and add batch/channel dimensions
    img_array = resized_image.astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Becomes (1, 48, 48)
    img_array = np.expand_dims(img_array, axis=-1) # Becomes (1, 48, 48, 1)

    # 5. Prediction
    if st.button("Predict Emotion"):
        with st.spinner('Analyzing facial expression...'):
            predictions = model.predict(img_array)
            max_index = np.argmax(predictions[0])
            label = emotion_labels[max_index]
            confidence = predictions[0][max_index] * 100
            
            st.success(f"Prediction: {label.upper()}")
            st.info(f"Confidence Level: {confidence:.2f}%")
