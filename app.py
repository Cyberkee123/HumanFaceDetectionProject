import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# 1. Initialize session state variables at the top
# This prevents them from being reset every time the script reruns
if 'prediction_label' not in st.session_state:
    st.session_state.prediction_label = None
if 'confidence_score' not in st.session_state:
    st.session_state.confidence_score = None

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("full_emotion_model.keras", compile=False)

model = load_emotion_model()
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

st.title("Human Emotion Detection")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Process image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    st.image(image, channels="BGR", caption="Uploaded Image")

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48))
    img_array = resized.astype('float32') / 255.0
    img_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)

    # 2. Update session state only when the button is clicked
    if st.button("Predict Emotion"):
        predictions = model.predict(img_array)
        max_index = np.argmax(predictions[0])
        
        # SAVE TO SESSION STATE
        st.session_state.prediction_label = emotion_labels[max_index]
        st.session_state.confidence_score = predictions[0][max_index] * 100

# 3. Always display the result if it exists in session state
# This makes sure the result doesn't disappear on subsequent reruns
if st.session_state.prediction_label:
    st.success(f"Result: {st.session_state.prediction_label.upper()}")
    st.info(f"Confidence: {st.session_state.confidence_score:.2f}%")
