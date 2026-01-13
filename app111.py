import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# # 1. Load the trained model
# @st.cache_resource
# def load_emotion_model():
#     # Ensure this file name matches the one exported from your notebook
#     return tf.keras.models.load_model('full_emotion_model.keras')

# model = load_emotion_model()


# -------------------------------
# Load model (cached, safe)
# -------------------------------
@st.cache_resource
def load_emotion_model():
    # Using a raw string (r"") to handle Windows backslashes correctly
    model_path = r"C:\Users\cyber\HumanFaceDetectionProject\full_emotion_model.keras"
    
    model = tf.keras.models.load_model(
        model_path,
        compile=False
    )
    return model

#@st.cache_resource
#def load_emotion_model():
#    model_path = "full_emotion_model.keras"
#    if not os.path.exists(model_path):
#       st.error("Model file not found!")
#       st.stop()
#    return load_model(model_path)

#model = load_emotion_model()

st.success("Emotion model loaded successfully!")



# 2. Define labels based on the notebook's dataset structure
# Note: Ensure the order matches your LabelBinarizer classes from training
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("Facial Emotion Recognition App")
st.write("Upload a clear photo of a face to predict the emotion.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the file into a format OpenCV understands
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Display the uploaded image
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)
    st.write("### Classifying...")

    # --- MATCHING PREPROCESSING FROM NOTEBOOK ---
    # Step A: Convert to Grayscale (Model was trained on 1-channel images)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Step B: Resize to 48x48 (The input size used in your FER-2013 dataset)
    resized_img = cv2.resize(gray_img, (48, 48))
    
    # Step C: Normalize (Scaling pixel values to [0, 1] range)
    normalized_img = resized_img / 255.0
    
    # Step D: Reshape for Model Input (Samples, Height, Width, Channels)
    # The final shape must be (1, 48, 48, 1)
    input_data = np.reshape(normalized_img, (1, 48, 48, 1))
    # --------------------------------------------

    # Perform Prediction
    predictions = model.predict(input_data)
    max_index = np.argmax(predictions[0])
    predicted_emotion = emotion_labels[max_index]
    confidence = np.max(predictions[0]) * 100

    # Display Results
    st.success(f"**Prediction:** {predicted_emotion}")
    st.info(f"**Confidence:** {confidence:.2f}%")
    
    # Optional: Display confidence bars for all emotions
    st.write("#### All Emotion Probabilities:")
    for label, prob in zip(emotion_labels, predictions[0]):
        st.write(f"{label}: {prob*100:.1f}%")
        st.progress(float(prob))
