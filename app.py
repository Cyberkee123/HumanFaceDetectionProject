import streamlit as st
import tensorflow as tf
import cv2
import numpy as np

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

@st.cache_resource
def load_emotion_model():
    return tf.keras.models.load_model("improved_emotion_model.keras", compile=False)

model = load_emotion_model()
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

st.title("Facial Emotion Detection (Improved)")
uploaded_file = st.file_uploader("Upload a face photo...", type=["jpg", "png"])

if uploaded_file:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 1. Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        st.warning("No face detected. Please try a clearer photo.")
    else:
        for (x, y, w, h) in faces:
            # 2. Crop only the face area
            face_crop = image[y:y+h, x:x+w]
            
            # 3. Preprocess the cropped face
            input_face = cv2.resize(face_crop, (48, 48))
            input_face = input_face.astype('float32') / 255.0
            input_face = np.expand_dims(input_face, axis=0)
            
            # 4. Predict
            preds = model.predict(input_face)
            label = emotion_labels[np.argmax(preds)]
            confidence = np.max(preds) * 100
            
            # Draw on original image for display
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        st.image(image, channels="BGR", caption="Detection Result")
        st.success(f"Detected Emotion: {label} ({confidence:.1f}%)")
