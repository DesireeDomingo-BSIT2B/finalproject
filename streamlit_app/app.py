import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import os

st.title("Grapevine Image Classification")

MODEL_URL = "https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/model.h5"

def download_model(url, filename):
    if not os.path.exists(filename):
        with st.spinner("Downloading model..."):
            response = requests.get(url)
            with open(filename, 'wb') as file:
                file.write(response.content)
            st.success("Model downloaded successfully!")

@st.cache_resource
def load_model():
    download_model(MODEL_URL, 'model.h5')
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert('RGB')  # Ensure image is in RGB format
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # Preprocess the image
        img_array = np.array(image)
        img_array = tf.image.resize(img_array, [180, 180])  # Resize to match model input size
        img_array = img_array / 255.0  # Normalize to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Make predictions
        predictions = model.predict(img_array)
        
        # Assuming you have a list of class names
        class_names = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']
        predicted_class = class_names[np.argmax(predictions)]
        st.write(f'Prediction: {predicted_class}')
        
    except Exception as e:
        st.error(f"Error in classifying the image: {e}")
        st.write(e)  # Log the detailed exception for debugging
