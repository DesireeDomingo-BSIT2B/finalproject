import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
MODEL_URL = "https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/save_model.keras"
model = tf.keras.models.load_model(MODEL_URL)

# Function to preprocess image
def preprocess_image(image):
    img = image.convert('RGB').resize((180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Function to perform classification
def classify_image(image):
    img = preprocess_image(image)
    predictions = model.predict(img)
    class_names = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']  # Replace with your class names
    predicted_class = class_names[np.argmax(predictions)]
    return predicted_class

# Frontend code for Streamlit app
st.title("Grapevine Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Perform classification
    image = Image.open(uploaded_file)
    prediction = classify_image(image)
    st.write(f'Prediction: {prediction}')
