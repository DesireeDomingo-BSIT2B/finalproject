import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import requests
import io

st.title("Grapevine Image Classification")

@st.cache_resource
def load_model():
    model_url = 'https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/grapevinemodel.keras'
    response = requests.get(model_url)
    model_data = io.BytesIO(response.content)
    model = tf.keras.models.load_model(model_data)
    return model

model = load_model()

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    img_array = np.array(image)
    img_array = tf.image.resize(img_array, [224, 224])
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    try:
        predictions = model.predict(img_array)
        st.write(predictions)
    except Exception as e:
        st.write(f"Error in classifying the image: {e}")
