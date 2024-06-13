import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.title("Grapevine Image Classification")

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.keras')
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
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    st.write(predictions)
