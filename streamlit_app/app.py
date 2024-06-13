import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import requests
from io import BytesIO

# Function to load the saved model
@st.cache(allow_output_mutation=True)
def load_saved_model():
    model_url = "https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/saved_model/saved_model.pb"
    model = tf.saved_model.load(model_url)
    return model

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Function to make predictions
def make_prediction(image, model):
    image = preprocess_image(image)
    prediction = model(image)['predictions'][0].numpy()
    return prediction

# Streamlit app
def main():
    st.title('Image Classifier')

    # Load the saved model
    model = load_saved_model()

    # Upload image
    uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Make prediction
        prediction = make_prediction(image, model)

        st.write("Prediction:", prediction)

if __name__ == "__main__":
    main()
