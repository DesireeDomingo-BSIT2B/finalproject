import streamlit as st
import requests

# Frontend code for Streamlit app
st.title("Grapevine Image Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Send image to backend for classification
    response = requests.post("http://localhost:8000/classify", files={"image": uploaded_file})
    
    if response.status_code == 200:
        prediction = response.json()['prediction']
        st.write(f'Prediction: {prediction}')
    else:
        st.error("Error occurred during classification.")
