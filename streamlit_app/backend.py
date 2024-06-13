import tensorflow as tf
from PIL import Image
import numpy as np
from flask import Flask, request, jsonify
import requests
import io

app = Flask(__name__)

# Function to load model from URL
def load_model_from_url(url):
    response = requests.get(url)
    model = tf.keras.models.load_model(io.BytesIO(response.content))
    return model

# Load model from URL
MODEL_URL = "https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/save_model.keras"
model = load_model_from_url(MODEL_URL)

# Function to preprocess image
def preprocess_image(image):
    img = image.convert('RGB').resize((180, 180))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Extract class names from model
def get_class_names(model):
    class_names = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']  # Replace with your class names
    return class_names

# Route for image classification
@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image = request.files['image']
    img = preprocess_image(Image.open(image))
    predictions = model.predict(img)
    class_names = get_class_names(model)
    predicted_class = class_names[np.argmax(predictions)]
    
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
