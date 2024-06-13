import tensorflow as tf
from PIL import Image
import numpy as np
import requests
from flask import Flask, request, jsonify

app = Flask(__name__)

# Function to load model from URL
def load_model_from_url(url):
    response = requests.get(url)
    with open('model.keras', 'wb') as f:
        f.write(response.content)
    return tf.keras.models.load_model('model.keras')

# Load model from URL
MODEL_URL = "https://github.com/DesireeDomingo-BSIT2B/finalproject/raw/main/save_model.keras"
model = load_model_from_url(MODEL_URL)

# Extract class names from model
class_names = model.predict(np.zeros((1, 180, 180, 3))).shape[1]  # Assuming the model input shape is (180, 180, 3)

@app.route('/classify', methods=['POST'])
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image = Image.open(request.files['image']).convert('RGB')
    img = np.array(image.resize((180, 180))) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    return jsonify({'prediction': predicted_class})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
