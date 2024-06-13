<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grapevine Image Classification</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            text-align: center;
            margin: 0;
            padding: 20px;
        }
        h1 {
            margin-bottom: 20px;
        }
        #imageContainer {
            margin-bottom: 20px;
            display: flex;
            justify-content: center;
        }
        #selectedImage {
            max-width: 400px;
            max-height: 400px;
            border: 1px solid #ccc;
            border-radius: 5px;
            display: none;
        }
        #prediction {
            font-size: 18px;
            font-weight: bold;
            margin-top: 20px;
        }
        #uploadButton, #classifyButton {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            margin: 10px;
            transition: background-color 0.3s;
        }
        #uploadButton:hover, #classifyButton:hover {
            background-color: #45a049;
        }
        #loadingOverlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.7);
            display: none;
        }
        #loadingSpinner {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
    <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@3.6.0/dist/tf.min.js"></script>
    <script>
        let model;
        const classNames = ['Ak', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli'];
        const datasetURL = 'https://raw.githubusercontent.com/DesireeDomingo-BSIT2B/finalproject/main/Grapevine_Leaves/';

        async function loadModel() {
            try {
                // Show loading overlay
                document.getElementById('loadingOverlay').style.display = 'block';

                model = await tf.loadLayersModel('https://github.com/DesireeDomingo-BSIT2B/finalproject/blob/main/save_model.keras');
                console.log("Model loaded successfully");
                // Enable classify button after model is loaded
                document.getElementById('classifyButton').disabled = false;
                // Hide loading overlay
                document.getElementById('loadingOverlay').style.display = 'none';
            } catch (error) {
                console.error("Error loading the model:", error);
            }
        }

        function handleFileSelect(event) {
            const file = event.target.files[0];
            const reader = new FileReader();
            reader.onload = function() {
                const output = document.getElementById('selectedImage');
                output.src = reader.result;
                output.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        async function classifyImage() {
            if (!model) {
                alert("Model not loaded yet. Please wait and try again.");
                return;
            }

            const inputElement = document.getElementById('imageInput');
            if (inputElement.files.length === 0) {
                alert("Please select an image first.");
                return;
            }

            const image = inputElement.files[0];
            const img = new Image();
            img.src = URL.createObjectURL(image);
            await img.decode();
            const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]).toFloat().expandDims();
            const normalizedTensor = tensor.div(tf.scalar(255));
            const predictions = model.predict(normalizedTensor);
            const classIdx = predictions.argMax(-1).dataSync()[0];

            document.getElementById('prediction').innerText = 'Prediction: ' + classNames[classIdx];
        }

        window.onload = loadModel;
    </script>
</head>
<body>
    <h1>Image Classification</h1>
    <div id="imageContainer">
        <input type="file" id="imageInput" accept="image/*" style="display: none;" onchange="handleFileSelect(event)">
        <img id="selectedImage">
    </div>
    <button id="uploadButton" onclick="document.getElementById('imageInput').click()">Upload Image</button>
    <button id="classifyButton" onclick="classifyImage()" disabled>Classify</button>
    <div id="prediction"></div>
    <div id="loadingOverlay">
        <div id="loadingSpinner"></div>
    </div>
</body>
</html>
