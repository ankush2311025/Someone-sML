# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load your model
model = load_model('model.keras')

# Adjust to your training image size (e.g. 224x224, 128x128)
IMG_SIZE = (224, 224)

# ✅ Define your class labels in the same order as your training data
class_labels = ["Healthy", "Early Blight", "Late Blight"]

@app.route('/')
def home():
    return "Image classification model is ready!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'})

        # Save temporarily
        filepath = os.path.join('uploads', file.filename)
        os.makedirs('uploads', exist_ok=True)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # shape (1, H, W, C)
        img_array = img_array / 255.0  # normalize if trained that way

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        # ✅ Return human-readable result
        result = {
            "class": class_labels[predicted_class],
            "confidence": confidence
        }

        # Clean up
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
