# app.py
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


model = load_model('model.keras')


IMG_SIZE = (224, 224)


class_labels = ["Healthy", "Early Blight", "Late Blight"]

# Allowed file extensions for security
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return "Potato Disease Classification Model is ready!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: png, jpg, jpeg, bmp, gif'}), 400

        
        os.makedirs('uploads', exist_ok=True)
        
        # Save temporarily
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=IMG_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  
        img_array = img_array / 255.0  

        # Prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = float(np.max(prediction))

        
        result = {
            "class": class_labels[predicted_class],
            "confidence": round(confidence * 100, 2),  
            "probabilities": {
                class_labels[i]: round(float(prediction[0][i]) * 100, 2) 
                for i in range(len(class_labels))
            }
        }

        
        os.remove(filepath)

        return jsonify(result)

    except Exception as e:
        # Clean up file if error occurs during processing
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create uploads directory on startup
    os.makedirs('uploads', exist_ok=True)
    
    app.run(debug=False, host='0.0.0.0', port=5000)
    
   