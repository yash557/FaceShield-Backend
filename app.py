from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import os
import io

app = Flask(__name__)
CORS(app)

model = None

def get_model():
    global model
    if model is None:
        # Loading the highly stable .h5 format
        model = tf.keras.models.load_model("model.h5", compile=False)
    return model

@app.route("/")
def home():
    return "FaceShield Backend is Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files["image"]
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Read image safely from memory
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)

        # Resize properly for the model
        image = cv2.resize(image, (128, 128))
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)

        # Fetch model and predict
        loaded_model = get_model() 
        prediction = loaded_model.predict(image)

        result = "REAL" if prediction[0][0] > 0.5 else "FAKE"

        return jsonify({
            "result": result,
            "confidence": round(float(prediction[0][0]), 4)
        })

    except Exception as e:
        print("Prediction Error:", str(e))
        return jsonify({"error": "Failed to process image. Make sure it is a valid image file."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Threaded=True helps handle multiple requests at once
    app.run(host="0.0.0.0", port=port, threaded=True)