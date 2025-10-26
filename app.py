from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Force TensorFlow to use CPU only
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

app = Flask(__name__)

# Load model with error handling
try:
    model = tf.keras.models.load_model('best_model.h5', compile=False)
except Exception as e:
    print("Failed to load model:", e)
    model = None

# Define class names
class_names = ['cataract', 'conjunctivitis', 'normal']

# Preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Main route
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and model:
            try:
                img = Image.open(file.stream).convert("RGB")
                input_img = preprocess_image(img)
                pred = model.predict(input_img)
                label = class_names[np.argmax(pred)]
                confidence = round(100 * np.max(pred), 2)
                return render_template('index.html', prediction=label, confidence=confidence)
            except Exception as e:
                print("Prediction error:", e)
                return render_template('index.html', prediction="Error", confidence=0)
    return render_template('index.html', prediction=None)

# Run the app (for local testing or Render deployment)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
