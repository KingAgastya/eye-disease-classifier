from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

app = Flask(__name__)
model = tf.keras.models.load_model('best_model.h5')
class_names = ['cataract', 'conjunctivitis', 'normal']

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            img = Image.open(file.stream).convert("RGB")
            input_img = preprocess_image(img)
            pred = model.predict(input_img)
            label = class_names[np.argmax(pred)]
            confidence = round(100 * np.max(pred), 2)
            return render_template('index.html', prediction=label, confidence=confidence)
    return render_template('index.html', prediction=None)
