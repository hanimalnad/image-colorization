from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the trained model
model = load_model('colorization_model.h5')

# Define image size
IMAGE_SIZE = 128

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/colorize', methods=['POST'])
def colorize():
    if 'image' not in request.files:
        return 'No image uploaded', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    # Save the uploaded file
    filepath = os.path.join('static', 'uploads', file.filename)
    file.save(filepath)

    # Load and preprocess the image
    gray_img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    gray_img = cv2.resize(gray_img, (IMAGE_SIZE, IMAGE_SIZE))
    gray_img = gray_img.astype('float32') / 255.0
    gray_img = gray_img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 1)

    # Predict the ab channels
    pred_ab = model.predict(gray_img)[0]
    pred_ab = pred_ab * 128.0

    # Reconstruct the Lab image
    l_channel = gray_img[0][:,:,0] * 255.0
    lab_output = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
    lab_output[:,:,0] = l_channel
    lab_output[:,:,1:] = pred_ab

    # Convert Lab to RGB
    rgb_output = cv2.cvtColor(lab_output.astype(np.uint8), cv2.COLOR_LAB2RGB)

    # Save the colorized image
    output_path = os.path.join('static', 'outputs', 'colorized_' + file.filename)
    cv2.imwrite(output_path, cv2.cvtColor(rgb_output, cv2.COLOR_RGB2BGR))

    return render_template('result.html', original=filepath, colorized=output_path)

if __name__ == '__main__':
    os.makedirs(os.path.join('static', 'uploads'), exist_ok=True)
    os.makedirs(os.path.join('static', 'outputs'), exist_ok=True)
    app.run(debug=True)
