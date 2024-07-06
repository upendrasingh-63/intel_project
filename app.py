import base64
import io
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim

app = Flask(__name__)

# Load models
pixelation_model = tf.keras.models.load_model('models/pixelation_detection_model.h5')
correction_model = tf.keras.models.load_model('models/pixelation_correction_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/check_pixelation', methods=['POST'])
def check_pixelation():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    prediction = pixelation_model.predict(image_array)
    pixelated = np.argmax(prediction, axis=1)[0] == 1
    
    if pixelated:
        message = "The image is pixelated."
    else:
        message = "The image is not pixelated."
        
    return jsonify({'message': message})

@app.route('/correct_image', methods=['POST'])
def correct_image():
    file = request.files['image']
    image = Image.open(file.stream).convert('RGB')
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    corrected_image = correction_model.predict(image_array)[0]
    corrected_image = (corrected_image * 255).astype(np.uint8)
    
    # Convert the corrected image to base64
    pil_image = Image.fromarray(corrected_image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    # Calculate PSNR and SSIM
    original_image = (image_array[0] * 255).astype(np.uint8)
    psnr_value = psnr(original_image, corrected_image)
    ssim_value = ssim(original_image, corrected_image, multichannel=True)
    
    return jsonify({
        'corrected_image': img_str,
        'psnr': psnr_value,
        'ssim': ssim_value
    })

if __name__ == '__main__':
    app.run(debug=True)
