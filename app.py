from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import io
import gdown


app = Flask(__name__)

# Load the model
model_path = 'model/vgg16_model.h5'
if not os.path.exists(model_path):
    os.makedirs('model', exist_ok=True)
    url = 'https://drive.google.com/uc?id=1-2eIMPNNFruPTQEGInsSDKWncrHP7sOy'
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', prediction="No file uploaded.")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', prediction="No selected file.")

        img_bytes = io.BytesIO(file.read())
        img = image.load_img(img_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)

        preds = model.predict(img_array)
        class_names = ['Healthy', 'Glaucoma']  # <-- Use your actual class labels
        predicted_class = np.argmax(preds[0])
        confidence = preds[0][predicted_class]
        prediction_text = f"{class_names[predicted_class]} ({confidence*100:.2f}%)"


        return render_template('index.html', prediction=prediction_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
