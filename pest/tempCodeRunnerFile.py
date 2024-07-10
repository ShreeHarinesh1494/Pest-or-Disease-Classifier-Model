from flask import Flask, render_template, request
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials

# Azure setup
subscription_key = "cd6cb499771e4b7f83f0e9a4aa0d5494"
endpoint = "https://franz-k.cognitiveservices.azure.com/"
computervision_client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# Load model
model = load_model("C:\\Users\\hari\\OneDrive\\Desktop\\model2\\Cotton-Plant-Disease-Classification-Web-Application\\my_model.h5")
print('*Model loaded')

def predict_disease(image_path, model, threshold=0.7):
    img = image.load_img(image_path, target_size=(256, 256))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    predictions = model.predict(img_array)
    classes = ["bacterial_blight", "curl_virus", "fusarium_wilt", "healthy"]
    
    max_prob = np.max(predictions)
    predicted_class = classes[np.argmax(predictions)]
    
    if max_prob < threshold:
        return "Unknown disease or dataset not found", max_prob
    else:
        return predicted_class, max_prob

def check_if_cotton_plant(image_path):
    with open(image_path, "rb") as image_stream:
        analysis = computervision_client.analyze_image_in_stream(image_stream, visual_features=["Tags"])
    
    cotton_tags = ["cotton", "plant", "leaf"]
    confidence_threshold = 0.6
    cotton_confidence = 0
    plant_confidence = 0
    
    for tag in analysis.tags:
        if tag.name == "cotton":
            cotton_confidence = tag.confidence
        elif tag.name in ["plant", "leaf"]:
            plant_confidence = max(plant_confidence, tag.confidence)
    
    is_cotton = cotton_confidence > confidence_threshold
    is_plant = plant_confidence > confidence_threshold
    
    return is_cotton, is_plant, cotton_confidence, plant_confidence

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        is_cotton, is_plant, cotton_conf, plant_conf = check_if_cotton_plant(file_path)
        
        if not is_plant:
            return f"The uploaded image is not a plant (plant confidence: {plant_conf:.2f})"
        elif not is_cotton:
            return f"The uploaded image is a plant but not cotton (cotton confidence: {cotton_conf:.2f})"
        else:
            prediction, confidence = predict_disease(file_path, model)
            return f"Prediction: {prediction} (confidence: {confidence:.2f})"
    
    return None

if __name__ == '__main__':
    app.run(debug=True)