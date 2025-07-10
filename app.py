import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin
from pathlib import Path

from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.predictions import Predictions
from cnnClassifier.pipeline import PredictionPipeline
from cnnClassifier.utils.common import decode_image_Base64, create_directories

from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Set locale environment variables 
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

class ClientApplication:
    def __init__(self) -> None:
        # Initializing the predictions classifier
        self.classifier = PredictionPipeline()

        # Setting image path
        self.image_path = "artifacts/user_inputs/input_image.png"

        # Creating user_inputs directory
        create_directories([self.image_path.parent])
        

# Instantiate once and reuse
client_application = ClientApplication()

@app.route("/", methods=["GET"])
def home():
    logger.info(f"Flask application initialized.")
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train_route():
    logger.info("Triggering model training...")
    
    # Running the pipeline
    os.system("dvc repro")


@app.route("/predict", methods=['POST'])
def predict_route():
    input_image = request.json['image']

    if not input_image:
        return jsonify({"error": "No image data received"}), 400
    
    try:
        # Save decoded image to file
        decode_image_Base64(image_string=input_image, save_path=client_application.image_path)

        # Predict class
        prediction_label, confidence = client_application.classifier.predict_with_confidence(image_path=client_application.image_path)

        return jsonify({"prediction": prediction_label, "confidence": f"{confidence* 100:.4f}%"})
    
    except Exception as exception_error:
        logger.error(f"Unexpected prediction error: {exception_error}")
        return jsonify({"error": str(exception_error)}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
