import os
import subprocess
import uuid

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pathlib import Path

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

        # Creating directory path
        self.directory_path = "artifacts/user_inputs/"

        # Creating user_inputs directory
        create_directories([self.directory_path])


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
    try:
        subprocess.run(["dvc", "repro"], check=True)
        logger.info("Training completed successfully.")
    except subprocess.CalledProcessError as exception_error:
        logger.error(f"/train method DVC repro failed: {exception_error}")
        return "Training failed", 500


@app.route("/predict", methods=['POST'])
def predict_route():
    input_image = request.json['image']

    if not input_image:
        return jsonify({"error": "No image data received"}), 400
    
    # Unique filepath
    unique_id = uuid.uuid4().hex
    image_path = Path(client_application.directory_path) / f"/{unique_id}.png"

    try:
        # Save decoded image to file
        decode_image_Base64(image_string=input_image, save_path=image_path)

        # Predict class
        prediction_label, confidence = client_application.classifier.predict_with_confidence(image_path=image_path)

        return jsonify({
            "prediction": prediction_label,
            "confidence": f"{confidence* 100:.4f}%"
            })
    
    except Exception as exception_error:
        logger.error(f"Unexpected prediction error: {exception_error}")
        return jsonify({"error": str(exception_error)}), 500
    
    finally:
        try:
            if image_path.exists() and image_path.is_file():
                image_path.unlink()

        except Exception as exception_error:
            logger.warning(f"Failed to delete user image {unique_id}: {exception_error}")


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
