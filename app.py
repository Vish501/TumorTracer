import os
import subprocess
import uuid

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pathlib import Path

from cnnClassifier.pipeline import PredictionPipeline
from cnnClassifier.utils.common import decode_image_Base64, create_directories

from cnnClassifier import get_logger

# Initialize application-wide logger
logger = get_logger()

# Set locale to UTF-8 for consistent encoding
os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

# Create Flask app and allow CORS
app = Flask(__name__)
CORS(app)


# ----------------------------
# Main App Class
# ----------------------------

class ClientApplication:
    """
    Encapsulates the model pipeline and handles file saving/prediction.
    Keeps prediction logic clean and reusable.
    """
    def __init__(self) -> None:
        # Load prediction pipeline
        self.classifier = PredictionPipeline()

        # Folder to temporarily save user-uploaded images
        self.directory_path = "artifacts/user_inputs/"

        # Ensure the folder exists, and if not create it
        create_directories([self.directory_path])


    def save_and_predict(self, base64_image: str) -> tuple[str, float]:
        """
        Saves the image temporarily, predicts using the model,
        and then deletes the image regardless of success/failure.
        """
        # Generate a unique filename to avoid collisions
        unique_id = uuid.uuid4().hex
        image_path = Path(self.directory_path) / f"{unique_id}.png"

        try:
            # Decode and save the image
            decode_image_Base64(image_string=base64_image, save_path=image_path)

            # Run model prediction
            prediction_label, confidence = self.classifier.predict_with_confidence(image_path=image_path)

            return prediction_label, confidence
        
        finally:
            # Always clean up the image file to avoid clutter
            try:
                if image_path.exists() and image_path.is_file():
                    image_path.unlink()

            except Exception as exception_error:
                logger.warning(f"Failed to delete user image {image_path.name}: {exception_error}")


# Instantiate this once so the model isn't reloaded on every request
client_application = ClientApplication()


# ----------------------------
# Routes
# ----------------------------

@app.route("/", methods=["GET"])
def home():
    """
    Home route renders a static HTML interface.
    Used for interacting with the model via browser.
    """
    logger.info(f"Flask application initialized.")
    return render_template("index.html")


@app.route("/train", methods=["GET", "POST"])
def train_route():
    """
    Endpoint to trigger the training pipeline using DVC.
    """
    logger.info("Triggering model training...")
    
    try:
        # Run DVC pipeline reproducibly
        subprocess.run(["dvc", "repro"], check=True)
        logger.info("Training completed successfully.")
        return "Training done successfully", 200
    
    except subprocess.CalledProcessError as exception_error:
        logger.error(f"/train method DVC repro failed: {exception_error}")
        return "Training failed", 500


@app.route("/predict", methods=['POST'])
def predict_route():
    """
    Endpoint to accept a base64 image via JSON and return a prediction.
    """
    input_image = request.json.get("image")

    if not input_image:
        # Return a message if the input is missing
        return jsonify({"error": "No image data received"}), 400
    
    try:
        # Perform prediction
        prediction_label, confidence = client_application.save_and_predict(input_image)

        # Return structured JSON response
        return jsonify({
                "prediction": prediction_label,
                "confidence": f"{confidence* 100:.4f}%"
                })

    except Exception as exception_error:
        logger.error(f"Unexpected prediction error: {exception_error}")
        return jsonify({"error": str(exception_error)}), 500
    

# ----------------------------
# App Entry Point
# ----------------------------

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
