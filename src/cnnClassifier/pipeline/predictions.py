from pathlib import Path

from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.predictions import Predictions
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = "Prediction"

class PredictionPipeline:
    """
    Thin wrapper around `Predictions` that
    1. Loads the prediction configuration once,
    2. Delegates all prediction calls.
    """
    def __init__(self) -> None:
        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_config()
        self.prediction_constructor = Predictions(config=prediction_config)


    def predict(self, image_path: Path) -> str:
        """
        Return the predicted label for a single image.
        """
        return self.prediction_constructor.predict(image_path=image_path)


    def predict_with_confidence(self, image_path: Path) -> tuple[str, float]:
        """
        Return (label, confidence) for a single image.
        Confidence is a float in [0, 1].
        """
        return self.prediction_constructor.predict_with_confidence(image_path=image_path)


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        Predictions = PredictionPipeline()
        Predictions.predict(image_path="artifacts/data_ingestion/Data/train/normal/2 - Copy - Copy.png")

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    