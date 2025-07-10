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
    Pipeline class to predict.
    """
    def __init___(self):
        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_config()
        self.prediction_constructor = Predictions(config=prediction_config)


    def predict(self, image_path: Path):
        self.prediction_constructor.predict(image_path=image_path)


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        Predictions = PredictionPipeline()
        Predictions.main(image_path="artifacts/data_ingestion/Data/train/normal/2 - Copy - Copy.png")

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    