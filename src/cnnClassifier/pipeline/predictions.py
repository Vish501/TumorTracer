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
    Pipeline class to orchestrate the data ingestion process:
    - Load configuration
    - Download dataset from Kaggle
    - Extract dataset contents
    """
    @staticmethod
    def main(image_path: Path):
        config_manager = ConfigurationManager()
        prediction_config = config_manager.get_prediction_config()

        prediction_constructor = Predictions(config=prediction_config)
        prediction_constructor.predict(image_path=image_path)


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        PredictionPipeline.main(image_path="artifacts/data_ingestion/Data/train/normal/2 - Copy - Copy.png")

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    