from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.model_selection import ModelSelection
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = "Model Selection"

class ModelSelectionPipeline:
    """
    Thin wrapper around `Predictions` that
    1. loads the prediction configuration once,
    2. delegates all prediction calls.
    """
    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        selector_config = config_manager.get_model_selector_config()
        ModelSelection(config=selector_config)


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        ModelSelectionPipeline.main()

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    
