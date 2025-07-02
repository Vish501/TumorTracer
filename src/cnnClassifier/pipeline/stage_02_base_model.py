from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.base_model import BaseModelConstruction
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = "Base Model"

class BaseModelPipeline:
    """
    Pipeline class to orchestrate the obtaining the base model and updating it:
    - Downloads a pretrained VGG16 model
    - Appends custom dense layers for classification
    - Saves both base and updated models
    """
    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        base_model_config = config_manager.get_base_model_config()

        base_model_constructor = BaseModelConstruction(config=base_model_config)
        base_model_constructor.get_model()
        base_model_constructor.updated_model()


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        BaseModelPipeline.main()

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    