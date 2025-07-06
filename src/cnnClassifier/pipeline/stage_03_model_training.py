from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.model_training import ModelTraining
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = "Model Training"

class ModelTrainingPipeline:
    """
    Pipeline class to train the updated model.
    """
    @staticmethod
    def main():
        config_manager = ConfigurationManager()
        training_config = config_manager.get_training_config()

        training_constructor = ModelTraining(config=training_config)
        training_constructor.get_base_model()
        training_constructor.get_data_generators()
        training_constructor.train()
        training_constructor.save_class_indices()


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        ModelTrainingPipeline.main()

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    