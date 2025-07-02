from cnnClassifier.config.configurations import ConfigurationManager
from cnnClassifier.components.data_ingestion import DataIngestion
from cnnClassifier.utils.kaggle_auth import setup_kaggle_auth_from_secret
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = "Data Ingestion"

class DataIngestionTrainingPipeline:
    """
    Pipeline class to orchestrate the data ingestion process:
    - Load configuration
    - Download dataset from Kaggle
    - Extract dataset contents
    """
    @staticmethod
    def main():
        setup_kaggle_auth_from_secret(secret_env_var="KAGGLE_JSON")

        config_manager = ConfigurationManager()
        ingestion_config = config_manager.get_ingestion_config()

        data_ingestor = DataIngestion(config=ingestion_config)
        data_ingestor.download_files()
        data_ingestor.extract_files()


if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        DataIngestionTrainingPipeline.main()

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
    