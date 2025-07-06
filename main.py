from cnnClassifier.pipeline import DataIngestionTrainingPipeline, BaseModelPipeline, ModelTrainingPipeline
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

# Stage identifier for logging purposes
STAGE_NAME = ""

if __name__ == "__main__":
    try:
        # Running the Data Ingestion Pipeline
        STAGE_NAME = "Data Ingestion"
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        DataIngestionTrainingPipeline.main() # Run the pipeline
        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
        logger.info(f"******************************************")

        # Running the Base Model Pipeline
        STAGE_NAME = "Base Model"
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        BaseModelPipeline.main() # Run the pipeline
        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
        logger.info(f"******************************************")

        # Running the Model Training Pipeline
        STAGE_NAME = "Model Training"
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        ModelTrainingPipeline.main() # Run the pipeline
        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
        logger.info(f"******************************************")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during {STAGE_NAME} pipeline: {exception}")
        raise exception
