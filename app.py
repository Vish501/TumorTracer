from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

STAGE_NAME = "Data Ingestion"

if __name__ == "__main__":
    try:
        logger.info(f">>>> {STAGE_NAME} stage has started <<<<")
        
        # Run the pipeline
        ingestion_pipeline = DataIngestionTrainingPipeline()
        ingestion_pipeline.main()

        logger.info(f">>>> {STAGE_NAME} stage has completed <<<<")
    
    except Exception as exception:
        # Catch and log any unexpected errors during the ingestion stage
        logger.exception(f"Unexpected error during data ingestion pipeline: {exception}")
        raise exception
