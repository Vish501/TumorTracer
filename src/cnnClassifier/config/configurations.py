from pathlib import Path

from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

class ConfigurationManager:
    def __init__(self, config_file_path=CONFIG_FILE_PATH, params_file_path=PARAMS_FILE_PATH) -> None:
        """
        Reads configuration files (config.yaml and params.yaml), 
        ensures necessary directories exist, and prepares structured config objects.

        Args:
        - config_file_path (str): Path to the config.yaml file.
        - params_file_path (str): Path to the params.yaml file.
        """
        # Validate and load config.yaml
        if not Path(config_file_path).exists():
            logger.error(f"Config file not found at: {config_file_path}")
            raise FileNotFoundError(f"Config file not found at: {config_file_path}")
        self.config = read_yaml(config_file_path)

        # Validate and load params.yaml
        if not Path(config_file_path).exists():
            logger.error(f"Params file not found at: {params_file_path}")
            raise FileNotFoundError(f"Params file not found at: {params_file_path}")
        self.params = read_yaml(params_file_path)

        logger.info(f"Loading configuration from {config_file_path} and parameters from {params_file_path}")

        # Create the root artifacts directory (if not already present)
        create_directories([self.config.artifacts_root])


    def get_ingestion_config(self) -> DataIngestionConfig:
        """
        Creates and returns a DataIngestionConfig object with paths defined 
        for downloading and extracting the dataset.
        
        Returns:
        - DataIngestionConfig: Structured config object for ingestion stage.
        """
        config = self.config.data_ingestion

        # Ensure the data_ingestion directory exists
        create_directories([config.root_dir])

        # Build and return a structured configuration object for ingestion
        ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            kaggle_dataset=config.kaggle_dataset,
            download_zip=Path(config.download_zip),
            extracted_file=Path(config.extracted_file),
        )
        
        logger.info(f"DataIngestionConfig created with: {ingestion_config}")

        return ingestion_config
