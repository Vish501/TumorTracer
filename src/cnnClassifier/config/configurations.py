from pathlib import Path

from cnnClassifier.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from cnnClassifier.utils.common import read_yaml, create_directories
from cnnClassifier.entity.config_entity import DataIngestionConfig, BaseModelConfig, ModelTrainingConfig
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
            rename_map_yaml=Path(config.rename_map_yaml)
        )
        
        logger.info(f"DataIngestionConfig created with: {ingestion_config}")

        return ingestion_config


    def get_base_model_config(self) -> BaseModelConfig:
        """
        Prepares and returns the BaseModelConfig object.

        Returns:
        - BaseModelConfig: Structured config for downloading and updating base model.
        """
        config = self.config.base_model
        params = self.params.base_model

        # Ensure the data_ingestion directory exists
        create_directories([config.root_dir])

        # Build and return a structured configuration object for base model construction
        base_model_config = BaseModelConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            updated_model_path=Path(config.updated_model_path),
            params_image_size=tuple(params.IMAGE_SIZE),             # Convert list to tuple for immutability
            params_include_top=params.INCLUDE_TOP,
            params_classes=params.CLASSES,
            params_weights=params.WEIGHTS,
            params_learning_rate=params.LEARNING_RATE,
        )
        
        logger.info(f"BaseModelConfig created with: {base_model_config}")

        return base_model_config
    
    
    def get_training_config(self) -> ModelTrainingConfig:
        """
        Prepares and returns the ModelTrainingConfig object.

        Returns:
        - ModelTrainingConfig: Structured config for training the updated base model.
        """
        training_config = self.config.model_training
        training_params = self.params.model_training

        # Ensure the data_ingestion directory exists
        create_directories([training_config.root_dir])

        # Load augmentation params only if augmentation is enabled and params for it are present
        params_for_augmentation = {}
        if training_params.AUGMENTATION and hasattr(training_params, "AUGMENTATION_PARAMS"):
            params_for_augmentation = dict(training_params.AUGMENTATION_PARAMS )

        training_config = ModelTrainingConfig(
            root_dir=Path(training_config.root_dir),
            trained_model_path=Path(training_config.trained_model_path),
            updated_base_model=Path(training_config.updated_model_path),
            training_data=Path(training_config.training_dataset),
            validation_data=Path(training_config.validation_dataset),
            params_augmentation=training_params.AUGMENTATION,
            params_checkpoint=training_params.CHECKPOINT,
            params_mlflow=training_params.MLFLOW_TRACKING,
            params_image_size=tuple(training_params.IMAGE_SIZE),
            params_batch_size=training_params.BATCH_SIZE,
            params_epochs=training_params.EPOCHS,
            params_optimizer=training_params.OPTIMIZER,
            params_learning_rate=training_params.LEARNING_RATE,
            params_if_augmentation=params_for_augmentation,
        )

        logger.info(f"ModelTrainingConfig created with: {training_config}")

        return training_config
    