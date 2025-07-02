import kaggle
import zipfile
from box import ConfigBox
from pathlib import Path
from cnnClassifier.utils.common import read_yaml
from cnnClassifier.entity.config_entity import DataIngestionConfig
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()


class DataIngestion:
    """
    Handles the dataset ingestion step of the pipeline.

    This includes:
    - Downloading the dataset from Kaggle using the official Kaggle API
    - Extracting the downloaded ZIP file to a specified directory
    - Renaming class folders using a YAML-defined mapping

    Attributes:
    - config (DataIngestionConfig): Configuration object containing paths and dataset info

    Public Methods:
    - download_files(): Downloads the dataset zip file from Kaggle
    - extract_files(): Extracts the zip file contents into the destination folder
    - rename_class_folders_from_yaml(): Renames raw class folders based on a YAML mapping for provided folders

    Private Methods:
    - _load_renaming_file: Loads YAML file from location and returns it as ConfigBox
    """
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def download_files(self) -> None:
        """
        Downloads dataset from Kaggle using kaggle API.
        """
        try:
            kaggle.api.dataset_download_files(
                dataset=self.config.kaggle_dataset,
                path=self.config.root_dir,
                unzip=False
            )

            logger.info(f"Successfully downloaded dataset {self.config.kaggle_dataset} at: {self.config.root_dir}")

        except Exception as exception_error:
            logger.error(f"Unexpected error while downloading dataset: {exception_error}")
            raise exception_error


    def extract_files(self) -> None:
        """
        Extracts the downloaded ZIP file.
        """
        try:
            with zipfile.ZipFile(self.config.download_zip, "r") as zip_ref:
                zip_ref.extractall(self.config.root_dir)
                logger.info(f"Successfully extracted dataset {self.config.kaggle_dataset} at: {self.config.extracted_file}")

            if not self.config.extracted_file.exists():
                logger.warning(f"Expected file not found after extraction: {self.config.extracted_file}")

        except zipfile.BadZipFile:
            logger.error(f"Invalid zip file format.")
            raise
        
        except Exception as exception_error:
            logger.error(f"Unexpected error file unziping dataset: {exception_error}")
            raise


    def rename_class_folders_from_yaml(self) -> None:
        """
        Renames raw class folders based on a YAML mapping for provided folders.
        """
        try:
            renaming_map = self._load_renaming_file()

            for folder, class_mapping in renaming_map.items():
                current_path = self.config.extracted_file / folder

                if not current_path.exists():
                    logger.error(f"While renaming could not find {folder} at: {current_path}")
                    continue
                
                for old_name, new_name in class_mapping.items():
                    old_path = current_path / old_name
                    new_path = current_path / new_name

                    if new_path.exists():
                        logger.error(f"Target folder {new_name} already exists. Skipping rename.")
                        continue

                    if not old_path.exists():
                        logger.error(f"Source folder missing: {old_path}")
                        continue

                    old_path.rename(new_path)
                    logger.info(f"Successfully renamed '{old_name}' to '{new_path}' in {folder} folder.")
                    
        except Exception as exception_error:
            logger.error(f"Unexpected error while renaming folders: {exception_error}")
            raise
    
    def _load_renaming_file(self) -> ConfigBox:
        """
        Loads the folder renaming mapping file.

        Returns:
        - ConfigBox: YAML content with old-to-new folder name mappings.
        """
        yaml_path = Path(self.config.rename_map_yaml)

        if not yaml_path.exists():
            logger.error(f"Renaming YAML map not found at: {yaml_path}")
            raise FileNotFoundError(f"Renaming YAML map not found at: {yaml_path}")
        return read_yaml(yaml_path)
