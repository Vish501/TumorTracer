import kaggle
import zipfile
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

    Attributes:
    - config (DataIngestionConfig): Configuration object containing paths and dataset info

    Public Methods:
    - download_files(): Downloads the dataset zip file from Kaggle
    - extract_files(): Extracts the zip file contents into the destination folder
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
            logger.error(f"Unexpected error file downloading dataset: {exception_error}")
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
        
        except Exception as exception_error:
            logger.error(f"Unexpected error file unziping dataset: {exception_error}")
            raise exception_error
        