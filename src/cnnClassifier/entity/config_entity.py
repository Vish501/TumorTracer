from dataclasses import dataclass
from pathlib import Path
from cnnClassifier import get_logger

logger = get_logger()

@dataclass(frozen=True)
class DataIngestionConfig:
    """
    Immutable configuration class to hold all necessary paths 
    and dataset identifiers required for the data ingestion stage.
    """
    root_dir: Path          # Base directory for all ingestion outputs
    kaggle_dataset: str     # The Kaggle dataset identifier "owner/dataset"
    download_zip: Path      # Path where the downloaded ZIP file will be saved
    extracted_file: Path    # Path where the final extracted file will be stored
