from dataclasses import dataclass
from pathlib import Path
from cnnClassifier import get_logger

# Initializing the logger
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


@dataclass(frozen=True)
class BaseModelConfig:
    """
    Immutable configuration class to hold all necessary paths 
    and parameters required for the base model stage.
    """
    root_dir: Path                              # Directory where models will be stored
    model_path: Path                            # Path to the downloaded base model
    updated_model_path: Path                    # Path to the updated model with custom layers
    params_image_size: tuple[int, int, int]     # Input image size, e.g., [224, 224, 3]
    params_include_top: bool                    # Whether to include fully connected layers
    params_classes: int                         # Number of output classes
    params_weights: str                         # Pre-trained weights source
    params_learning_rate: float                 # Learning rate for training
