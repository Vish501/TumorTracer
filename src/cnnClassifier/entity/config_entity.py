from dataclasses import dataclass
from pathlib import Path
from cnnClassifier import get_logger
from typing import Optional, Dict, Any

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
    rename_map_yaml: Path   # Path where the dataset renaming file is located
    

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


@dataclass(frozen=True)
class ModelTrainingConfig:
    """
    Immutable configuration class to store all parameters 
    and paths required for model training. 
    """
    root_dir: Path                                          # Directory for training artifacts
    trained_model_path: Path                                # Final model output path
    updated_base_model: Path                                # Pretrained model with custom head
    training_data: Path                                     # Directory with training images
    validation_data: Path                                   # Directory with validation images
    params_augmentation: bool                               # Whether to apply augmentation
    params_checkpoint: bool                                 # Whether created models need to be checkpointed
    params_mlflow: bool                                     # Whether models need to be tracker in mlflow
    params_image_size: tuple[int, int, int]                 # Input image size, e.g., [224, 224, 3]
    params_batch_size: int                                  # Batch size for training
    params_epochs: int                                      # Total epochs
    params_optimizer: str                                   # Optimizer to be used when recompling model
    params_learning_rate: float                             # Learning rate for training
    params_if_augmentation: Optional[Dict[str, Any]] = None # Dict of augmentation hyperparameters


@dataclass(frozen=True)
class PredictionConfig:
    """
    Immutable configuration class to store all parameters 
    and paths required for model prediction. 
    """
    trained_model_path: Path                     # Path to the trained model that will be used to predict
    class_indices_path: Path                    # Path to the model's class_indices
    params_image_size: tuple[int, int, int]     # Input image size, e.g., [224, 224, 3]
    params_normalization: float 				# Normalization factor used in training
