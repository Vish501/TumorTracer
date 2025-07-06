import os
import random
import numpy as np
import tensorflow as tf
from math import ceil
from typing import Union
from tensorflow.keras.preprocessing.image import ImageDataGenerator, DirectoryIterator # type: ignore
from tensorflow.keras.callbacks import Callback # type: ignore
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from cnnClassifier.utils.common import create_directories, save_json, save_tf_model
from cnnClassifier.entity.config_entity import ModelTrainingConfig
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

class CustomCheckpointCallback(Callback):
    """
    A custom Keras callback that saves the model whenever the validation accuracy improves.
    The model file name includes the epoch number, training accuracy, and validation accuracy.

    Attributes:
        save_directory (Path): Directory where the model should be saved.
        model (tf.keras.Model): Reference to the model being trained.
        best_val_acc (float): Tracks the best validation accuracy seen so far.
    """
    def __init__(self, save_directory: Path, model_to_save: tf.keras.Model) -> None:
        """
        Initializes the callback with directory and model reference.

        Args:
            save_directory (Path): Where to save the best model checkpoints.
            model (tf.keras.Model): The model being trained (required for manual saving).
        """
        super().__init__()
        self.best_val_accuracy = 0
        self.save_directory = save_directory
        self.model_to_save = model_to_save


    def on_epoch_end(self, epoch: int, logs: dict[str, float] = None) -> None:
        """
        Called at the end of each epoch. Checks validation accuracy and saves model if improved.

        Args:
            epoch (int): The index of the current epoch (0-based).
            logs (dict[str, float]): Dictionary containing metrics like accuracy, val_accuracy, etc.
        """
        # Validating logs is available
        if not logs:
            logger.warning(f"Logs not found. Skipping model save.")
            return

        # Getting metrics from logs
        val_acc = logs.get("val_accuracy")
        train_acc = logs.get("accuracy")

        # If validation accuracy isn't available, skip saving
        if val_acc is None:
            logger.warning("val_accuracy not found in logs. Skipping model save.")
            return

        # If current model is better than prior best model, saving the model
        if val_acc > self.best_val_accuracy:
            self.best_val_accuracy = val_acc

            # Construct model filename with padded epoch, train_acc, and val_acc
            model_path = Path(self.save_directory / f"model_e{epoch+1:02d}_acc{train_acc:.4f}_vacc{val_acc:.4f}.h5")

            # Save the model to disk
            save_tf_model(save_path=model_path, model=self.model_to_save)
            logger.info(f"Saved new best model at {model_path}")


class ModelTraining:
    """
    Initializes training pipeline with given configuration.

    Core Responsibilities:
    - Load a pre-defined base model from disk and recompile it with a fresh optimizer.
    - Set up data generators for training and validation, with optional augmentation.
    - Train the model across multiple epochs with optional checkpointing.
    - Resume training from where it left off.
    - Save class label mappings and model artifacts.

    Public Methods:
    - get_base_model(): Load and compile the pre-trained base model.
    - get_data_generators(): Prepare train and validation data generators.
    - train(): Train the model with checkpointing on best validation accuracy.
    - resume_train(add_epochs): Continue training the model for additional epochs.
    - save_class_indices(): Save class-to-index mapping as JSON for reproducibility.

    Private Utilities:
    - _build_generator(): Helper to construct data generators with standard settings.
    - _create_checkpoint(): Creates a checkpoint directory and stores training metadata.
    - _get_optimizer(): Returns optimizer based on config.
    - _count_images_in_directory(): Utility to count image files recursively.
    """
    def __init__(self, config: ModelTrainingConfig) -> None:
        """
        Initializes the model training pipeline.

        - Sets random seeds for reproducibility.
        - Prepares internal attributes for managing training, checkpoints, and model state.
        """
        # Store configuration
        self.config = config

        # Set random seeds
        seed = self.config.params_seed if hasattr(self.config, "params_seed") else 1234
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # Initialize model and training attributes
        self.output_model = None
        self.training_generator = None
        self.valid_generator = None
        self.training_images = None
        self.validation_images = None
        self.last_epoch = 0
        self.additional_epochs = 0
        self.best_val_accuracy = 0

        # Initialize checkpoint directory path with timestamp
        curr_time = datetime.now().strftime("%Y%m%d_%H%M")
        self.checkpoint_path = Path(self.config.root_dir / f"Checkpoint_{curr_time}")


    def get_base_model(self) -> None:
        """
        Loads the base model form specified path.
        """
        model_path = Path(self.config.updated_base_model)

        if not model_path.exists():
            logger.error(f"Could not find model at {model_path}. Run the Base Model pipeline stage first.")
            raise FileNotFoundError(f"Could not find model at {model_path}. Run the Base Model pipeline stage first.")
        
        try:
            self.output_model = tf.keras.models.load_model(model_path)
            logger.info(f"Successfully loaded the base model from {model_path}.")

            # Enabling Eager Execution (optional in TF 2.x) due to library requirement
            tf.config.run_functions_eagerly(True)

            # Recompile model with a fresh optimizer (required after loading)
            self.output_model.compile(
                optimizer=self._get_optimizer(),
                loss=tf.keras.losses.CategoricalCrossentropy(),
                metrics=["accuracy"]
            )
            logger.info(f"Successfully recomplied the model.")

        except Exception as exception_error:
            logger.error(f"Unexpected error while loading the update base model at {model_path}: {exception_error}")
            raise 
            
    
    def get_data_generators(self) -> None:
        """
        Update train and validation data generators using ImageDataGenerator.
        Applies augmentation only on training data if enabled.
        """
        try:
            logger.info("Preparing ImageDataGenerators...")

            train_datagen = ImageDataGenerator(rescale=1.0/255, **self.config.params_if_augmentation)
            valid_datagen = ImageDataGenerator(rescale=1.0/255)

            self.training_generator = self._build_generator(train_datagen, self.config.training_data, "Train")
            self.valid_generator = self._build_generator(valid_datagen, self.config.validation_data, "Valid")

            # Ensure class-to-index mapping is consistent
            if self.training_generator.class_indices != self.valid_generator.class_indices:
                logger.error("Mismatch in class indices between train and validation generators!")
                raise ValueError("Mismatch in class indices between train and validation generators!")

            logger.info("ImageDataGenerators created successfully.")
        
        except Exception as exception_error:
            logger.error(f"Unexpected error while creating data generators: {exception_error}")
            raise


    def train(self) -> None:
        """
        Trains the model using prepared generators.
        """
        if self.output_model == None:
            logger.error("Base model not found. Run get_base_model() before calling train().")
            raise ValueError("Base model not found. Run get_base_model() before calling train().")
        
        if (self.config.params_checkpoint) and (not self.checkpoint_path.exists()) and (self.config.params_epochs >= 1):
            self._create_checkpoint()
        
        try:
            logger.info("Initializing model training...")

            # Counting the images in each of the datasets
            self.training_images = self._count_images_in_directory(self.config.training_data)
            self.validation_images = self._count_images_in_directory(self.config.validation_data)

            # Initializing the custom callback from tf.Keras
            custom_callback = CustomCheckpointCallback(save_directory=self.checkpoint_path, model_to_save=self.output_model)

            # Fitting the model
            total_epochs = self.config.params_epochs + self.additional_epochs

            self.output_model.fit(
                self.training_generator,
                validation_data=self.valid_generator,
                initial_epoch=self.last_epoch,         # Sets starting point for correct logging
                epochs=total_epochs,                        
                steps_per_epoch=ceil(self.training_images / self.config.params_batch_size),
                validation_steps=ceil(self.validation_images / self.config.params_batch_size),
                callbacks=[custom_callback],
                verbose=1
            )
    
            # Updating number of epochs completed for resume train
            self.last_epoch = total_epochs

            logger.info("Successfully trained model based on provided parameters.")
            save_tf_model(save_path=self.config.trained_model_path, model=self.output_model)

        except Exception as exception_error:
            logger.error(f"Unexpected error while training the model: {exception_error}")
            raise


    def resume_train(self, add_epochs: int) -> None:
        """
        Resumes model training for additional number of epochs.
        """
        try:
            if self.additional_epochs == None:
                self.additional_epochs = 0
            self.additional_epochs += add_epochs

            self.train()

        except Exception as exception_error:
            logger.error(f"Unexpected error while resuming training: {exception_error}")
            raise
    

    def save_class_indices(self) -> None:
        """
        Saves the class index mapping as a JSON file for future reference.
        """
        if self.training_generator == None:
            logger.error("Class indices not found. Run get_data_generators() before calling save_class_indices().")
            raise ValueError("Class indices not found. Run get_data_generators() before calling save_class_indices().")

        try:
            save_path = Path(self.config.root_dir / "class_indices.json")
            save_json(save_path=save_path, data=self.training_generator.class_indices)

            if self.config.params_checkpoint and self.checkpoint_path.exists():
                checkpoint_save_path = Path(self.checkpoint_path / "class_indices.json")
                save_json(save_path=checkpoint_save_path, data=self.training_generator.class_indices)

        except Exception as exception_error:
            logger.error(f"Unexpected error while saving class indices: {exception_error}")
            raise


    def _build_generator(self, datagen: ImageDataGenerator, data_path: Union[str, Path], tag: str) -> DirectoryIterator:
        """
        Helper to build a flow_from_directory generator with consistent options.

        Args:
        - datagen (ImageDataGenerator): Instance of the ImageDataGenerator.
        - data_path (Union[str, Path]): Path to the directory containing images.
        - tag (str): Label for logging context ("Train" or "Valid").

        Returns:
        - DirectoryIterator: Configured Keras generator for the given directory.
        """
        try:
            data_path = Path(data_path)

            if not data_path.exists():
                logger.error(f"{tag.title()} directory not found: {data_path}")
                raise FileNotFoundError(f"{tag.title()} directory not found: {data_path}")

            # Building generator
            generator_unit = datagen.flow_from_directory(
                directory=data_path,
                target_size=self.config.params_image_size[:2],
                batch_size=self.config.params_batch_size,
                class_mode="categorical",
                shuffle=True,
            )

            return generator_unit

        except Exception as exception_error:
            logger.error(f"Unexpected error while build generator: {exception_error}")
            raise
    

    def _create_checkpoint(self) -> None:
        """
        Creates a checkpoint directory and saves the training configuration as a JSON file.

        Purpose:
        - Ensures the checkpoint directory exists.
        - Saves the current training configuration (hyperparameters) used in that run.
        - Helps with reproducibility and traceability for saved models.
        """
        try:
            logger.info("Creating checkpoint directory...")

            save_path = Path(self.checkpoint_path / "params_used.json")
            create_directories([self.checkpoint_path])

            # Convert all Path objects to str recursively
            config_dict = self._convert_paths_to_str(asdict(self.config))

            save_json(save_path=save_path, data=config_dict)
            
            logger.info(f"Checkpoint directory created.")

        except Exception as exception_error:
            logger.error(f"Unexpected error while creating checkpoint directroy: {exception_error}")
            raise 


    def _get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        Dynamically selects and returns a TensorFlow optimizer based on the configuration.

        Returns:
            tf.keras.optimizers.Optimizer: Configured optimizer instance for model compilation.
        """
        try:
            # Normalize optimizer name to lowercase for consistent matching
            optimizer_name = self.config.params_optimizer.strip().upper()
            optimizer = None

            # Select optimizer based on configuration
            if optimizer_name == "SGD":
                optimizer = tf.keras.optimizers.SGD(learning_rate=self.config.params_learning_rate)

            elif optimizer_name == "RMSPROP":
                optimizer = tf.keras.optimizers.RMSprop(learning_rate=self.config.params_learning_rate) 

            else:
                # Default to Adam if unsupported optimizer name is provided
                if optimizer_name != "ADAM":
                    logger.info(f"Unsupported optimizer name {optimizer_name} provided. Falling back to 'Adam'.")
                    optimizer_name = "ADAM"

                optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.params_learning_rate)

            logger.info(f"Optimizer '{optimizer_name}' initialized and returned.")
            return optimizer
        
        except Exception as exception_error:
            logger.error(f"Unexpected error while loading optimizer: {exception_error}")
            raise


    @staticmethod
    def _count_images_in_directory(directory_path: Union[str, Path]) -> int:
        """
        Counts the total number of image files in a directory and its subfolders.

        Args:
        - directory_path (str or Path): Path to the dataset root (e.g., train or valid)

        Returns:
        - int: Total number of images found
        """
        try:
            directory_path = Path(directory_path)
            total_images = 0

            if not directory_path.exists():
                logger.error(f"Could not find the path {directory_path}")
                raise FileNotFoundError(f"Could not find the path {directory_path}")
            
            for _, _, files in os.walk(directory_path):
                total_images += len([f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))])

            if total_images == 0:
                logger.error(f"No images found in {directory_path}")
                raise ValueError(f"No images found in {directory_path}")       

            return total_images

        except Exception as exception_error:
            logger.error(f"Unexpected error while counting images in directory: {exception_error}")
            raise

    
    @staticmethod
    def _convert_paths_to_str(obj: dict) -> dict:
        """
        Recursively convert Path objects in a nested dictionary to strings.
        """
        output = {}
        for key, value in obj.items():
            if isinstance(value, Path):
                output[key] = str(value)
            elif isinstance(value, dict):
                output[key] = ModelTraining._convert_paths_to_str(value)
            else:
                output[key] = value
        return output
    