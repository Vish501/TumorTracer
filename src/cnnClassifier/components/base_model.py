import tensorflow as tf
from pathlib import Path
from cnnClassifier.entity.config_entity import BaseModelConfig
from cnnClassifier.utils.common import create_directories
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

class BaseModelConstruction:
    """
    Handles the retrieval and customization of the base model (VGG16) for transfer learning.

    Responsibilities:
    - Downloads a pretrained VGG16 model
    - Appends custom dense layers for classification (if called)
    - Saves both base and updated models

    Attributes:
    - config (BaseModelConfig): Configuration object with model paths and hyperparameters

    Public Methods:
    - get_model(): Downloads and saves the base VGG16 model
    - updated_model(): Adds custom dense layers and saves the modified model

    Internal Methods:
    - _prepare_model(): Modifies the base model for the current classification task
    - _save_model(): Saves the model to the specified path
    """
    def __init__(self, config: BaseModelConfig) -> None:
        self.config = config
        self.model = None
        self.enhanced_model = None


    def get_model(self) -> None:
        """
        Downloads the pretrained the pretrained VGG16 model and saves it.
        """
        try:
            logger.info(f"Downloading base VGG16 model...")
            self.model = tf.keras.applications.vgg16.VGG16(
                input_shape=self.config.params_image_size,
                weights=self.config.params_weights,
                include_top=self.config.params_include_top,
            )

            logger.info(f"Successfully downloaded VGG16 base model.")
            self._save_model(save_path=self.config.model_path, model=self.model)
        
        except Exception as exception_error:
            logger.error(f"Unexpected error while downloading the base model: {exception_error}")
            raise 


    def updated_model(self) -> None:
        """
        Updates the downloaded model with custom dense layers and saves it.
        """
        if self.model is None:
            logger.error("Base model not found. Run get_model() before calling updated_model().")
            raise ValueError("Base model not found. Run get_model() before calling updated_model().")

        try: 
            logger.info("Preparing the enhanced model with custom dense layers...")

            self.enchanced_model = self._prepare_model(
                model=self.model,
                classes=self.config.params_classes,
                freeze_all=True,
                freeze_till=None,
                learning_rate=self.config.params_learning_rate,
            )

            logger.info("Successfully created the enhanced model with custom dense layers.")
            self._save_model(save_path=self.config.updated_model_path, model=self.enchanced_model)

        except Exception as exception_error:
            logger.error(f"Unexpected error while creating the enchanced model: {exception_error}")
            raise 

    @staticmethod
    def _prepare_model(model: tf.keras.Model, classes: int, freeze_all: bool, freeze_till: int, learning_rate: float) -> tf.keras.Model:
        """
        Customizes the base model by freezing layers and adding classification head.

        Parameters:
        - model (tf.keras.Model): Pretrained base model
        - classes (int): Number of output classes
        - freeze_all (bool): Whether to freeze all layers
        - freeze_till (int): Number of layers from the end to remain trainable
        - learning_rate (float): Learning rate for optimizer

        Returns:
        - tf.keras.Model: Fully compiled transfer learning model
        """
        # Make all layers trainable first (so we can selectively freeze them)
        model.trainable = True

        # If freeze_all is True, freeze the entire base model
        if freeze_all:
            for layer in model.layers:
                layer.trainable = False
        # Optionally freeze up to a certain layer, allowing fine-tuning of last few
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                layer.trainable = False

        # Converts feature maps to a 1D vector
        flatten_model = tf.keras.layers.Flatten()(model.output)
        
        # Adds output neurons for each class.
        prediction = tf.keras.layers.Dense(units=classes, activation="softmax")(flatten_model)

        # Wraps the base model and the new classification head into one Model
        full_model = tf.keras.models.Model(inputs=model.input, outputs=prediction)
        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    
    
    def _save_model(self, save_path: Path, model: tf.keras.Model) -> None:
        """
        Saves a given model to the specified path.
        """
        try:
            create_directories([save_path.parent])
            model.save(save_path)
            logger.info(f"Model saved at: {save_path}")
        
        except Exception as exception_error:
            logger.error(f"Unexpected error while saving the model at {save_path}: {exception_error}")
            raise
