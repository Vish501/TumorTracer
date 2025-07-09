import random
import numpy as np
import tensorflow as tf

from typing import Union
from tensorflow.keras.preprocessing import image  # type:ignore
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

from cnnClassifier.config.configurations import PredictionConfig
from cnnClassifier.utils.common import load_json
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

class Predictions:
	"""
	Wrapper for loading a trained model and generating predictions on input images.

	Responsibilities:
	- Load a trained Keras model and class indices.
	- Preprocess input images for inference.
	- Predict the class label for a given image.
	"""
	def __init__(self, config: PredictionConfig) -> None:
		self.config = config
		self.model = self._load_model()
		self.classes = self._get_classes()

		# Set random seeds
		seed = self.config.params_seed if hasattr(self.config, "params_seed") else 1234
		tf.random.set_seed(seed)
		np.random.seed(seed)
		random.seed(seed)


	def predict(self, image_path: Union[str, Path], verbose: bool = True) -> str:
		"""
		Predict the class label for a single image.

		Args:
        - image_path (str | Path): Path to the image file.

		Returns:
        - str: Predicted class label.
		"""
		try:
			prediction_label, _ = self.predict_with_confidence(image_path=image_path, verbose=verbose)
			return prediction_label

		except Exception as exception_error:
			logger.error(f"Unexpected while trying to predict: {exception_error}")
			raise 

     
	def predict_with_confidence(self, image_path: Union[str, Path], verbose: bool = True) -> tuple[str, float]:
		"""
		Predict the class label for a single image, and it confidence

		Args:
        - image_path (str | Path): Path to the image file.

		Returns:
        - str: Predicted class label.
		- float: Confidence in prediction.
		"""
		try:
			image_path = Path(image_path)
			if not image_path.exists():
				raise FileNotFoundError(f"Image file not found: {image_path}")

			image_object = image.load_img(image_path, target_size=self.config.params_image_size[0:2])
			image_array = image.img_to_array(image_object) / self.config.params_normalization

			# Add batch dimension
			image_dims = np.expand_dims(image_array, axis=0)

			# Predicting image
			prediction = self.model.predict(image_dims, verbose=0)
			prediction_idx = np.argmax(prediction, axis=1)[0]
			prediction_label = self.classes[prediction_idx]

			confidence = prediction[0][prediction_idx]

			if verbose:
				logger.info(f"Predicted '{prediction_label}' for image: {image_path}")

			return prediction_label, confidence
		
		except Exception as exception_error:
			logger.error(f"Unexpected while trying to predict: {exception_error}")
			raise 


	def evaluate(self, image_directory: Union[str, Path], verbose: bool = False) -> dict[str, float]:
		"""
		Evaluates model performance on a labeled image dataset organized in subfolders by class.

		Args:
		- image_directory (str | Path): Root directory containing subfolders (each representing a class).

		Returns:
		- dict[str, float]: Dictionary containing per-class accuracy and overall accuracy as "OVERALL".
		"""
		image_directory_path = Path(image_directory)

		if not image_directory_path.exists():
			logger.error(f"Provided dataset directory does not exist: {image_directory_path}")
			raise FileNotFoundError(f"Provided dataset directory does not exist: {image_directory_path}")
		
		correct_per_class = defaultdict(int)
		total_per_class = defaultdict(int)

		for subfolder in tqdm(list(image_directory_path.iterdir()), desc=f"Evaluating Subfolders", leave=True):
			if not subfolder.is_dir():
				continue
			
			true_label = subfolder.name

			for image in list(subfolder.glob("*")):
				if not image.suffix.casefold() in [".jpg", ".jpeg", ".png"]:
					continue

				try:
					predicted_label = self.predict(image_path=image, verbose=verbose)
					if predicted_label.casefold() == true_label.casefold():
						correct_per_class[true_label] += 1
					total_per_class[true_label] += 1

				except Exception as exception_error:
					logger.warning(f"Skipping {image.name}: {exception_error}")
					continue

		result = {}

		# Overall accuracy
		total_correct = sum(correct_per_class.values())
		total_images = sum(total_per_class.values())
		result["Overall"] = total_correct / total_images if total_images > 0 else 0.0

		# Per-class accuracy
		for class_name in total_per_class:
			result[class_name] = correct_per_class[class_name] / total_per_class[class_name] if total_images > 0 else 0.0

		logger.info(f"Evaluation Summary: {str(result)}")
		return result


	def _load_model(self) -> tf.keras.Model:
		"""
		Loads the trained model from disk.

		Returns:
		- tf.keras.Model: Loaded Keras model.
		"""
		model_path = Path(self.config.trained_model_path)

		if not model_path.exists():
			logger.error(f"Could not find model at {model_path}.")
			raise FileNotFoundError(f"Could not find model at {model_path}.")
		
		try:
			output_model = tf.keras.models.load_model(model_path)
			logger.info(f"Successfully loaded the base model from {model_path}.")
			return output_model

		except Exception as exception_error:
			logger.error(f"Unexpected error while loading the update base model at {model_path}: {exception_error}")
			raise 
			

	def _get_classes(self) -> dict[int, str]:
		"""
		Loads and inverts class indices from JSON file.

        Returns:
		- dict: Mapping from index to class label.
		"""
		try:
			classes = load_json(self.config.class_indices_path)
			return {value: key for key, value in classes.items()}
		
		except Exception as exception_error:
			logger.error(f"Unexpected error while loading class indicies: {exception_error}")
			raise 
		