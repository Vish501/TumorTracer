import re
import shutil

from pathlib import Path

from cnnClassifier.config.configurations import ModelSelectorConfig
from cnnClassifier.utils.common import create_directories
from cnnClassifier import get_logger

# Initializing the logger
logger = get_logger()

class ModelSelection:
	def __init__(self, config: ModelSelectorConfig) -> None:
		"""
		Initializes the ModelSelection class with configuration.

		Validates the regex pattern used for parsing model filenames.
        """
		self.config = config
		self.best_model_path = None
		self.class_indices_path = None
		self.best_vacc = -1.0

		self._validate_regex_pattern()
		self.select_model()
		self.store_best_model()
        

	def _validate_regex_pattern(self) -> None:
		"""
		Validates the regex pattern with a test filename.
		"""
		try:
			test_filename = "model_e02_acc0.6297_vacc0.7083.keras"
			regex_match = re.match(self.config.filename_regex, test_filename)

			assert regex_match is not None, f"Regex failed to match filename: {test_filename}"
			assert regex_match.group(1) == "0.7083", f"Expected '0.7083', got '{regex_match.group(1)}'"

			logger.info(f"Regex loaded and validated successfully.")

		except AssertionError as exception_error:
			logger.error(f"Regex validation failed: {exception_error}")

		except re.error as exception_error:
			logger.error(f"Invalid regex pattern provided: {exception_error}")

		except Exception as exception_error:
			logger.error(f"Unexpected error during regex validation: {exception_error}")


	def select_model(self) -> None:
		"""
		Iterates over checkpoint folders to find the model with highest validation accuracy (vacc).
		"""
		source_directory = Path(self.config.source_dir)

		if not source_directory.exists():
			logger.error(f"Source directory does not exist at: {source_directory}")
			return

		try:
			# Looping all Checkpoint_* folders in soruce-directory
			for subfolder in source_directory.glob("Checkpoint_*"):
				if not subfolder.is_dir():
					continue
			
				# Search for model files
				for model_file in subfolder.glob("model_*.keras"):
					regex_match = re.match(self.config.filename_regex, model_file.name)

					if not regex_match:
						continue
						
					try:
						vacc = float(regex_match.group(1))
					except (IndexError, ValueError):
						logger.warning(f"Skipping file (invalid vacc group): {model_file.name}")
						continue
					
					# Check for best validation accuracy
					if vacc > self.best_vacc:
						self.best_model_path = model_file
						self.class_indices_path = subfolder / "class_indices.json"
						self.best_vacc = vacc

			if not self.best_model_path or not self.class_indices_path:
				logger.error(f"No suitable model found during selection at: {source_directory}")

		except Exception as exception_error:
			logger.error(f"Unexpected error during model selection: {exception_error}")


	def store_best_model(self) -> None:
		"""
		 Stores the selected best model and its class indices file into the destination directory.
		"""
		if not self.best_model_path or not self.class_indices_path:
			logger.error(f"Unable to find model or indices. Please run select_model() before store_best_model().")
			return

		try:
			# Create directory to store the models
			destination_directory = Path(self.config.destination_dir)
			create_directories([destination_directory])

			# Copy best model and class indices file to final location
			shutil.copy(self.best_model_path, self.config.model_path)
			shutil.copy(self.class_indices_path, self.config.indices_path)

			logger.info(f"Best model (vacc={self.best_vacc:.4f}) copied to: {destination_directory}")

		except Exception as exception_error:
			logger.error(f"Unexpected error during saving best model: {exception_error}")
			