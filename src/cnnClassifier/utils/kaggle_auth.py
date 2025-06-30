import os
import json

from cnnClassifier.utils.common import create_directories, save_json
from pathlib import Path

def setup_kaggle_auth_from_secret(secret_env_var: str = "KAGGLE_JSON") -> None:
    """
    Sets up Kaggle API authentication using a secret stored in an environment variable.

    Parameters:
    - secret_env_var (str): The name of the environment variable that contains
                            the Kaggle credentials as a JSON string.

    Raises:
    - ValueError: If the environment variable is missing or contains invalid JSON.
    - Exception: For any other unhandled errors.
    """
    # Read from environment variable (injected from Codespaces secret)
    kaggle_json_str = os.getenv(secret_env_var)

    if kaggle_json_str is None:
        raise ValueError(f"{secret_env_var} secret not found.")
    
    try:
        # Validate it's a proper JSON
        kaggle_json_data = json.loads(kaggle_json_str)
    except json.JSONDecodeError as exception:
        raise ValueError(f"{secret_env_var} does not contain valid JSON: {exception}")

    # Setting directory path
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json_path = kaggle_dir / "kaggle.json"

    try:
        create_directories([kaggle_dir])
        save_json(kaggle_json_path, kaggle_json_data)

        # Set permissions
        os.chmod(kaggle_json_path, 0o600)

        # Set the environment variable explicitly for kaggle to pick up
        os.environ["KAGGLE_CONFIG_DIR"] = str(kaggle_dir)
    
    except Exception as exception:
        raise exception
    