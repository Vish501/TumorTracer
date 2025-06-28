import os
import yaml
import json
import joblib
import base64

from cnnClassifier import logger
from box.exceptions import BoxValueError
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any, Union


@ensure_annotations
def read_yaml(path: Path) -> ConfigBox:
    """
    Reads a YAML file from the given path and returns a ConfigBox object (like a dictionary with dot access).

    Parameters:
    - path (Path): Path to the YAML file.

    Returns:
    - ConfigBox: Parsed YAML content as a dot-accessible dictionary.
    """
    pass


@ensure_annotations
def create_directories(path_list: list, verbose=True) -> None:
    """
    Creates directories from the given list of paths if they don't already exist.

    Parameters:
    - path_list (list): List of directory paths to create.
    - verbose (bool): If True, prints log messages when directories are created.
    """
    pass


@ensure_annotations
def save_json(save_path: Path, data: Any) -> None:
    """
    Saves the provided data as a JSON file to the specified path.

    Parameters:
    - save_path (Path): Destination path for the JSON file.
    - data (Any): Serializable data (usually dict or list) to store in JSON format.
    """
    pass


@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file and returns its contents as a ConfigBox object.

    Parameters:
    - path (Path): Path to the JSON file.

    Returns:
    - ConfigBox: Parsed JSON content with dot-access support.
    """
    pass


@ensure_annotations
def save_bin(save_path: Path, data: Any) -> None:
    """
    Serializes and saves data to a binary (.bin or .pkl) file using joblib.

    Parameters:
    - save_path (Path): File path where the binary file will be saved.
    - data (Any): Any Python object to be serialized and stored.
    """
    pass

@ensure_annotations
def load_bin(path: Path) -> Any:
    """
    Loads and deserializes a binary file.

    Parameters:
    - path (Path): Path to the binary file.

    Returns:
    - Any: The original Python object that was serialized.
    """
    pass


@ensure_annotations
def get_kb_size(path: Path) -> str:
    """
    Returns the size of a file in kilobytes (KB).
    
    Parameters:
    - path (Path): Path to the file whose size is to be computed.

    Returns:
    - str: File size in kilobytes, as a string with 'KB' suffix.
    """
    pass


def decode_image_Base64(image_string: str, save_path: Union[str, Path]) -> None:
    """
    Decodes a base64 string and writes it as an image file.

    Parameters:
    - image_string (str): Base64-encoded image string.
    - save_path (str or Path): File path where the image should be saved.
    """
    pass


def encode_image_Base64(path: Union[str, Path]) -> str:
    """
    Reads an image file and returns its base64-encoded content as a UTF-8 string.

    Parameters:
    - image_path (str or Path): Path to the image file.

    Returns:
    - str: Base64-encoded image content as a string.
    """
    pass
