import os
import logging

from pathlib import Path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# List of files and directories that need to be created or amended with new files
list_of_files = [
    ".github/workflows/.gitkeep",
    "src/cnnClassifier/__init__.py",
    "src/cnnClassifier/components/__init__.py",
    "src/cnnClassifier/utils/__init__.py",
    "src/cnnClassifier/config/__init__.py",
    "src/cnnClassifier/config/configurations.py",
    "src/cnnClassifier/pipeline/__init__.py",
    "src/cnnClassifier/entity/__init__.py",
    "src/cnnClassifier/constants/__init__.py",
    "config/config.yaml",
    "dvc.yaml",
    "params.yaml",
    "pyproject.toml",
    "requirements.txt",
    "research/trials.ipynb",
    "templates/index.html"
]

# Iterate through each file path
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, _ = os.path.split(filepath)
    
    # Create the directory if it doesn't already exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir}')
    
    # Create the file if it doesn't exist or is currently empty
    if not filepath.exists() or filepath.stat().st_size == 0:
        filepath.touch()
        logging.info(f"Created empty file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")
