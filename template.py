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
    "setup.py",
    "requirements.txt",
    "research/trials.ipynb",
    "templates/index.html"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)
    
    # If the directory does not exist create it
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f'Creating directory: {filedir}')
    
    # If the file within the directory does not exist 
    # or the size of the file is 0, then create the file
    # Else if the file already exists within the specified path, log that it already exists
    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as file:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filepath} already exists")
        