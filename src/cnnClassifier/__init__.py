"""
CNN Classifier Initialization Module

Constructs a logger to track issues during deployment.
"""

import os
import sys
import logging

# Log message format
LOG_FORMAT = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_logs.log")

os.makedirs(LOG_FILEPATH, exist_ok=True)

# Configure the logging system
logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,

    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

# Named logger to be reused across the entire cnnClassifer package
logger = logging.getLogger("cnnClassifierLogger")
