import config
from utils import create_folder_if_not_exists

import logging
from pathlib import Path

def initialize_logger(log_file: str = Path(config.LOGS_DIR, "info.log")):
    create_folder_if_not_exists(config.LOGS_DIR)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    file_handler = logging.FileHandler(filename=log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    
    return logger
