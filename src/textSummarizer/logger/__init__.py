import os
import sys
import logging
from datetime import datetime

logging_str = "[%(asctime)s: %(levelname)s: %(lineno)d: %(module)s: %(message)s]"

log_dir = "logs"
log_filename = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_filepath = os.path.join(log_dir, log_filename)
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level = logging.INFO,
    format= logging_str,
    handlers= [
        logging.FileHandler(log_filepath, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)