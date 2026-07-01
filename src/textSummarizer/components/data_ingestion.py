import os
from pathlib import Path
import urllib.request as request
import logging
import zipfile

from textSummarizer import logger
from textSummarizer.utils.common import get_size
from textSummarizer.entity import DataIngestionConfig

class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                filename = self.config.local_data_file,
                url = self.config.source_URL
            )
            logging.info(f"{filename} downloaded with the following info: \n{headers}")
        else:
            logging.info(f"File already exists of size:{get_size(Path(self.config.local_data_file))}")

    def extract_zip_file(self):
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)

        with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
            zip_ref.extractall(unzip_path)