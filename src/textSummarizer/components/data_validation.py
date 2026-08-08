# C:\Users\param\projects\text-summarizer\src\textSummarizer\components\data_validation.py
import os
import sys
import logging
from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.entity import DataValidationConfig


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            all_files = os.listdir(self.config.data_path)
            missing_files = [
                f for f in self.config.ALL_REQUIRED_FILES
                if f not in all_files
            ]
            if missing_files:
                validation_status = False
            else:
                validation_status = True

            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise CustomException(e, sys) from e