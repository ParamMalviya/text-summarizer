import sys
import logging

from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipelines.stage_02_data_validation import DataValidationTrainingPipeline


stage_name = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> Stage {stage_name} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logging.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise CustomException(e,sys)


stage_name = "Data Validation Stage"

try:
    logging.info(f">>>>>> Stage {stage_name} started <<<<<<")

    data_validation = DataValidationTrainingPipeline()
    data_validation.main()

    logging.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise CustomException(e,sys)
