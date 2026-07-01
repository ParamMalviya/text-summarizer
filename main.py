import sys
import logging

from textSummarizer import logger
from textSummarizer.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.exception import CustomException

stage_name = "Data Ingestion Stage"

try:
    logging.info(f">>>>>> Stage {stage_name} started <<<<<<")

    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()

    logging.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise CustomException(e,sys)