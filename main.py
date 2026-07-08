import sys
import logging

from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipelines.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipelines.stage_04_model_trainer import ModelTrainerTrainingPipeline




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


stage_name = "Data Transformation Stage"

try:
    logging.info(f">>>>>> Stage {stage_name} started <<<<<<")

    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.main()

    logging.info(f">>>>>> Stage {stage_name} completed <<<<<<\n\nx==========x")

except Exception as e:
    raise CustomException(e,sys)


STAGE_NAME = "Model Trainer stage"
try:
    logging.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainerTrainingPipeline()
    model_trainer.main()
    logging.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    raise CustomException(e, sys)
