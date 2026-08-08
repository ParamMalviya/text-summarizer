# C:\Users\param\projects\text-summarizer\main.py  (full file — wraps everything below the imports in the standard guard)
import sys

from textSummarizer.logger import setup_logging, logger
from textSummarizer.exception import CustomException
from textSummarizer.pipelines.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipelines.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipelines.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipelines.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipelines.stage_05_model_evaluation import ModelEvaluationTrainingPipeline


if __name__ == "__main__":
    setup_logging()   # set up logging once, right at the start
    
    STAGE_NAME = "Data Ingestion Stage"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_ingestion = DataIngestionTrainingPipeline()
        data_ingestion.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys) from e

    STAGE_NAME = "Data Validation Stage"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_validation = DataValidationTrainingPipeline()
        data_validation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys) from e

    STAGE_NAME = "Data Transformation Stage"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        data_transformation = DataTransformationTrainingPipeline()
        data_transformation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys) from e

    STAGE_NAME = "Model Trainer stage"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        model_trainer = ModelTrainerTrainingPipeline()
        model_trainer.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys) from e

    STAGE_NAME = "Model Evaluation stage"
    try:
        logger.info(f">>>>>> Stage {STAGE_NAME} started <<<<<<")
        model_evaluation = ModelEvaluationTrainingPipeline()
        model_evaluation.main()
        logger.info(f">>>>>> Stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        raise CustomException(e, sys) from e