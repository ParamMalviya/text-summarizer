import sys

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.exception import CustomException


class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        model_evaluation.evaluate()


if __name__ == "__main__":
    try:
        obj = ModelEvaluationTrainingPipeline()
        obj.main()
    except Exception as e:
        raise CustomException(e, sys) from e