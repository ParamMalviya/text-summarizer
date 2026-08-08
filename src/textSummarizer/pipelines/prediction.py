import sys
import logging
from transformers import AutoTokenizer, pipeline

from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
            gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

            pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

            output = pipe(text, **gen_kwargs)[0]["summary_text"]
            output = output.replace("<n>", " ")

            logging.info("Prediction generated successfully")
            return output
        except Exception as e:
            raise CustomException(e, sys) from e