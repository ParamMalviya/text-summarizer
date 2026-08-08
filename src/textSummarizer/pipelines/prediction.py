import sys
from transformers import AutoTokenizer, pipeline

from textSummarizer.logger import logger
from textSummarizer.exception import CustomException
from textSummarizer.config.configuration import ConfigurationManager


class PredictionPipeline:
    def __init__(self):
        # build the heavy stuff ONCE here, not on every predict() call.
        # loading the ~2.2GB model is the slow bit, so I pay it a single time
        # when this object is created, then reuse self.pipe for every request after
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}

        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        self.pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)

    def predict(self, text):
        try:
            output = self.pipe(text, **self.gen_kwargs)[0]["summary_text"]
            output = output.replace("<n>", " ")   # pegasus uses <n> for newlines, swap them for spaces
            logger.info("Prediction generated successfully")
            return output
        except Exception as e:
            raise CustomException(e, sys) from e