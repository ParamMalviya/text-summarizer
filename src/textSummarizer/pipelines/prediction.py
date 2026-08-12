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
        # no_repeat_ngram_size blocks the model from repeating any 3-word phrase
        # it's already generated -- without this, beam search can get stuck
        # looping the same high-probability sentence until max_length cuts it off
        # (seen live on short/casual inputs, e.g. "I just started a new job last
        # week." repeating ~15 times). early_stopping lets beams finish once
        # they're done instead of being pushed to fill toward max_length.
        self.gen_kwargs = {
            "length_penalty": 0.8,
            "num_beams": 8,
            "max_length": 128,
            "no_repeat_ngram_size": 3,
            "early_stopping": True,
        }

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