# C:\Users\param\projects\text-summarizer\src\textSummarizer\components\data_transformation.py
import os
import sys
from textSummarizer.logger import logger
from textSummarizer.exception import CustomException
from textSummarizer.entity import DataTransformationConfig
from transformers import AutoTokenizer
from datasets import load_from_disk


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        input_encodings = self.tokenizer(
            example_batch['dialogue'],
            max_length=1024,
            truncation=True
        )
        target_encodings = self.tokenizer(
            text_target=example_batch['summary'],
            max_length=128,
            truncation=True
        )
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        try:
            save_path = os.path.join(self.config.root_dir, "samsum_dataset")
            if os.path.exists(save_path):
                logger.info(f"Transformed dataset already exists at ({save_path}), skipping transformation")
                return
            dataset_samsum = load_from_disk(self.config.data_path)
            dataset_samsum_transformed = dataset_samsum.map(
                self.convert_examples_to_features,
                batched=True
            )
            dataset_samsum_transformed.save_to_disk(save_path)
            logger.info("Data transformation completed successfully")
        except Exception as e:
            raise CustomException(e, sys) from e