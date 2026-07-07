# C:\Users\param\projects\text-summarizer\src\textSummarizer\components\model_trainer.py
import os
import sys
import logging
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq
from transformers import create_optimizer
from datasets import load_from_disk
from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            model = TFAutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_ckpt,
                from_pt=True
            )

            data_collator = DataCollatorForSeq2Seq(
                tokenizer,
                model=model,
                return_tensors="tf",
                padding=True
            )

            dataset = load_from_disk(self.config.data_path)

            train_dataset = dataset["train"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "labels"],
                shuffle=True,
                batch_size=self.config.batch_size,
                collate_fn=data_collator
            )
            eval_dataset = dataset["validation"].to_tf_dataset(
                columns=["input_ids", "attention_mask", "labels"],
                shuffle=False,
                batch_size=self.config.batch_size,
                collate_fn=data_collator
            )

            num_train_steps = len(train_dataset) * int(self.config.num_train_epochs)

            optimizer,_ = create_optimizer(
            init_lr=float(self.config.learning_rate),
            num_warmup_steps=int(self.config.warmup_steps),
            num_train_steps=int(num_train_steps),
            weight_decay_rate=float(self.config.weight_decay)                
            )

            model.compile(optimizer=optimizer)

            model.fit(
                train_dataset,
                validation_data=eval_dataset,
                epochs=self.config.num_train_epochs
            )

            model.save_pretrained(
                os.path.join(self.config.root_dir, "pegasus-samsum-model")
            )
            tokenizer.save_pretrained(
                os.path.join(self.config.root_dir, "tokenizer")
            )
            logging.info("Model training completed and saved successfully")

        except Exception as e:
            raise CustomException(e, sys)