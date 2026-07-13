# C:\Users\param\projects\text-summarizer\src\textSummarizer\components\model_trainer.py
import os
import sys
import torch
import logging

from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForSeq2Seq, AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk

from textSummarizer import logger
from textSummarizer.exception import CustomException
from textSummarizer.entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config : ModelTrainerConfig):
        self.config = config

    def train(self):
        try:
            save_path = os.path.join(self.config.root_dir, "pegasus-samsum-model")
            if os.path.exists(save_path):
                logging.info(f"Trained model already exists at ({save_path}), skipping training")
                return

            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
            model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)
            seq2seq_data_collator = DataCollatorForSeq2Seq( tokenizer, model=model_pegasus)

            dataset_samsum_pt = load_from_disk(self.config.data_path)
            dataset_samsum_pt["train"] = dataset_samsum_pt["train"].select(range(200))
            dataset_samsum_pt["validation"] = dataset_samsum_pt["validation"].select(range(50))

            trainer_args = TrainingArguments(
                output_dir = self.config.root_dir,
                num_train_epochs = self.config.num_train_epochs,
                warmup_steps=self.config.warmup_steps,
                per_device_train_batch_size=self.config.per_device_train_batch_size,
                per_device_eval_batch_size=self.config.per_device_train_batch_size,
                weight_decay=self.config.weight_decay,
                logging_steps=self.config.logging_steps,
                eval_strategy=self.config.evaluation_strategy,
                eval_steps=self.config.eval_steps,
                save_steps=self.config.save_steps,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                fp16=self.config.fp16
                )
            
            trainer = Trainer(
                model=model_pegasus,
                args=trainer_args,
                tokenizer=tokenizer,
                data_collator=seq2seq_data_collator,
                train_dataset=dataset_samsum_pt['train'],
                eval_dataset = dataset_samsum_pt['validation']
            )

            trainer.train()

            model_pegasus.save_pretrained(
                os.path.join(self.config.root_dir, "pegasus-samsum-model")
            )
            tokenizer.save_pretrained(
                os.path.join(self.config.root_dir, "tokenizer")
            )
            logging.info("Model training completed and saved successfully")
        except Exception as e:
            raise CustomException(e,sys)