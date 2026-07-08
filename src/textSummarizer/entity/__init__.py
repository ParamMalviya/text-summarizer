from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir        : Path
    source_URL      : str
    local_data_file : Path
    unzip_dir       : Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir           : Path
    STATUS_FILE        : str
    ALL_REQUIRED_FILES : list
    data_path          : Path

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir       : Path
    data_path      : Path
    tokenizer_name : str

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir        : Path
    data_path       : Path
    model_ckpt      : str
    num_train_epochs : int
    warmup_steps    : int
    weight_decay    : float
    per_device_train_batch_size : int
    logging_steps   : int
    evaluation_strategy : str
    eval_steps      : int
    save_steps      : int
    gradient_accumulation_steps : int
    fp16: bool


    

# A blueprint file. It defines the exact "shape" of settings each component receives — a labeled container with specific named slots, each with a type.