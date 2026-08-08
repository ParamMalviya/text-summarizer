import os
import sys
import logging
import yaml
from pathlib import Path
from ensure import ensure_annotations
from textSummarizer.exception import CustomException

@ensure_annotations
def read_yaml(path_to_yaml : Path) -> dict:
    """
    Args:
        path_to_yaml(Path) : Path to the yaml file

    Raises:
        CustomException : If the yaml file is empty or cannot be read

    Returns:
        dict : yaml content as a dictionary
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError("Yaml file is empty")
            logging.info(f"yaml file ({path_to_yaml}) is loaded successfully")
            return content

    except Exception as e:
        raise CustomException(e, sys) from e
    
@ensure_annotations
def create_directories(path_to_directories : list, verbose = True):
    """Create a list of directories.

    Args:
        path_to_directories (list): list of paths of directories to create.
        verbose (bool, optional): log each directory created. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logging.info(f"Created directory at ({path})")

@ensure_annotations
def get_size(path: Path) -> str:
    """Get size in KB.

    Args:
        path (Path): path of the file.

    Returns:
        str: size in KB.
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"