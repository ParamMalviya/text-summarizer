import os
from pathlib import Path

project_name = "textSummarizer"

list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/logger/__init__.py",
    "requirements.txt",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",
    f"src/{project_name}/exception/__init__.py",
    "setup.py",
    "config/config.yaml",
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/constants/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/config/__init__.py",
    "research/01_data_ingestion.ipynb",
    "params.yaml",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/pipelines/stage_01_data_ingestion.py",
    "main.py",
    "research/02_data_validation.ipynb",
    f"src/{project_name}/components/data_validation.y",
    f"src/{project_name}/pipelines/stage_02_data_validation.py"
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

    if not os.path.exists(filepath):
        with open(filepath, "w") as f:
            pass
    else:
        print("Already exist")