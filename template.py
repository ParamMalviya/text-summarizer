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
    "setup.py"
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