from pathlib import Path

CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# A tiny file holding fixed values that never change while the program runs — here, just the file paths to config.yaml and params.yaml. ConfigurationManager needs these two paths to find the files.