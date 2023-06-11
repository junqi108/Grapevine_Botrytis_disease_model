from os.path import join as path_join

# Path
BASE_DIR = path_join("..", "..")

# Data
DATA_DIR = "data"
DATA_FILE = path_join("..", BASE_DIR, DATA_DIR, "sauvignon_blanc_severity_calc.csv")

# Config
CONFIG_FILE = "config.yaml"
CONFIG_PATH = path_join(BASE_DIR, CONFIG_FILE)