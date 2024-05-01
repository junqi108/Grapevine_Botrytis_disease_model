from os.path import join as path_join

# Path
BASE_DIR = path_join("..", "..")

# Data
DATA_DIR = "data"
DATA_FILE = path_join(BASE_DIR, DATA_DIR, "sauvignon_blanc_severity_calc.csv")
OUT_DIR = path_join(BASE_DIR, DATA_DIR)

# Grapevine
GRAPEVINE_PARAMS_FILE = path_join("..", BASE_DIR, DATA_DIR, "grape_params.csv")
GRAPEVINE_DATA_FILE = path_join("..", BASE_DIR, DATA_DIR, "grape_data.csv")
GRAPEVINE_PARAMS_TRAINING = path_join("..", BASE_DIR, DATA_DIR, "training_grapevine_params.csv")
GRAPEVINE_PARAMS_TESTING = path_join("..", BASE_DIR, DATA_DIR, "testing_grapevine_params.csv")
GRAPEVINE_DATA_TRAINING = path_join("..", BASE_DIR, DATA_DIR, "training_grapevine_data.csv")
GRAPEVINE_DATA_TESTING = path_join("..", BASE_DIR, DATA_DIR, "testing_grapevine_data.csv")
GRAPEVINE_RESP = ["malicConcentration_fruit_scaled", "Lphloem_scaled", "Fresh weight", "Dry weight"]

# Config
CONFIG_FILE = "config.yaml"
CONFIG_PATH = path_join(BASE_DIR, CONFIG_FILE)

# Gaussian Process
GP_MODEL_FILE = path_join(OUT_DIR, "gp.pth")
DKL_MODEL_FILE = path_join(OUT_DIR, "dkl.pth")

# Optimisation
DKL_MODEL_FILE = path_join(OUT_DIR, "opt.pkl")
