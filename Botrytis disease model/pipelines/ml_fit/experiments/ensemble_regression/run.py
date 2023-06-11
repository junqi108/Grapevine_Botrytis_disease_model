##############################################
# Imports

import os
from os.path import join as path_join
from matplotlib import pyplot as plt
import sys

base_dir = "../.."
sys.path.insert(0, os.path.abspath(base_dir))

from ml_fit_lib.config import get_config
from ml_fit_lib.constants import *
from ml_fit_lib.data import Data
from ml_fit_lib.experiment import setup_experiment, regression_experiment
from ml_fit_lib.mlops import MLFlowPipeline

##############################################
# Constants

if __name__ == "__main__":
    CONFIG = get_config(CONFIG_PATH)

    PIPELINE = MLFlowPipeline()
    RUN_DISTRIBUTED = CONFIG.get("run_distributed")

    DATA = Data()
    
##############################################
# Main

def main():
    tmp_dir = PIPELINE.init(CONFIG)
    df = DATA.read_csv(DATA_FILE)

    setup_experiment(CONFIG, df)
    
    final_model = regression_experiment(CONFIG, RUN_DISTRIBUTED)

    config_out = path_join(tmp_dir, CONFIG_FILE)
    CONFIG.export(config_out)
    PIPELINE.log_artifact(config_out)

if __name__ == "__main__":
    main()